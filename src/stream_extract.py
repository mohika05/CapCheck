#!/usr/bin/env python3
"""
Extract CLIP ViT-B/32 + DCT features for Track 5.

The extracted classifier target is binary:
    0 = authentic
    1 = AIGC / AI-edited

SID raw label 2 (tampered) is mapped to binary AIGC=1.

Source caps are optional:
- local smoke tests use small caps and stop early;
- TC1 uses max_per_class=None, so every available SID/CIFAKE example is read.

WildFake must NOT appear in config.TRAIN_SOURCES. It is evaluated only after
training by predict.py.

Each output shard stores:
    clip   (N, 512) float32
    dct    (N, 64)  float32
    label  (N,)     int64    binary target
    group  (N,)     int64    source-image id shared by augmented copies
    aug    (N,)     int64    index into AUG_NAMES (0 = clean)
    src    (N,)     int64    index into config.TRAIN_SOURCES

A fresh extraction is required after changing preprocessing/label mapping.
By default this script refuses to append to an existing shard directory.
Use --overwrite to explicitly replace old shards.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C

ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0)

# ─────────────────────────────────────────────────────────────────────
# Track-5 transformations
# ─────────────────────────────────────────────────────────────────────

def _jpeg(rgb, quality):
    ok, enc = cv2.imencode(
        '.jpg', rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return rgb
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]


def _blur(rgb, sigma):
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(rgb, (k, k), sigma)


def _resize(rgb, scale):
    """Downscale and then upscale back to the original resolution."""
    h, w = rgb.shape[:2]
    small = cv2.resize(
        rgb,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _noise(rgb, sigma, rng):
    noise = rng.normal(0.0, sigma * 255.0, rgb.shape)
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jitter(rgb, amount, rng):
    # Independent brightness / contrast / saturation factors in [1-a, 1+a].
    brightness, contrast, saturation = 1.0 + rng.uniform(-amount, amount, 3)

    out = np.clip(rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    mean = out.mean()
    out = np.clip(
        (out.astype(np.float32) - mean) * contrast + mean, 0, 255
    ).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _center_crop(rgb, frac):
    h, w = rgb.shape[:2]
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return rgb[y0:y0 + ch, x0:x0 + cw]


AUGS = [
    ('clean',       lambda x, r: x),
    ('jpeg_90',     lambda x, r: _jpeg(x, 90)),
    ('jpeg_70',     lambda x, r: _jpeg(x, 70)),
    ('jpeg_50',     lambda x, r: _jpeg(x, 50)),
    ('jpeg_30',     lambda x, r: _jpeg(x, 30)),
    ('blur_0.5',    lambda x, r: _blur(x, 0.5)),
    ('blur_1.0',    lambda x, r: _blur(x, 1.0)),
    ('blur_2.0',    lambda x, r: _blur(x, 2.0)),
    ('resize_0.5',  lambda x, r: _resize(x, 0.5)),
    ('resize_0.25', lambda x, r: _resize(x, 0.25)),
    ('noise_0.02',  lambda x, r: _noise(x, 0.02, r)),
    ('noise_0.05',  lambda x, r: _noise(x, 0.05, r)),
    ('noise_0.10',  lambda x, r: _noise(x, 0.10, r)),
    ('jitter_0.2',  lambda x, r: _jitter(x, 0.2, r)),
    ('crop_0.8',    lambda x, r: _center_crop(x, 0.8)),
]
AUG_NAMES = [name for name, _ in AUGS]


def dct_feature(rgb, size=128):
    """Return the exact 64-D DCT feature used by both train and predict."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray, (size, size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    block = cv2.dct(gray)[:8, :8].flatten()
    ac = block[1:] / (np.abs(block[1:]).max() + 1e-8)
    dc = np.log1p(abs(block[0])) * np.sign(block[0])
    return np.concatenate([[dc], ac]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Label handling
# ─────────────────────────────────────────────────────────────────────

RAW_VALID_LABELS = {0, 1, 2}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

REAL_TOKENS = {
    'real', 'reals', 'real_images', '0_real', 'authentic', 'genuine',
    'natural', 'nature', 'pristine', 'original', 'non_aigc', 'nonaigc',
    'coco', 'val2017', 'coco_val2017', 'imagenet', 'openimages',
}
FAKE_TOKENS = {
    'fake', 'fakes', 'fake_images', '1_fake', 'ai', 'aigc',
    'synthetic', 'full_synthetic', 'fully_synthetic', 'generated',
    'ai_generated', 'gan', 'diffusion', 'dalle', 'dall_e', 'dalle3',
    'midjourney', 'stable_diffusion', 'sdxl', 'sd', 'flux',
    'kandinsky', 'imagen', 'firefly',
}
TAMPER_TOKENS = {
    'tampered', 'manipulated', 'edited', 'spliced', 'inpainted', 'forged'
}
SKIP_TOKENS = {
    'mask', 'masks', 'gt', 'groundtruth', 'ground_truth',
    'annotation', 'annotations', 'label', 'labels', 'seg',
}

STR_LABELS = {
    'real': 0,
    'authentic': 0,
    'natural': 0,
    'non_aigc': 0,
    'fake': 1,
    'ai': 1,
    'aigc': 1,
    'synthetic': 1,
    'full_synthetic': 1,
    'fully_synthetic': 1,
    'generated': 1,
    'tampered': 2,
    'manipulated': 2,
}


def _norm(token):
    return token.strip().lower().replace('-', '_').replace(' ', '_')


def infer_label_from_path(path, root):
    """Infer the raw 0/1/2 label from folders below a source root."""
    try:
        parts = Path(path).relative_to(root).parts[:-1]
    except ValueError:
        parts = Path(path).parts[:-1]

    for part in parts:
        token = _norm(part)
        if token in SKIP_TOKENS:
            return None
        if token in TAMPER_TOKENS:
            return 2
        if token in FAKE_TOKENS:
            return 1
        if token in REAL_TOKENS:
            return 0

    stem = _norm(Path(path).stem).split('_')[0]
    if stem in FAKE_TOKENS:
        return 1
    if stem in REAL_TOKENS:
        return 0
    return None


def label_from_value(raw):
    """Normalise an HF label value to raw integer label 0/1/2."""
    if isinstance(raw, str):
        return STR_LABELS.get(_norm(raw))
    if isinstance(raw, (bool, int, np.integer)):
        return int(raw)
    return None


def to_binary_label(raw):
    """Map Track-5 training labels to 0=authentic, 1=AIGC/AI-edited."""
    if raw == 0:
        return 0
    if raw == 1:
        return 1
    return None


# ─────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────

class FolderSource:
    kind = 'dir'

    def __init__(self, name, path, label=None, shuffle=True, **_):
        self.name = name
        self.root = Path(path).expanduser().resolve()
        self.forced = label
        self.shuffle = shuffle
        if not self.root.exists():
            raise SystemExit(f'[{name}] path does not exist: {self.root}')

    def n_shards(self):
        # FolderSource is explicitly split across DataLoader workers below.
        return 10 ** 6

    def _paths(self):
        for path in sorted(self.root.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                yield path

    def examples(self, wid, nw, seed=0):
        paths = list(self._paths())
        if self.shuffle:
            np.random.default_rng(seed).shuffle(paths)

        for path in paths[wid::max(1, nw)]:
            raw = (
                self.forced
                if self.forced is not None
                else infer_label_from_path(path, self.root)
            )
            if raw is None:
                continue
            try:
                with Image.open(path) as image:
                    yield image.convert('RGB').copy(), raw
            except Exception:
                continue

    def preview(self, limit=4000):
        counts, unlabelled, seen, skipped = {}, [], 0, 0
        for path in self._paths():
            raw = (
                self.forced
                if self.forced is not None
                else infer_label_from_path(path, self.root)
            )
            rel = path.relative_to(self.root)
            if raw is None:
                if any(_norm(x) in SKIP_TOKENS for x in rel.parts[:-1]):
                    skipped += 1
                    seen += 1
                    continue
                if len(unlabelled) < 5:
                    unlabelled.append(str(rel))
            counts[raw] = counts.get(raw, 0) + 1
            seen += 1
            if seen >= limit:
                break

        if skipped:
            counts['skipped'] = skipped
        return counts, unlabelled, seen


class HFSource:
    """Hugging Face streaming source.

    Important:
    Hugging Face IterableDataset is already aware of PyTorch DataLoader
    workers. When this dataset is iterated inside a worker, HF partitions its
    underlying shards for that worker automatically.

    Therefore we MUST NOT call ds.shard(num_shards=nw, index=wid) manually
    here. Doing both would shard SID twice and can reduce a 4-worker run to
    roughly one quarter of the intended dataset.
    """

    kind = 'hf'

    def __init__(
        self,
        name,
        repo,
        split='train',
        image_key='image',
        label_key='label',
        label=None,
        shuffle_buffer=10000,
        **_,
    ):
        self.name = name
        self.repo = repo
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.forced = label
        self.shuffle_buffer = shuffle_buffer
        self._ds = None

    def _load(self):
        if self._ds is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise SystemExit(
                    f'[{self.name}] requires datasets>=2.14'
                ) from exc
            self._ds = load_dataset(
                self.repo, split=self.split, streaming=True
            )
        return self._ds

    def n_shards(self):
        return int(getattr(self._load(), 'n_shards', 1) or 1)

    def examples(self, wid, nw, seed=0):
        ds = self._load()

        # Do NOT manually shard here.
        # HF IterableDataset performs DataLoader-worker partitioning itself.
        # Use the same shuffle seed in every worker so all workers share the
        # same deterministic global shuffle before HF assigns worker shards.
        ds = ds.shuffle(seed=seed, buffer_size=self.shuffle_buffer)

        for example in ds:
            raw = (
                self.forced
                if self.forced is not None
                else label_from_value(example.get(self.label_key))
            )
            image = example.get(self.image_key)
            if raw is None or image is None:
                continue
            yield image, raw

    def preview(self, limit=500):
        counts, seen = {}, 0
        for _, raw in self.examples(0, 1):
            counts[raw] = counts.get(raw, 0) + 1
            seen += 1
            if seen >= limit:
                break
        return counts, [], seen


def _make_source(entry):
    entry = dict(entry)
    kind = entry.pop('type')
    cap = entry.pop('max_per_class', C.MAX_PER_CLASS)
    n_aug = entry.pop('n_aug', C.N_AUG)

    if kind == 'dir':
        source = FolderSource(**entry)
    elif kind == 'hf':
        source = HFSource(**entry)
    else:
        raise SystemExit(f'Unknown source type: {kind!r}')
    return source, cap, n_aug


def build_sources():
    return [_make_source(entry) for entry in C.TRAIN_SOURCES]


# ─────────────────────────────────────────────────────────────────────
# Iterable feature stream
# ─────────────────────────────────────────────────────────────────────

class MultiStream(IterableDataset):
    def __init__(self, sources, preprocess):
        self.sources = sources
        self.preprocess = preprocess

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        rng = np.random.default_rng(20260830 + wid)

        for src_idx, (source, cap, n_aug) in enumerate(self.sources):
            # cap=None means unlimited (the full TC1 run).
            # A numeric cap is used for bounded/local smoke tests.
            worker_cap = None
            if cap is not None:
                worker_cap = cap // nw + (1 if wid < (cap % nw) else 0)

            raw_counts = {0: 0, 1: 0, 2: 0}

            # If a source has a forced label (e.g. local data_test/REAL),
            # only that label must reach the cap before we stop. A mixed-label
            # source such as SID waits for each raw SID class.
            forced = getattr(source, "forced", None)
            target_raw_labels = (
                {int(forced)}
                if forced is not None
                else {0, 1}
            )

            local_idx = 0

            for image, raw in source.examples(wid, nw, seed=C.SEED):
                if worker_cap is not None and all(
                    raw_counts[c] >= worker_cap for c in target_raw_labels
                ):
                    break

                if raw not in RAW_VALID_LABELS:
                    continue
                if raw == 2:  # skip tampered — not present in eval set
                    continue
                if worker_cap is not None and raw_counts[raw] >= worker_cap:
                    continue

                label = to_binary_label(raw)
                if label is None:
                    continue

                try:
                    rgb = np.array(image.convert('RGB'))
                except Exception:
                    continue
                if rgb.ndim != 3 or min(rgb.shape[:2]) < 16:
                    continue

                raw_counts[raw] += 1
                local_idx += 1
                group = src_idx * 10**15 + wid * 10**12 + local_idx

                chosen = [0]
                if n_aug > 0:
                    chosen.extend(
                        rng.choice(
                            np.arange(1, len(AUGS)),
                            size=min(n_aug, len(AUGS) - 1),
                            replace=False,
                        ).tolist()
                    )

                for aug_id in chosen:
                    try:
                        transformed = AUGS[aug_id][1](rgb, rng)
                        yield (
                            self.preprocess(Image.fromarray(transformed)),
                            dct_feature(transformed),
                            label,
                            group,
                            int(aug_id),
                            src_idx,
                        )
                    except Exception:
                        continue


# ─────────────────────────────────────────────────────────────────────
# Inspection helpers
# ─────────────────────────────────────────────────────────────────────

def do_probe(spec):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('pip install "datasets>=2.14" to use --probe') from exc

    body = spec[3:] if spec.startswith('hf:') else spec
    repo, split = (body.split('@', 1) + ['train'])[:2]
    ds = load_dataset(repo, split=split, streaming=True)
    example = next(iter(ds))

    print(f'\nColumns in {repo} [{split}]:')
    for key, value in example.items():
        desc = type(value).__name__
        if hasattr(value, 'size'):
            desc += f' size={value.size}'
        elif isinstance(value, (str, int, float)):
            desc += f' = {value!r}'
        print(f'  {key:20s} {desc}')
    print(f'\nShards: {getattr(ds, "n_shards", "unknown")}\n')



def _cap_text(cap):
    return "ALL" if cap is None else f"{cap:,}/raw-class"

def do_dry_run(sources):
    print(
        '\nRaw-label preview '
        '(0=authentic, 1=synthetic, 2=tampered; 1/2 -> binary AIGC=1)\n'
    )
    for source, cap, n_aug in sources:
        counts, unlabelled, seen = source.preview()
        pretty = ', '.join(
            f'{"?" if key is None else key}: {value:,}'
            for key, value in sorted(
                counts.items(), key=lambda item: (str(item[0] is None), str(item[0]))
            )
        )
        print(
            f'  {source.name:<16} [{source.kind}] {seen:,} sampled | '
            f'{pretty} | cap={_cap_text(cap)} | n_aug={n_aug}'
        )
        for path in unlabelled:
            print(f'      unrecognised: {path}')
    print()


# ─────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────

def run(overwrite=False):
    import open_clip

    sources = build_sources()
    out_dir = Path(C.FEATURES_TRAIN)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob('shard_*.npz'))
    meta_path = out_dir / 'meta.json'
    if existing or meta_path.exists():
        if not overwrite:
            raise SystemExit(
                f'{out_dir} already contains extracted features. '
                'Refusing to append because rerunning would duplicate examples. '
                'Use --overwrite for a fresh extraction.'
            )
        for path in existing:
            path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        print(f'Cleared previous extraction in {out_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    model, _, preprocess = open_clip.create_model_and_transforms(
        C.CLIP_MODEL, pretrained=C.CLIP_PRETRAINED
    )
    model = model.to(device).eval()

    workers = C.N_WORKERS
    min_shards = min(source.n_shards() for source, _, _ in sources)
    if min_shards < workers:
        print(
            f'! Smallest streamed source has {min_shards} shard(s); '
            f'workers {workers} -> {min_shards}'
        )
        workers = max(1, min_shards)

    print('Sources:')
    for idx, (source, cap, n_aug) in enumerate(sources):
        print(
            f'  [{idx}] {source.name:<16} {source.kind:<4} '
            f'cap={_cap_text(cap)} n_aug={n_aug}'
        )

    stream = MultiStream(sources, preprocess)
    loader_kwargs = dict(
        dataset=stream,
        batch_size=C.EXTRACT_BATCH_SIZE,
        num_workers=workers,
        pin_memory=(device.type == 'cuda'),
    )
    if workers > 0:
        loader_kwargs['prefetch_factor'] = 4
    loader = DataLoader(**loader_kwargs)

    keys = ('clip', 'dct', 'label', 'group', 'aug', 'src')
    buffer = {key: [] for key in keys}
    buffer_rows = 0
    shard_id = 0
    total = 0
    start = time.time()

    # Verification counters.
    # "originals" counts clean rows (aug == 0), i.e. source images.
    # "vectors" counts clean + augmented feature vectors.
    source_vector_counts = np.zeros(len(sources), dtype=np.int64)
    source_original_counts = np.zeros(len(sources), dtype=np.int64)

    def flush():
        nonlocal shard_id, buffer, buffer_rows
        if buffer_rows == 0:
            return

        arrays = {key: np.concatenate(buffer[key]) for key in keys}
        arrays['clip'] = arrays['clip'].astype(np.float32)
        arrays['dct'] = arrays['dct'].astype(np.float32)
        arrays['label'] = arrays['label'].astype(np.int64)
        arrays['group'] = arrays['group'].astype(np.int64)
        arrays['aug'] = arrays['aug'].astype(np.int64)
        arrays['src'] = arrays['src'].astype(np.int64)

        path = out_dir / f'shard_{shard_id:05d}.npz'
        np.savez_compressed(path, **arrays)
        print(f'  wrote {path.name} ({len(arrays["label"]):,} rows)')

        shard_id += 1
        buffer = {key: [] for key in keys}
        buffer_rows = 0

    try:
        for tensors, dct, label, group, aug, src in loader:
            with torch.no_grad():
                clip = model.encode_image(
                    tensors.to(device, non_blocking=True)
                )

            buffer['clip'].append(clip.cpu().numpy())
            for key, value in zip(
                ('dct', 'label', 'group', 'aug', 'src'),
                (dct, label, group, aug, src),
            ):
                buffer[key].append(value.numpy())

            rows = len(label)
            total += rows
            buffer_rows += rows

            src_np = src.numpy()
            aug_np = aug.numpy()
            for src_id in np.unique(src_np):
                src_id = int(src_id)
                src_mask = src_np == src_id
                source_vector_counts[src_id] += int(src_mask.sum())
                source_original_counts[src_id] += int(
                    np.logical_and(src_mask, aug_np == 0).sum()
                )

            if total % (C.EXTRACT_BATCH_SIZE * 50) < C.EXTRACT_BATCH_SIZE:
                speed = total / max(1e-6, time.time() - start)
                print(f'{total:,} vectors | {speed:.0f}/s', flush=True)

            if buffer_rows >= C.SHARD_SIZE:
                flush()
    except KeyboardInterrupt:
        print('\nInterrupted — flushing the completed batches currently in memory.')
    finally:
        flush()

    meta_path.write_text(
        json.dumps(
            {
                'total': total,
                'aug_names': AUG_NAMES,
                'n_classes': 2,
                'class_names': ['authentic', 'aigc'],
                'raw_to_binary': {
                    '0': 0,
                    '1': 1,
                    '2': 'skipped',
                },
                'sources': [
                    {
                        'name': source.name,
                        'kind': source.kind,
                        'max_per_raw_class': cap,
                        'n_aug': n_aug,
                        'original_images': int(source_original_counts[i]),
                        'feature_vectors': int(source_vector_counts[i]),
                    }
                    for i, (source, cap, n_aug) in enumerate(sources)
                ],
            },
            indent=2,
        )
    )

    print('\nFinal source counts:')
    for i, (source, _, n_aug) in enumerate(sources):
        print(
            f'  {source.name:<16} '
            f'originals={int(source_original_counts[i]):,} | '
            f'vectors={int(source_vector_counts[i]):,} | '
            f'n_aug={n_aug}'
        )

    print(
        f'\nDone. {total:,} feature vectors in {out_dir}/ '
        f'({time.time() - start:.0f}s)'
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    overwrite = '--overwrite' in args

    if '--probe' in args:
        idx = args.index('--probe')
        if idx + 1 >= len(args):
            raise SystemExit('Usage: stream_extract.py --probe REPO[@SPLIT]')
        do_probe(args[idx + 1])
    elif dry_run:
        do_dry_run(build_sources())
    else:
        run(overwrite=overwrite)