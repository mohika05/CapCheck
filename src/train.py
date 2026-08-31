#!/usr/bin/env python3
"""
Train the binary CLIP+DCT AIGC classifier.

Important competition split:
- Optimizer data: SID + CIFAKE train.
- Internal model selection: source(s) in config.VAL_SOURCE_NAMES
  (TC1 config uses CIFAKE test).
- WildFake: never loaded here. It is evaluated only by predict.py after the
  checkpoint is frozen.

The checkpoint stores the feature mean/std so inference applies exactly the
same CLIP L2-normalisation + concatenation + standardisation as training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C

KEYS = ('clip', 'dct', 'label', 'group', 'aug', 'src')
CLASS_NAMES = ['authentic', 'aigc']


def load_shards(dirs):
    """Load one or more independently extracted shard directories."""
    parts = {key: [] for key in KEYS}
    source_names = []
    group_offset = 0
    source_offset = 0

    for directory in dirs:
        directory = Path(directory)
        files = sorted(directory.glob('shard_*.npz'))
        if not files:
            raise SystemExit(f'No shard_*.npz files found in {directory}')

        meta_path = directory / 'meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        local_names = [item['name'] for item in meta.get('sources', [])]

        for file in files:
            with np.load(file) as z:
                for key in KEYS:
                    if key not in z:
                        raise SystemExit(f'{file} is missing key {key!r}')
                    arr = z[key].copy()
                    if key == 'group':
                        arr = arr + group_offset
                    elif key == 'src':
                        arr = arr + source_offset
                    parts[key].append(arr)

        local_src_max = max(int(parts['src'][-1].max()) - source_offset + 1, 0)
        n_sources = max(local_src_max, len(local_names))
        while len(local_names) < n_sources:
            local_names.append(f'{directory.name}:src{len(local_names)}')

        source_names.extend(local_names)
        source_offset += n_sources
        group_offset += 10**18

    data = {key: np.concatenate(value) for key, value in parts.items()}
    data['source_names'] = source_names
    data['aug_names'] = _aug_names(dirs)

    labels = np.unique(data['label'])
    if not set(labels.tolist()).issubset({0, 1}):
        raise SystemExit(
            f'Expected binary labels 0/1, found {labels.tolist()}. '
            'Re-run stream_extract.py --overwrite with the updated extractor.'
        )

    print(f'Loaded {len(data["label"]):,} vectors')
    for idx, name in enumerate(source_names):
        mask = data['src'] == idx
        if mask.any():
            n_real = int((data['label'][mask] == 0).sum())
            n_aigc = int((data['label'][mask] == 1).sum())
            print(
                f'  [{idx}] {name:<18} {mask.sum():>9,} rows | '
                f'authentic {n_real:,} / AIGC {n_aigc:,}'
            )
    return data


def _aug_names(dirs):
    for directory in dirs:
        path = Path(directory) / 'meta.json'
        if path.exists():
            names = json.loads(path.read_text()).get('aug_names')
            if names:
                return names
    return None


def build_features(clip, dct):
    """Exact train-side feature construction mirrored by predict.py."""
    clip = clip.astype(np.float32)
    dct = dct.astype(np.float32)
    clip /= np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8
    return np.concatenate([clip, dct], axis=1).astype(np.float32)


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _binary_metrics(y, probs):
    preds = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(y, preds)
    auc = None
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, probs)
    return acc, auc, preds


def breakdown(title, tag_values, tag_names, y, probs, min_n=20):
    print(f'\n{title}')
    print(f'{"":18}{"n":>9}{"acc":>10}{"auc":>10}')
    for tag_id in sorted(np.unique(tag_values)):
        mask = tag_values == tag_id
        if mask.sum() < min_n:
            continue
        name = (
            tag_names[tag_id]
            if tag_names and 0 <= tag_id < len(tag_names)
            else str(tag_id)
        )
        acc, auc, _ = _binary_metrics(y[mask], probs[mask])
        auc_text = f'{auc:.4f}' if auc is not None else 'n/a'
        print(f'{name:<18}{mask.sum():>9,}{acc:>10.4f}{auc_text:>10}')


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run():
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    data = load_shards([C.FEATURES_TRAIN])
    names = data['source_names']
    aug_names = data['aug_names']

    X = build_features(data['clip'], data['dct'])
    y = data['label'].astype(np.int64)
    groups = data['group']
    augs = data['aug']
    srcs = data['src']

    # ── Split: NEVER WildFake ─────────────────────────────────────────
    if C.VAL_SOURCE_NAMES:
        val_ids = [idx for idx, name in enumerate(names) if name in C.VAL_SOURCE_NAMES]
        if not val_ids:
            raise SystemExit(
                f'VAL_SOURCE_NAMES={C.VAL_SOURCE_NAMES} matched nothing. '
                f'Available sources: {names}'
            )
        val_mask = np.isin(srcs, val_ids)
        print(
            f'Internal validation source(s): {C.VAL_SOURCE_NAMES} '
            f'({val_mask.sum():,} rows)'
        )
    else:
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(C.SEED)
        rng.shuffle(unique_groups)
        n_val = max(1, int(len(unique_groups) * C.VAL_FRACTION))
        val_groups = set(unique_groups[:n_val].tolist())
        val_mask = np.fromiter(
            (group in val_groups for group in groups),
            dtype=bool,
            count=len(groups),
        )
        print(f'Grouped random validation: {C.VAL_FRACTION:.0%}')

    train_mask = ~val_mask
    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]
    ava, sva = augs[val_mask], srcs[val_mask]

    if len(ytr) == 0 or len(yva) == 0:
        raise SystemExit('Split left train or validation empty.')
    if len(np.unique(ytr)) < 2:
        raise SystemExit('Training data does not contain both binary classes.')

    print(f'\nTrain {len(ytr):,} | Val {len(yva):,} | dim {X.shape[1]}')
    train_counts = np.bincount(ytr, minlength=2)
    print(
        f'Train class balance: authentic={train_counts[0]:,}, '
        f'AIGC={train_counts[1]:,}'
    )

    # Fit scaling on optimizer data only.
    mean = Xtr.mean(axis=0).astype(np.float32)
    std = (Xtr.std(axis=0) + 1e-6).astype(np.float32)
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xva = ((Xva - mean) / std).astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_ds = TensorDataset(
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr.astype(np.float32)),
    )
    loader = DataLoader(
        train_ds,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == 'cuda'),
    )

    Xva_t = torch.from_numpy(Xva).to(device)
    yva_np = yva.astype(np.int64)

    model = Classifier(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)

    negatives = max(1, int((ytr == 0).sum()))
    positives = max(1, int((ytr == 1).sum()))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    model_path = Path(C.MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(C.EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xva_t)).cpu().numpy()
        acc, auc, _ = _binary_metrics(yva_np, probs)
        scheduler.step(acc)

        if epoch % 5 == 0 or acc > best_acc:
            auc_text = f'{auc:.4f}' if auc is not None else 'n/a'
            print(
                f'Epoch {epoch:3d} | loss {total_loss / max(1, len(loader)):.4f} '
                f'| val acc {acc:.4f} | val auc {auc_text}'
            )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'input_dim': int(X.shape[1]),
                    'mean': torch.from_numpy(mean),
                    'std': torch.from_numpy(std),
                    'threshold': 0.5,
                    'n_classes': 2,
                    'class_names': CLASS_NAMES,
                    'clip_model': C.CLIP_MODEL,
                    'clip_pretrained': C.CLIP_PRETRAINED,
                    'raw_to_binary': {0: 0, 1: 1, 2: 1},
                },
                model_path,
            )
            print(f'           -> new best, saved to {model_path}')

    # ── Final INTERNAL validation report ──────────────────────────────
    checkpoint = _load_checkpoint(model_path, device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xva_t)).cpu().numpy()

    acc, auc, preds = _binary_metrics(yva_np, probs)
    print(f'\nBest internal val accuracy: {acc:.4f}')
    if auc is not None:
        print(f'Internal val ROC-AUC:      {auc:.4f}')
    print()
    print(
        classification_report(
            yva_np,
            preds,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    breakdown('Internal validation by transform', ava, aug_names, yva_np, probs)
    breakdown('Internal validation by source', sva, names, yva_np, probs)

    # ── Threshold search (informational only) ─────────────────────────
    # Finds the threshold that maximises balanced accuracy on the internal
    # validation set. This does NOT change the saved model or predictions.json.
    # Note: the optimal threshold here is for CIFAKE test distribution and
    # may differ from the optimal threshold on WildFake — treat as diagnostic.
    print('\nOptimal threshold search on internal validation set:')
    print(f'{"threshold":>10}  {"acc":>8}  {"real_rec":>10}  {"aigc_rec":>10}  {"bal_acc":>10}')

    best_bal_acc = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.95, 0.025):
        p = (probs >= thresh).astype(int)
        real_rec = ((p == 0) & (yva_np == 0)).sum() / max(1, (yva_np == 0).sum())
        aigc_rec = ((p == 1) & (yva_np == 1)).sum() / max(1, (yva_np == 1).sum())
        bal = (real_rec + aigc_rec) / 2
        acc_t = accuracy_score(yva_np, p)
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_thresh = float(thresh)
        print(f'{thresh:>10.3f}  {acc_t:>8.4f}  {real_rec:>10.4f}  {aigc_rec:>10.4f}  {bal:>10.4f}')

    print(f'\nBest balanced accuracy: {best_bal_acc:.4f} at threshold {best_thresh:.3f}')
    print('Note: internal val is CIFAKE test — threshold may not transfer to WildFake.')


if __name__ == '__main__':
    run()