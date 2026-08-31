#!/usr/bin/env python3
"""
Track 5: AI-Generated Image Detection Inference Pipeline.

Provides standard directory evaluation and optional robustness benchmarking.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import config as C
from src.stream_extract import (
    AUGS,
    AUG_NAMES,
    IMG_EXTS,
    dct_feature,
    infer_label_from_path,
    to_binary_label,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint
    elif isinstance(checkpoint, dict) and 'net.0.weight' in checkpoint:
        input_dim = checkpoint['net.0.weight'].shape[1]
        return {
            'state_dict': checkpoint,
            'input_dim': input_dim,
            'mean': torch.zeros(input_dim),
            'std': torch.ones(input_dim),
            'clip_model': 'ViT-L-14',
            'clip_pretrained': 'openai'
        }
    else:
        required = {'state_dict', 'input_dim', 'mean', 'std'}
        missing = required.difference(checkpoint.keys())
        raise SystemExit(
            f'Checkpoint {path} is incompatible; missing {sorted(missing)}.'
        )


def _collect_paths(image_dir):
    root = Path(image_dir).expanduser()
    if not root.exists():
        raise SystemExit(f'Image directory does not exist: {root}')
    paths = sorted(
        path for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not paths:
        raise SystemExit(f'No supported images found under {root}')
    return root, paths


def _path_label(path, root):
    raw = infer_label_from_path(path, root)
    return to_binary_label(raw)


def _predict_transform(
    paths,
    aug_id,
    model_clip,
    preprocess,
    classifier,
    mean_t,
    std_t,
    device,
    batch_size,
):
    output = []

    for start in tqdm(
        range(0, len(paths), batch_size),
        desc=AUG_NAMES[aug_id],
        leave=False,
    ):
        batch_paths = paths[start:start + batch_size]
        tensors = []
        dcts = []
        valid_paths = []

        for offset, path in enumerate(batch_paths):
            try:
                with Image.open(path) as image:
                    rgb = np.array(image.convert('RGB'))
                if rgb.ndim != 3 or min(rgb.shape[:2]) < 16:
                    continue

                global_index = start + offset
                rng = np.random.default_rng(
                    C.SEED + aug_id * 1_000_003 + global_index
                )
                transformed = AUGS[aug_id][1](rgb, rng)

                tensors.append(preprocess(Image.fromarray(transformed)))
                dcts.append(dct_feature(transformed))
                valid_paths.append(path)
            except Exception:
                continue

        if not tensors:
            continue

        image_tensor = torch.stack(tensors).to(device, non_blocking=True)
        dct_tensor = torch.from_numpy(np.stack(dcts).astype(np.float32)).to(
            device, non_blocking=True
        )

        with torch.no_grad():
            clip = model_clip.encode_image(image_tensor).float()
            clip = clip / (torch.linalg.norm(clip, dim=1, keepdim=True) + 1e-8)
            features = torch.cat([clip, dct_tensor], dim=1)
            features = (features - mean_t) / std_t
            probs = torch.sigmoid(classifier(features)).cpu().numpy()

        output.extend(zip(valid_paths, probs.tolist()))

    return output


def _metrics(results, root):
    labels = []
    scores = []
    for path, prob in results:
        label = _path_label(path, root)
        if label is None:
            continue
        labels.append(label)
        scores.append(prob)

    if not labels:
        return {'n_labeled': 0, 'accuracy': None, 'roc_auc': None}

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float32)
    pred = (p >= 0.5).astype(np.int64)

    auc = None
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))

    return {
        'n_labeled': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'roc_auc': auc,
    }


def _write_predictions(results, output_path):
    payload = [
        {'image_path': Path(path).as_posix(), 'pred': float(prob)}
        for path, prob in results
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return len(payload)


def main():
    parser = argparse.ArgumentParser(description="Track-5 AI-Generated Image Detector")
    parser.add_argument(
        'image_dir_pos', nargs='?', default=None,
        help='Directory of images to score (defaults to config.PREDICT_IMAGES).'
    )
    parser.add_argument(
        '--input_dir', '--input-dir', dest='image_dir_flag', default=None,
        help='Directory of images to score via named flag.'
    )
    parser.add_argument(
        '--output', default=C.PREDICTIONS_OUT,
        help='Required clean-image prediction JSON output.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=C.EXTRACT_BATCH_SIZE,
        help='CLIP inference batch size.'
    )
    parser.add_argument(
        '--report-transforms', action='store_true',
        help='Also evaluate all Track-5 transformations.'
    )
    parser.add_argument(
        '--robustness-output', default=C.ROBUSTNESS_OUT,
        help='Where to write the robustness summary JSON.'
    )
    args = parser.parse_args()

    target_image_dir = args.image_dir_flag or args.image_dir_pos or C.PREDICT_IMAGES

    import open_clip

    root, paths = _collect_paths(target_image_dir)
    print(f'Found {len(paths):,} images under {root}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    checkpoint = _load_checkpoint(C.MODEL_PATH, device)
    input_dim = int(checkpoint['input_dim'])

    classifier = Classifier(input_dim).to(device)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()

    mean_t = checkpoint['mean'].to(device=device, dtype=torch.float32).view(1, -1)
    std_t = checkpoint['std'].to(device=device, dtype=torch.float32).view(1, -1)
    if mean_t.shape[1] != input_dim or std_t.shape[1] != input_dim:
        raise SystemExit('Checkpoint mean/std dimension does not match input_dim.')

    clip_model_name = checkpoint.get('clip_model', 'ViT-B-32')
    clip_pretrained = checkpoint.get('clip_pretrained', 'openai')
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    model_clip = model_clip.to(device).eval()

    clean_results = _predict_transform(
        paths,
        0,
        model_clip,
        preprocess,
        classifier,
        mean_t,
        std_t,
        device,
        args.batch_size,
    )
    n_written = _write_predictions(clean_results, args.output)
    print(f'Saved {n_written:,} clean predictions to {args.output}')

    clean_metrics = _metrics(clean_results, root)
    if clean_metrics['n_labeled']:
        auc_text = (
            f'{clean_metrics["roc_auc"]:.4f}'
            if clean_metrics['roc_auc'] is not None else 'n/a'
        )
        print(
            f'Clean | n={clean_metrics["n_labeled"]:,} '
            f'| acc={clean_metrics["accuracy"]:.4f} | auc={auc_text}'
        )

    if not args.report_transforms:
        return

    summary = []
    clean_row = {'transform': 'clean', **clean_metrics}
    summary.append(clean_row)

    print('\nRobustness evaluation')
    print(f'{"transform":<16}{"n":>9}{"acc":>10}{"auc":>10}')

    def print_row(row):
        acc = 'n/a' if row['accuracy'] is None else f'{row["accuracy"]:.4f}'
        auc = 'n/a' if row['roc_auc'] is None else f'{row["roc_auc"]:.4f}'
        print(f'{row["transform"]:<16}{row["n_labeled"]:>9,}{acc:>10}{auc:>10}')

    print_row(clean_row)

    for aug_id in range(1, len(AUGS)):
        results = _predict_transform(
            paths,
            aug_id,
            model_clip,
            preprocess,
            classifier,
            mean_t,
            std_t,
            device,
            args.batch_size,
        )
        row = {'transform': AUG_NAMES[aug_id], **_metrics(results, root)}
        summary.append(row)
        print_row(row)

    valid_rows = [
        row for row in summary
        if row['accuracy'] is not None and row['n_labeled'] > 0
    ]
    total_n = sum(row['n_labeled'] for row in valid_rows)
    overall_accuracy = (
        sum(row['accuracy'] * row['n_labeled'] for row in valid_rows) / total_n
        if total_n else None
    )
    mean_transform_accuracy = (
        float(np.mean([row['accuracy'] for row in valid_rows]))
        if valid_rows else None
    )
    worst = min(valid_rows, key=lambda row: row['accuracy']) if valid_rows else None

    report = {
        'clean_accuracy': clean_metrics['accuracy'],
        'overall_accuracy_clean_plus_transforms': overall_accuracy,
        'mean_per_condition_accuracy': mean_transform_accuracy,
        'total_labeled_predictions_across_conditions': int(total_n),
        'worst_condition': (
            {
                'transform': worst['transform'],
                'accuracy': worst['accuracy'],
            } if worst else None
        ),
        'transforms': summary,
    }

    print('\nOverall WildFake results')
    if clean_metrics['accuracy'] is not None:
        print(f'Clean overall accuracy:                 {clean_metrics["accuracy"]:.4f}')
    if overall_accuracy is not None:
        print(f'Overall accuracy (clean + transforms): {overall_accuracy:.4f}')
    if mean_transform_accuracy is not None:
        print(f'Mean per-condition accuracy:           {mean_transform_accuracy:.4f}')
    if worst is not None:
        print(f'Worst condition: {worst["transform"]} ({worst["accuracy"]:.4f})')

    robustness_path = Path(args.robustness_output)
    robustness_path.parent.mkdir(parents=True, exist_ok=True)
    robustness_path.write_text(json.dumps(report, indent=2))
    print(f'\nSaved robustness summary to {robustness_path}')


if __name__ == '__main__':
    main()
