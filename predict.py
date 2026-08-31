#!/usr/bin/env python3
"""
Track-5 inference and WildFake robustness evaluation.

Required submission behaviour:
    python predict.py IMAGE_DIR

writes a JSON list with exactly:
    {"image_path": "...", "pred": 0.0_to_1.0}

where pred is the model's AIGC likelihood.

Optional robustness evaluation:
    python predict.py IMAGE_DIR --report-transforms

scores clean images plus every Track-5 transformation and writes a compact
robustness_summary.json. This evaluation happens after the trained checkpoint
is loaded; it never feeds back into training or checkpoint selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)
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

CLASS_NAMES = ['authentic', 'aigc']


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),  # wider to match ViT-L-14's 768-dim input
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

    required = {'state_dict', 'input_dim', 'mean', 'std'}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise SystemExit(
            f'Checkpoint {path} is incompatible; missing {sorted(missing)}. '
            'Re-run train.py with the updated binary pipeline.'
        )
    return checkpoint


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
    """Predict one transformation for all paths using exact train preprocessing."""
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
    labels, scores = [], []
    for path, prob in results:
        label = _path_label(path, root)
        if label is None:
            continue
        labels.append(label)
        scores.append(prob)

    if not labels:
        return {'n_labeled': 0, 'accuracy': None, 'roc_auc': None,
                'real_precision': None, 'real_recall': None, 'real_f1': None,
                'aigc_precision': None, 'aigc_recall': None, 'aigc_f1': None}

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float32)
    pred = (p >= 0.5).astype(np.int64)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    acc = float(accuracy_score(y, pred))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        'n_labeled': int(len(y)),
        'accuracy': acc,
        'roc_auc': auc,
        'real_precision': float(prec[0]),
        'real_recall': float(rec[0]),
        'real_f1': float(f1[0]),
        'aigc_precision': float(prec[1]),
        'aigc_recall': float(rec[1]),
        'aigc_f1': float(f1[1]),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
    }


def _print_full_report(metrics, label=''):
    if label:
        print(f'\n{label}')
    n = metrics['n_labeled']
    if not n:
        print('  No labeled images found.')
        return
    print(f'  Images:   {n:,}')
    print(f'  Accuracy: {metrics["accuracy"]:.4f}')
    if metrics['roc_auc'] is not None:
        print(f'  ROC-AUC:  {metrics["roc_auc"]:.4f}')
    print()
    print(f'  {"":12}  {"precision":>10}  {"recall":>10}  {"f1":>10}')
    print(f'  {"authentic":12}  {metrics["real_precision"]:>10.4f}  '
          f'{metrics["real_recall"]:>10.4f}  {metrics["real_f1"]:>10.4f}')
    print(f'  {"aigc":12}  {metrics["aigc_precision"]:>10.4f}  '
          f'{metrics["aigc_recall"]:>10.4f}  {metrics["aigc_f1"]:>10.4f}')
    print()
    print(f'  Confusion matrix:')
    print(f'    TP (AIGC correctly detected): {metrics["true_positives"]:,}')
    print(f'    TN (Real correctly cleared):  {metrics["true_negatives"]:,}')
    print(f'    FP (Real wrongly flagged):    {metrics["false_positives"]:,}')
    print(f'    FN (AIGC wrongly missed):     {metrics["false_negatives"]:,}')


def _write_predictions(results, output_path):
    payload = [
        {'image_path': str(path), 'pred': float(prob)}
        for path, prob in results
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return len(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_dir', nargs='?', default=C.PREDICT_IMAGES,
        help='Directory of images to score (defaults to config.PREDICT_IMAGES).'
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

    import open_clip

    root, paths = _collect_paths(args.image_dir)
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

    # Clean pass — always required for submission JSON
    clean_results = _predict_transform(
        paths, 0, model_clip, preprocess, classifier,
        mean_t, std_t, device, args.batch_size,
    )
    n_written = _write_predictions(clean_results, args.output)
    print(f'Saved {n_written:,} clean predictions to {args.output}')

    clean_metrics = _metrics(clean_results, root)
    _print_full_report(clean_metrics, label='Clean WildFake results')

    if not args.report_transforms:
        return

    summary = [{'transform': 'clean', **clean_metrics}]

    print('\nRobustness evaluation')
    print(f'{"transform":<16}{"n":>9}{"acc":>10}{"auc":>10}'
          f'{"real_rec":>10}{"aigc_rec":>10}')

    def print_row(name, m):
        acc = 'n/a' if m['accuracy'] is None else f'{m["accuracy"]:.4f}'
        auc = 'n/a' if m['roc_auc'] is None else f'{m["roc_auc"]:.4f}'
        rr  = 'n/a' if m['real_recall'] is None else f'{m["real_recall"]:.4f}'
        ar  = 'n/a' if m['aigc_recall'] is None else f'{m["aigc_recall"]:.4f}'
        print(f'{name:<16}{m["n_labeled"]:>9,}{acc:>10}{auc:>10}{rr:>10}{ar:>10}')

    print_row('clean', clean_metrics)

    for aug_id in range(1, len(AUGS)):
        results = _predict_transform(
            paths, aug_id, model_clip, preprocess, classifier,
            mean_t, std_t, device, args.batch_size,
        )
        m = {'transform': AUG_NAMES[aug_id], **_metrics(results, root)}
        summary.append(m)
        print_row(AUG_NAMES[aug_id], m)

    valid_rows = [r for r in summary if r['accuracy'] is not None and r['n_labeled'] > 0]
    total_n = sum(r['n_labeled'] for r in valid_rows)
    overall_acc = (
        sum(r['accuracy'] * r['n_labeled'] for r in valid_rows) / total_n
        if total_n else None
    )
    mean_acc = float(np.mean([r['accuracy'] for r in valid_rows])) if valid_rows else None
    worst = min(valid_rows, key=lambda r: r['accuracy']) if valid_rows else None

    report = {
        'clean_accuracy': clean_metrics['accuracy'],
        'clean_roc_auc': clean_metrics['roc_auc'],
        'overall_accuracy_clean_plus_transforms': overall_acc,
        'mean_per_condition_accuracy': mean_acc,
        'total_labeled_predictions_across_conditions': int(total_n),
        'worst_condition': (
            {'transform': worst['transform'], 'accuracy': worst['accuracy']}
            if worst else None
        ),
        'transforms': summary,
    }

    print('\nOverall WildFake results')
    if clean_metrics['accuracy'] is not None:
        print(f'Clean accuracy:                        {clean_metrics["accuracy"]:.4f}')
    if clean_metrics['roc_auc'] is not None:
        print(f'Clean ROC-AUC:                         {clean_metrics["roc_auc"]:.4f}')
    if overall_acc is not None:
        print(f'Overall accuracy (clean + transforms): {overall_acc:.4f}')
    if mean_acc is not None:
        print(f'Mean per-condition accuracy:           {mean_acc:.4f}')
    if worst:
        print(f'Worst condition: {worst["transform"]} ({worst["accuracy"]:.4f})')

    robustness_path = Path(args.robustness_output)
    robustness_path.parent.mkdir(parents=True, exist_ok=True)
    robustness_path.write_text(json.dumps(report, indent=2))
    print(f'\nSaved robustness summary to {robustness_path}')


if __name__ == '__main__':
    main()