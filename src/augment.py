import albumentations as A
import cv2
import numpy as np
from PIL import Image
import open_clip
import torch
from pathlib import Path
from tqdm import tqdm
from config import DATA_REAL, DATA_FAKE, FEATURES_DIR
import os

# ── Load CLIP ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)
model.eval()

TRANSFORMS = {
    # ── JPEG Compression  (quality = 90, 70, 50, 30) ──────────────
    'jpeg_90': A.ImageCompression(
        quality_lower=90, quality_upper=90, p=1.0),
    'jpeg_70': A.ImageCompression(
        quality_lower=70, quality_upper=70, p=1.0),
    'jpeg_50': A.ImageCompression(
        quality_lower=50, quality_upper=50, p=1.0),
    'jpeg_30': A.ImageCompression(
        quality_lower=30, quality_upper=30, p=1.0),

    # ── Gaussian Blur  (σ = 0.5, 1.0, 2.0) ───────────────────────
    'blur_05': A.GaussianBlur(
        blur_limit=(3, 3), sigma_limit=(0.5, 0.5), p=1.0),
    'blur_10': A.GaussianBlur(
        blur_limit=(5, 5), sigma_limit=(1.0, 1.0), p=1.0),
    'blur_20': A.GaussianBlur(
        blur_limit=(9, 9), sigma_limit=(2.0, 2.0), p=1.0),

    # ── Resize  (0.5× / 0.25× then upscale) ──────────────────────
    # CLIP preprocess upscales back to 224, so we only need to downscale.
    'resize_half': A.Resize(112, 112),      # 0.5 × 224
    'resize_quarter': A.Resize(56, 56),     # 0.25 × 224

    # ── Gaussian Noise  (σ = 0.02 / 0.05 / 0.10 in [0,1]) ───────
    # var = (σ × 255)²: 26, 163, 650
    'noise_002': A.GaussNoise(var_limit=(26,  26),  p=1.0),
    'noise_005': A.GaussNoise(var_limit=(163, 163), p=1.0),
    'noise_010': A.GaussNoise(var_limit=(650, 650), p=1.0),

    # ── Color Jitter  (brightness / contrast / saturation ±20%) ──
    'color_jitter': A.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, p=1.0),

    # ── Center Crop  (80% of image area) ─────────────────────────
    # Normalise to 280 × 280 first so the 80% crop is always consistent.
    'center_crop': A.Compose([
        A.Resize(280, 280),       # standardise input
        A.CenterCrop(224, 224),   # 224 / 280 = 80.0%
        ]),
}

def _apply_transform(image_path, transform):
    """Load one image from disk and apply a transform. Returns RGB numpy array."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return transform(image=img_rgb)['image']

def build_augmented_set(real_dir, fake_dir, out_prefix,
                        max_per_class=5000, clip_batch_size=64):
    """
    For every transform, augment every image and extract CLIP features.

    Images are passed through CLIP in batches of `clip_batch_size` instead
    of one at a time — this is 30-50× faster on GPU for large datasets.
    """
    exts = {'.jpg', '.jpeg', '.png', '.webp'}

    real_paths = [p for p in Path(real_dir).rglob('*')
                if p.suffix.lower() in exts][:max_per_class]
    fake_paths = [p for p in Path(fake_dir).rglob('*')
                if p.suffix.lower() in exts][:max_per_class]

    total_images = len(real_paths) + len(fake_paths)
    print(f"Found {len(real_paths)} real + {len(fake_paths)} AI images.")
    print(f"Running {len(TRANSFORMS)} transforms "
            f"→ ~{len(TRANSFORMS) * total_images:,} CLIP forward passes "
            f"in batches of {clip_batch_size}.\n")

    all_features, all_labels = [], []

    for t_name, transform in TRANSFORMS.items():
        print(f"Transform: {t_name}")

        # Collect preprocessed tensors; flush through CLIP every clip_batch_size images
        buf_tensors: list[torch.Tensor] = []
        buf_labels:  list[int]          = []

        def _flush_buffer():
            if not buf_tensors:
                return
            batch = torch.stack(buf_tensors).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch).cpu().numpy()
            all_features.extend(feats)
            all_labels.extend(buf_labels)
            buf_tensors.clear()
            buf_labels.clear()

        for paths, label in [(real_paths, 0), (fake_paths, 1)]:
            tag = 'real' if label == 0 else 'fake'
            for path in tqdm(paths, desc=f'  {t_name}/{tag}', leave=False):
                try:
                    aug = _apply_transform(path, transform)
                    if aug is None:
                        continue
                    buf_tensors.append(preprocess(Image.fromarray(aug)))
                    buf_labels.append(label)
                    if len(buf_tensors) == clip_batch_size:
                        _flush_buffer()
                except Exception as e:
                    print(f'\n  Error on {path}: {e}')

        _flush_buffer()
        print(f'  → {len(all_labels):,} total samples so far')

    np.save(f'{out_prefix}_aug_features.npy', np.array(all_features))
    np.save(f'{out_prefix}_aug_labels.npy',   np.array(all_labels))
    print(f'\nDone. Saved {len(all_labels):,} augmented feature vectors.')


if __name__ == '__main__':
    os.makedirs(FEATURES_DIR, exist_ok=True)
    build_augmented_set(
        DATA_REAL, DATA_FAKE,
        f'{FEATURES_DIR}/train',
        max_per_class=10,         # keep small for local smoke test
        clip_batch_size=64,
    )