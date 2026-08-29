import open_clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from config import DATA_REAL, DATA_FAKE, FEATURES_DIR, DATA_TEST_REAL, DATA_TEST_FAKE
import os

# ── Load CLIP ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)
model.eval()


def extract_dataset(real_dir, fake_dir, out_prefix, batch_size=64):
    os.makedirs(
        os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else '.',
        exist_ok=True,
    )
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    real_paths = [p for p in Path(real_dir).rglob('*') if p.suffix.lower() in exts]
    fake_paths = [p for p in Path(fake_dir).rglob('*') if p.suffix.lower() in exts]

    # Build a flat list of (path, label) so we process everything in one loop
    all_items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    print(f"Extracting CLIP features: "
        f"{len(real_paths):,} real + {len(fake_paths):,} AI "
        f"= {len(all_items):,} images  |  batch_size={batch_size}")

    features, labels = [], []

    # Slice into batches; tqdm wraps the outer loop for a clean progress bar
    for i in tqdm(range(0, len(all_items), batch_size)):
        chunk = all_items[i : i + batch_size]
        tensors, chunk_labels = [], []

        for path, label in chunk:
            try:
                t = preprocess(Image.open(path).convert('RGB'))
                tensors.append(t)
                chunk_labels.append(label)
            except Exception as e:
                print(f'\nSkipping {path}: {e}')

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor).cpu().numpy()

        features.extend(feats)
        labels.extend(chunk_labels)

    features_arr = np.array(features)
    np.save(f'{out_prefix}_features.npy', features_arr)
    np.save(f'{out_prefix}_labels.npy',   np.array(labels))
    print(f"Saved {len(labels):,} feature vectors | shape: {features_arr.shape}")


if __name__ == '__main__':
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extract_dataset(DATA_REAL, DATA_FAKE,
                    f'{FEATURES_DIR}/train')
    extract_dataset(DATA_TEST_REAL, DATA_TEST_FAKE,
                    f'{FEATURES_DIR}/test')