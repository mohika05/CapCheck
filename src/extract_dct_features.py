import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from config import DATA_REAL, DATA_FAKE, FEATURES_DIR, DATA_TEST_REAL, DATA_TEST_FAKE
from multiprocessing import Pool, cpu_count
import os


def extract_dct(image_path, img_size=128):
    """
    Extract a 64-element DCT feature vector from one image.

    The top-left 8×8 block of the DCT spectrum captures the most
    informative low-frequency coefficients — exactly where AI-generated
    images differ from real ones.

    Kept as a standalone function (not a lambda / nested def) so it is
    picklable by Python's multiprocessing module.
    """
    try:
        img = Image.open(image_path).convert('L')   # grayscale
        img = img.resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32)
        dct = cv2.dct(arr)
        block = dct[:8, :8].flatten()               # top-left 8×8 → 64 numbers
        block = block / (np.max(np.abs(block)) + 1e-8)
        return block
    except Exception:
        return None


def _worker(args):
    """Worker function: unpack (path, label) and return (feature, label)."""
    path, label = args
    feat = extract_dct(str(path))
    return feat, label


def extract_dct_dataset(real_dir, fake_dir, out_prefix, num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    real_paths = [p for p in Path(real_dir).rglob('*') if p.suffix.lower() in exts]
    fake_paths = [p for p in Path(fake_dir).rglob('*') if p.suffix.lower() in exts]

    all_items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    print(f"Extracting DCT from {len(all_items):,} images "
        f"using {num_workers} worker processes...")

    with Pool(processes=num_workers) as pool:
        # imap preserves order and streams results so tqdm works correctly.
        # chunksize=32 reduces inter-process overhead for large datasets.
        results = list(
            tqdm(pool.imap(_worker, all_items, chunksize=32),
                total=len(all_items))
        )

    features, labels = [], []
    for feat, label in results:
        if feat is not None:
            features.append(feat)
            labels.append(label)

    skipped = len(all_items) - len(features)
    if skipped:
        print(f"Skipped {skipped} images due to load / processing errors.")

    features_arr = np.array(features)
    np.save(f'{out_prefix}_dct_features.npy', features_arr)
    np.save(f'{out_prefix}_dct_labels.npy',   np.array(labels))
    print(f"Saved {len(labels):,} DCT vectors | shape: {features_arr.shape}")


if __name__ == '__main__':
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extract_dct_dataset(DATA_REAL, DATA_FAKE,
                        f'{FEATURES_DIR}/train')
    extract_dct_dataset(DATA_TEST_REAL, DATA_TEST_FAKE,
                        f'{FEATURES_DIR}/test')