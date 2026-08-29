import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from config import DATA_REAL, DATA_FAKE, FEATURES_DIR, DATA_TEST_REAL, DATA_TEST_FAKE
import os

def extract_dct(image_path, img_size=128):
    try:
        # Convert to grayscale and resize
        img = Image.open(image_path).convert('L')
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32)
        
        # Apply DCT — converts pixel space to frequency space
        dct = cv2.dct(img_array)
        
        # Take top-left 8x8 block = 64 numbers
        # These are the most informative frequency coefficients
        dct_block = dct[:8, :8].flatten()
        
        # Normalize so all values are in similar range
        dct_block = dct_block / (np.max(np.abs(dct_block)) + 1e-8)
        
        return dct_block
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def extract_dct_dataset(real_dir, fake_dir, out_prefix):
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    real_paths = [p for p in Path(real_dir).rglob('*') 
                if p.suffix.lower() in exts]
    fake_paths = [p for p in Path(fake_dir).rglob('*') 
                if p.suffix.lower() in exts]
    
    features, labels = [], []
    
    print(f"Extracting DCT from {len(real_paths)} real images...")
    for path in tqdm(real_paths):
        f = extract_dct(path)
        if f is not None:
            features.append(f)
            labels.append(0)
    
    print(f"Extracting DCT from {len(fake_paths)} AI images...")
    for path in tqdm(fake_paths):
        f = extract_dct(path)
        if f is not None:
            features.append(f)
            labels.append(1)
    
    np.save(f'{out_prefix}_dct_features.npy', np.array(features))
    np.save(f'{out_prefix}_dct_labels.npy', np.array(labels))
    print(f"Saved {len(labels)} DCT vectors")
    print(f"DCT shape: {np.array(features).shape}")

if __name__ == '__main__':
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extract_dct_dataset(DATA_REAL, DATA_FAKE, f'{FEATURES_DIR}/train')
    extract_dct_dataset(DATA_TEST_REAL, DATA_TEST_FAKE, f'{FEATURES_DIR}/test')