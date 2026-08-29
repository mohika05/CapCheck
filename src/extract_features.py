import open_clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from config import DATA_REAL, DATA_FAKE, FEATURES_DIR, DATA_TEST_REAL, DATA_TEST_FAKE
import os

# Set up CLIP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)
model.eval()

def extract_one(image_path):
    try:
        img = preprocess(
            Image.open(image_path).convert('RGB')
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        return feat.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def extract_dataset(real_dir, fake_dir, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) 
                if os.path.dirname(out_prefix) else '.', exist_ok=True)
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    real_paths = [p for p in Path(real_dir).rglob('*') 
                if p.suffix.lower() in exts]
    fake_paths = [p for p in Path(fake_dir).rglob('*') 
                if p.suffix.lower() in exts]
    
    features, labels = [], []
    
    print(f"Extracting CLIP features from {len(real_paths)} real images...")
    for path in tqdm(real_paths):
        f = extract_one(path)
        if f is not None:
            features.append(f)
            labels.append(0)  # 0 = real
    
    print(f"Extracting CLIP features from {len(fake_paths)} AI images...")
    for path in tqdm(fake_paths):
        f = extract_one(path)
        if f is not None:
            features.append(f)
            labels.append(1)  # 1 = AI generated
    
    np.save(f'{out_prefix}_features.npy', np.array(features))
    np.save(f'{out_prefix}_labels.npy', np.array(labels))
    print(f"Saved {len(labels)} feature vectors to {out_prefix}_features.npy")
    print(f"Feature shape: {np.array(features).shape}")

if __name__ == '__main__':
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extract_dataset(DATA_REAL, DATA_FAKE, f'{FEATURES_DIR}/train')
    extract_dataset(DATA_TEST_REAL, DATA_TEST_FAKE, f'{FEATURES_DIR}/test')