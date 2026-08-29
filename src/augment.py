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

# Load CLIP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model = model.to(device)
model.eval()

# All 6 transforms from the problem statement
TRANSFORMS = {
    'jpeg_90':      A.ImageCompression(
                        quality_lower=90, quality_upper=90, p=1.0),
    'jpeg_70':      A.ImageCompression(
                        quality_lower=70, quality_upper=70, p=1.0),
    'jpeg_50':      A.ImageCompression(
                        quality_lower=50, quality_upper=50, p=1.0),
    'jpeg_30':      A.ImageCompression(
                        quality_lower=30, quality_upper=30, p=1.0),
    'blur_05':      A.GaussianBlur(
                        blur_limit=(3,3), sigma_limit=(0.5,0.5), p=1.0),
    'blur_10':      A.GaussianBlur(
                        blur_limit=(5,5), sigma_limit=(1.0,1.0), p=1.0),
    'blur_20':      A.GaussianBlur(
                        blur_limit=(9,9), sigma_limit=(2.0,2.0), p=1.0),
    'noise_002':    A.GaussNoise(var_limit=(5, 15), p=1.0),
    'noise_005':    A.GaussNoise(var_limit=(30, 50), p=1.0),
    'noise_010':    A.GaussNoise(var_limit=(100, 130), p=1.0),
    'color_jitter': A.ColorJitter(
                        brightness=0.2, contrast=0.2, 
                        saturation=0.2, p=1.0),
    'center_crop':  A.Compose([
                        A.CenterCrop(height=180, width=180),
                        A.Resize(224, 224)
                    ]),
}

def augment_and_extract(image_path, transform):
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        augmented = transform(image=img_rgb)['image']
        
        # Convert to PIL for CLIP
        pil = Image.fromarray(augmented)
        tensor = preprocess(pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model.encode_image(tensor)
        return feat.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error on {image_path}: {e}")
        return None

def build_augmented_set(real_dir, fake_dir, out_prefix, 
                        max_per_class=5000):
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    
    real_paths = [p for p in Path(real_dir).rglob('*') 
                if p.suffix.lower() in exts][:max_per_class]
    fake_paths = [p for p in Path(fake_dir).rglob('*') 
                if p.suffix.lower() in exts][:max_per_class]
    
    all_features, all_labels = [], []
    
    for t_name, transform in TRANSFORMS.items():
        print(f"\nApplying transform: {t_name}")
        
        for path in tqdm(real_paths):
            f = augment_and_extract(path, transform)
            if f is not None:
                all_features.append(f)
                all_labels.append(0)
        
        for path in tqdm(fake_paths):
            f = augment_and_extract(path, transform)
            if f is not None:
                all_features.append(f)
                all_labels.append(1)
        
        print(f"Total so far: {len(all_labels)} samples")
    
    np.save(f'{out_prefix}_aug_features.npy', np.array(all_features))
    np.save(f'{out_prefix}_aug_labels.npy', np.array(all_labels))
    print(f"\nDone. Saved {len(all_labels)} augmented samples")

if __name__ == '__main__':
    os.makedirs(FEATURES_DIR, exist_ok=True)
    build_augmented_set(
        DATA_REAL, DATA_FAKE,
        f'{FEATURES_DIR}/train',
        max_per_class=10  # small for local test
    )