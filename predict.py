import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import open_clip
from config import MODEL_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model_clip = model_clip.to(device)
model_clip.eval()

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

classifier = Classifier(input_dim=576).to(device)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
classifier.eval()

def extract_dct(path):
    try:
        img = Image.open(path).convert('L').resize((128, 128))
        arr = np.array(img, dtype=np.float32)
        dct = cv2.dct(arr)
        block = dct[:8, :8].flatten()
        return block / (np.max(np.abs(block)) + 1e-8)
    except:
        return np.zeros(64)

def predict_directory(image_dir, batch_size=64):
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    paths = [p for p in Path(image_dir).rglob('*') if p.suffix.lower() in exts]
    print(f"Found {len(paths)} images in {image_dir}")

    results = []
    for i in tqdm(range(0, len(paths), batch_size)):
        batch = paths[i:i+batch_size]
        tensors, dcts, batch_paths = [], [], []
        for p in batch:
            try:
                tensors.append(preprocess(Image.open(p).convert('RGB')))
                dcts.append(extract_dct(p))
                batch_paths.append(str(p))
            except:
                continue
        if not tensors:
            continue
        with torch.no_grad():
            clip_feats = model_clip.encode_image(torch.stack(tensors).to(device)).cpu().numpy()
        dct_feats = np.array(dcts)
        X = np.concatenate([clip_feats, dct_feats], axis=1)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(classifier(X_t)).cpu().numpy()
        for path, prob in zip(batch_paths, probs):
            results.append({
                'image_path': path,
                'pred': float(prob),
                'label': 'AI-generated' if prob > 0.5 else 'Real'
            })
    return results

if __name__ == '__main__':
    image_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    results = predict_directory(image_dir)
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} predictions to predictions.json")
