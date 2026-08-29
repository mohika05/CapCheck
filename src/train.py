import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# INPUT_DIM removed — model is built from the actual combined feature shape,
# which is the correct source of truth.
from config import FEATURES_DIR, MODEL_PATH, BATCH_SIZE, EPOCHS
import os

# ── Load features ─────────────────────────────────────────────────
print("Loading features...")
X_clip = np.load(f'{FEATURES_DIR}/train_features.npy')
y_clip = np.load(f'{FEATURES_DIR}/train_labels.npy')

X_dct  = np.load(f'{FEATURES_DIR}/train_dct_features.npy')

# Combine CLIP (512-d) + DCT (64-d) = 576-d per image
X_combined = np.concatenate([X_clip, X_dct], axis=1)
y = y_clip

print(f"Clean samples : {len(y):,}")
print(f"Feature dim   : {X_combined.shape[1]}")

aug_path = f'{FEATURES_DIR}/train_aug_features.npy'
if os.path.exists(aug_path):
    X_aug = np.load(aug_path)
    y_aug = np.load(f'{FEATURES_DIR}/train_aug_labels.npy')

    # DCT on augmented images is an optional enhancement (not yet run).
    # Pad with zeros so the feature dimension stays consistent.
    X_aug_dct      = np.zeros((X_aug.shape[0], 64))
    X_aug_combined = np.concatenate([X_aug, X_aug_dct], axis=1)

    X_combined = np.concatenate([X_combined, X_aug_combined])
    y          = np.concatenate([y, y_aug])
    print(f"With augmented: {len(y):,} total samples")

# ── Train / val split ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# ── MLP classifier ────────────────────────────────────────────────
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
            # No Sigmoid here.
            # BCEWithLogitsLoss fuses Sigmoid + BCE in a single numerically-stable
            # operation.  Apply torch.sigmoid() manually only at inference time.
        )

    def forward(self, x):
        return self.net(x).squeeze()


model     = Classifier(input_dim=X_combined.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# BCEWithLogitsLoss is numerically more stable than nn.Sigmoid() + nn.BCELoss()
# because it uses the log-sum-exp trick to avoid overflow / underflow.
criterion = nn.BCEWithLogitsLoss()

# Halve the learning rate if val accuracy doesn't improve for 10 eval periods.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=10, factor=0.5)


# ── Training loop ─────────────────────────────────────────────────
print("\nStarting training...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_dl:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluate every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs  = torch.sigmoid(val_logits).cpu().numpy()  # sigmoid at inference
            val_preds  = (val_probs > 0.5).astype(int)
            val_true   = y_val_t.cpu().numpy()
            acc        = accuracy_score(val_true, val_preds)

        scheduler.step(acc)

        print(f"Epoch {epoch:3d} | "
            f"Loss: {epoch_loss / len(train_dl):.4f} | "
            f"Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"           → New best! Saved to {MODEL_PATH}")


# ── Final evaluation ──────────────────────────────────────────────
print(f"\nTraining complete.")
print(f"Best validation accuracy: {best_acc:.4f}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
with torch.no_grad():
    final_logits = model(X_val_t)
    final_preds  = (torch.sigmoid(final_logits).cpu().numpy() > 0.5).astype(int)

print("\nFinal classification report:")
print(classification_report(
    y_val_t.cpu().numpy(),
    final_preds,
    target_names=['Real', 'AI-generated'],
))