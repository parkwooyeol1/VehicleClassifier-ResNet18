import os
import copy
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models

from torchsummary import summary
import io
from contextlib import redirect_stdout

# ----------------- 사용자 설정 -----------------
RAW_DATA_DIR = "/home/parkwooyeol/workspace/Resnet-Experiment/downloads"  # 원본 데이터 (클래스별 폴더만 있는 상태)
DATASET_DIR = "/home/parkwooyeol/workspace/Resnet-Experiment/dataset"     # train/val/test 저장 위치
BATCH_SIZE = 32
IMG_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
PATIENCE = 3
BEST_MODEL_PATH = "/home/parkwooyeol/workspace/Resnet-Experiment/best_resnet18.pth"
PLOT_PATH = "/home/parkwooyeol/workspace/Resnet-Experiment/train_val_curves.png"
SUMMARY_IMG_PATH = "/home/parkwooyeol/workspace/Resnet-Experiment/model_summary.png"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ----------------- 데이터셋 분리 -----------------
def prepare_dataset(src_dir, dst_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if os.path.exists(dst_dir):
        print(f"{dst_dir} already exists, skip splitting.")
        return

    classes = os.listdir(src_dir)
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(src_dir, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split, files in splits.items():
            for f in files:
                src_path = os.path.join(cls_dir, f)
                dst_path = os.path.join(dst_dir, split, cls, f)
                shutil.copy2(src_path, dst_path)

    print(f"Dataset prepared at {dst_dir}")

prepare_dataset(RAW_DATA_DIR, DATASET_DIR)

# ----------------- Transforms -----------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ----------------- Dataset / DataLoader -----------------
train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "val"), transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "test"), transform=val_test_transforms)

num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes} | Count={num_classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ----------------- Model -----------------
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ----------------- Training -----------------
best_wts = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
epochs_no_improve = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

def evaluate(loader):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            _, preds = torch.max(out, 1)
            loss_sum += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            n += x.size(0)
    return loss_sum/n, correct/n

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", ncols=100)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(out, 1)
        loss_sum += loss.item()*x.size(0)
        correct += (preds==y).sum().item()
        n += x.size(0)
        pbar.set_postfix({"loss": f"{loss_sum/n:.4f}", "acc": f"{correct/n:.4f}"})

    train_loss, train_acc = loss_sum/n, correct/n
    val_loss, val_acc = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_wts = copy.deepcopy(model.state_dict())
        torch.save({"model": best_wts, "classes": train_dataset.classes}, BEST_MODEL_PATH)
        print("  --> Best model saved")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping!")
            break

model.load_state_dict(best_wts)

# ----------------- 학습 곡선 -----------------
epochs = len(history["train_loss"])
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1,epochs+1), history["train_loss"], label="train_loss")
plt.plot(range(1,epochs+1), history["val_loss"], label="val_loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(range(1,epochs+1), history["train_acc"], label="train_acc")
plt.plot(range(1,epochs+1), history["val_acc"], label="val_acc")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Saved curves: {PLOT_PATH}")

# ----------------- Test -----------------
test_loss, test_acc = evaluate(test_loader)
print(f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")

# ----------------- 모델 summary 저장 -----------------
f = io.StringIO()
with redirect_stdout(f):
    summary(model, input_size=(3, IMG_SIZE, IMG_SIZE), device=str(DEVICE))
summary_str = f.getvalue()

plt.figure(figsize=(12, 8))
plt.axis("off")
plt.text(0.0, 1.0, summary_str, fontfamily="monospace", fontsize=8, va="top")
plt.savefig(SUMMARY_IMG_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved model summary image: {SUMMARY_IMG_PATH}")