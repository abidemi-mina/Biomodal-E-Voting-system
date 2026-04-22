"""
train_liveness.py
=================
Train the MobileNetV2-based liveness detector.

Usage:
    python train_liveness.py --data_dir /path/to/dataset --epochs 30 --output media/liveness_model.pth

Dataset Structure:
    data_dir/
        train/
            live/        (real face photos/frames)
            spoof/       (printed photos, screen replays, 3D masks)
        val/
            live/
            spoof/

Recommended public datasets:
    - CelebA-Spoof: https://github.com/ZhangYuanhan-AI/CelebA-Spoof
    - LCC-FASD:     https://csit.am/2019/proceedings/PRIP/PRIP3.pdf
    - CASIA-FASD:   https://www.cbsr.ia.ac.cn/users/jjyan/ZHANG-ICB2012.pdf
    - OULU-NPU:     https://sites.google.com/site/oulunpudatabase/

Run with Nigerian environment augmentation for best in-field performance.
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Model ──────────────────────────────────────────────────────
def build_model(pretrained=True):
    """
    MobileNetV2 with custom classifier head.
    Output: 2 classes (spoof=0, live=1)
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze early layers, fine-tune last few blocks + classifier
    for i, (name, param) in enumerate(model.features.named_parameters()):
        param.requires_grad = i >= 12 * 3  # Unfreeze last ~4 blocks

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return model.to(DEVICE)


# ── Transforms ─────────────────────────────────────────────────
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        # Simulate different screen types / print quality
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


# ── Training loop ───────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


# ── Main ────────────────────────────────────────────────────────
def main(args):
    logger.info(f"Training on: {DEVICE}")
    train_tf, val_tf = get_transforms()

    train_ds = ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_tf)
    val_ds   = ImageFolder(os.path.join(args.data_dir, 'val'),   transform=val_tf)

    # Class weights to handle imbalance (more live than spoof samples)
    class_counts = np.bincount(train_ds.targets)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
    logger.info(f"Dataset: {len(train_ds)} train | {len(val_ds)} val | classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = build_model(pretrained=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    best_auc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, auc = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), output_path)
            logger.info(f"  ✅ Best model saved (AUC={auc:.4f}) → {output_path}")

    logger.info(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Liveness Detector')
    parser.add_argument('--data_dir',   type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output',     type=str, default='media/liveness_model.pth')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--workers',    type=int, default=4)
    args = parser.parse_args()
    main(args)
