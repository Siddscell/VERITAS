"""
VERITAS Spatial Agent — Kaggle Training Script
===============================================
Architecture : EfficientNet-B0 (ImageNet pretrained) + custom 2-class head
Head         : Dropout(0.3) → Linear(1280,256) → ReLU → Dropout(0.2) → Linear(256,2)
Input        : (B, 3, 224, 224)   RGB, ImageNet-normalised
Output       : softmax over 2 classes  →  class 1 = FAKE
Loss         : Class-weighted CrossEntropyLoss (dataset is FAKE-heavy)

Dataset      : FakeAVCeleb v1.2  (MP4 videos → sampled frames)
Labels       : RealVideo-* → 0 (REAL)    FakeVideo-* → 1 (FAKE)

Checkpoint saved as (MUST match spatial_agent.py which loads key "state_dict", strict=True):
    {
        "epoch": int,
        "state_dict": model.state_dict(),      # ← key is "state_dict"
        "optimizer": ...,
        "scheduler": ...,
        "metrics": {...},
        "model_name": "efficientnet_b0",
        "num_classes": 2,
        "class_names": {"0": "REAL", "1": "FAKE"},
        "image_size": 224,
        "category_labels": {...},
        "class_weights": [...],
    }

Usage on Kaggle
---------------
1. Create a new Kaggle Notebook, add the FakeAVCeleb dataset, enable GPU (T4/P100).
2. Verify DATASET_ROOT below matches the attached dataset path.
3. Run all. (~2-4 h on T4.)
4. Download /kaggle/working/checkpoints/spatial_agent_best.pth
5. Rename to  spatial_agent.pth  and place in  veritas/backend/models/
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import efficientnet_b0

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these for your Kaggle environment
# ══════════════════════════════════════════════════════════════════════════════

DATASET_ROOT = Path(
    "/kaggle/input/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)
# NOTE: Kaggle mounts datasets at /kaggle/input/<slug>/... . If discovery finds 0
# videos, print the tree with:  !find /kaggle/input -maxdepth 3 -type d
# and adjust DATASET_ROOT to the folder that directly contains the 4 category dirs.

OUTPUT_DIR   = Path("/kaggle/working/checkpoints")
IMAGE_SIZE   = 224
FRAMES_PER_VIDEO = 4             # frames sampled per video (middle-biased)

BATCH_SIZE   = 32                # EfficientNet-B0 @ 224px is light; safe on T4
NUM_WORKERS  = 2
NUM_EPOCHS   = 8
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SAMPLES_PER_EPOCH = 12000        # balanced sampling per epoch (frame-level)
VAL_RATIO    = 0.15
SEED         = 42

# ImageNet normalisation — MUST match spatial_agent.py _TRANSFORM
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CATEGORY_LABELS = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 0,   # video is real
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("spatial_train")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET DISCOVERY  (identity-aware split to prevent leakage)
# ══════════════════════════════════════════════════════════════════════════════

def discover_videos(root: Path):
    """Walk the dataset → (paths, labels, identities)."""
    paths, labels, identities = [], [], []
    for category, label in CATEGORY_LABELS.items():
        cat_dir = root / category
        if not cat_dir.exists():
            log.warning("Category dir missing: %s", cat_dir)
            continue
        for mp4 in sorted(cat_dir.rglob("*.mp4")):
            paths.append(str(mp4))
            labels.append(label)
            # FakeAVCeleb encodes identity in the path (e.g. .../id00018/...).
            # Use the parent folder chain to group clips of the same person.
            ident = _identity_from_path(mp4, cat_dir)
            identities.append(ident)
    return paths, labels, identities


def _identity_from_path(mp4: Path, cat_dir: Path) -> str:
    """Best-effort identity key from the path under the category dir."""
    rel = mp4.relative_to(cat_dir)
    # First path component under the category is typically the identity bucket.
    return rel.parts[0] if len(rel.parts) > 1 else mp4.stem


def identity_aware_split(paths, labels, identities, val_ratio=0.15, seed=42):
    """Split so that no identity appears in both train and val."""
    rng = random.Random(seed)
    by_ident: dict[str, list[int]] = {}
    for i, ident in enumerate(identities):
        by_ident.setdefault(ident, []).append(i)

    idents = list(by_ident.keys())
    rng.shuffle(idents)
    n_val = max(1, int(len(idents) * val_ratio))
    val_idents = set(idents[:n_val])

    train_idx, val_idx = [], []
    for ident, idxs in by_ident.items():
        (val_idx if ident in val_idents else train_idx).extend(idxs)

    tp = [paths[i] for i in train_idx];  tl = [labels[i] for i in train_idx]
    vp = [paths[i] for i in val_idx];    vl = [labels[i] for i in val_idx]
    return tp, tl, vp, vl


# ══════════════════════════════════════════════════════════════════════════════
# 2. FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, n: int, size: int, augment=False) -> list[np.ndarray]:
    """
    Sample `n` frames from the middle 80% of the video (faces are most stable there).
    Returns a list of (3, H, W) float32 normalised arrays (may be shorter than n).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        lo, hi = int(total * 0.1), int(total * 0.9)
        hi = max(hi, lo + 1)
        indices = np.linspace(lo, hi - 1, n, dtype=int)
        if augment:
            jitter = np.random.randint(-2, 3, size=len(indices))
            indices = np.clip(indices + jitter, 0, total - 1)

        flip = augment and random.random() < 0.5
        out = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, bgr = cap.read()
            if not ret or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            if flip:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - MEAN) / STD
            out.append(arr.transpose(2, 0, 1))  # (3, H, W)
        return out
    except Exception:
        return []
    finally:
        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 3. PYTORCH DATASET  (one frame per __getitem__, video chosen by sampler)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialFrameDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = extract_frames(self.paths[idx], FRAMES_PER_VIDEO, IMAGE_SIZE, self.augment)
        if not frames:
            frame = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            frame = random.choice(frames)  # one representative frame per sample
        tensor = torch.from_numpy(frame)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL — EfficientNet-B0 with the EXACT head spatial_agent.py rebuilds
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    try:
        from torchvision.models import EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        log.info("Loaded EfficientNet-B0 ImageNet weights (new API).")
    except (ImportError, AttributeError):
        model = efficientnet_b0(pretrained=True)
        log.info("Loaded EfficientNet-B0 ImageNet weights (legacy API).")

    # MUST match spatial_agent._build_model() exactly for strict=True load.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, 2),
    )
    model.to(device)
    log.info("Head: Dropout(.3)→Linear(1280,256)→ReLU→Dropout(.2)→Linear(256,2)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VAL
# ══════════════════════════════════════════════════════════════════════════════

def make_sampler(labels, num_samples):
    counts = np.bincount(labels)
    w_per_class = 1.0 / np.maximum(counts, 1)
    sample_weights = [w_per_class[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running, correct, total = 0.0, 0, 0
    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total   += x.size(0)
        if (bi + 1) % 50 == 0:
            log.info("  Epoch %d  batch %d/%d  loss=%.4f  acc=%.3f",
                     epoch, bi + 1, len(loader), running / total, correct / total)
    return running / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        running += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(FAKE)
        correct += (logits.argmax(1) == y).sum().item()
        total   += x.size(0)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return running / max(total, 1), correct / max(total, 1), all_probs, all_labels


def compute_metrics(probs, labels, loss):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / max(len(labels), 1)
    m = {"acc": acc, "loss": loss}
    if HAS_SKLEARN and len(set(labels)) > 1:
        m["precision"] = float(precision_score(labels, preds, zero_division=0))
        m["recall"]    = float(recall_score(labels, preds, zero_division=0))
        m["f1"]        = float(f1_score(labels, preds, zero_division=0))
        try:
            m["auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            m["auc"] = float("nan")
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        m.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return m


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, class_weights, path):
    torch.save({
        "epoch":            epoch,
        "state_dict":       model.state_dict(),      # ← spatial_agent.py reads "state_dict"
        "optimizer":        optimizer.state_dict(),
        "scheduler":        scheduler.state_dict(),
        "metrics":          metrics,
        "model_name":       "efficientnet_b0",
        "num_classes":      2,
        "class_names":      {"0": "REAL", "1": "FAKE"},
        "image_size":       IMAGE_SIZE,
        "category_labels":  CATEGORY_LABELS,
        "class_weights":    class_weights,
        "dataset_root":     str(DATASET_ROOT),
    }, path)
    log.info("Checkpoint saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Scanning dataset at %s ...", DATASET_ROOT)
    paths, labels, idents = discover_videos(DATASET_ROOT)
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    log.info("Found %d videos  (REAL=%d  FAKE=%d)  unique identities=%d",
             len(paths), n_real, n_fake, len(set(idents)))
    if not paths:
        log.error("No videos found. Fix DATASET_ROOT: %s", DATASET_ROOT)
        return

    train_p, train_l, val_p, val_l = identity_aware_split(paths, labels, idents, VAL_RATIO, SEED)
    log.info("Train videos=%d  Val videos=%d", len(train_p), len(val_p))

    train_ds = SpatialFrameDataset(train_p, train_l, augment=True)
    val_ds   = SpatialFrameDataset(val_p,   val_l,   augment=False)

    train_sampler = make_sampler(train_l, SAMPLES_PER_EPOCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(device)

    # Class-weighted CE. FAKE-heavy dataset → upweight REAL (class 0).
    counts = np.bincount(train_l, minlength=2).astype(np.float32)
    inv = counts.sum() / np.maximum(counts, 1)
    class_weights = (inv / inv.sum() * 2).tolist()  # normalised, mean≈1
    log.info("Class weights [REAL, FAKE] = %s", class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        va_loss, va_acc, probs, labs = validate(model, val_loader, criterion, device)
        scheduler.step()
        metrics = compute_metrics(probs, labs, va_loss)
        metrics.update(train_loss=tr_loss, train_acc=tr_acc)
        log.info("Epoch %d/%d  train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f f1=%.4f auc=%.4f  (%.0fs)",
                 epoch, NUM_EPOCHS, tr_loss, tr_acc, va_loss, va_acc,
                 metrics.get("f1", float("nan")), metrics.get("auc", float("nan")), time.time() - t0)

        save_checkpoint(model, optimizer, scheduler, epoch, metrics, class_weights,
                        OUTPUT_DIR / "spatial_agent_latest.pth")
        score = metrics.get("f1", va_acc)
        if score > best_f1:
            best_f1 = score
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, class_weights,
                            OUTPUT_DIR / "spatial_agent_best.pth")
            log.info("  ↑ new best (f1/acc=%.4f)", best_f1)

    log.info("Training complete. Best val f1/acc = %.4f", best_f1)
    _verify(OUTPUT_DIR / "spatial_agent_best.pth", device)


def _verify(ckpt_path, device):
    """Reload exactly as spatial_agent.py does — proves format compatibility."""
    log.info("=" * 60); log.info("VERIFICATION"); log.info("=" * 60)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    log.info("keys: %s", list(ckpt.keys()))
    assert "state_dict" in ckpt, "checkpoint missing 'state_dict'"

    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.2), nn.Linear(256, 2),
    )
    res = m.load_state_dict(ckpt["state_dict"], strict=True)
    log.info("load_state_dict strict=True  missing=%s  unexpected=%s",
             res.missing_keys, res.unexpected_keys)
    m.eval()
    with torch.no_grad():
        out = torch.softmax(m(torch.randn(1, 3, 224, 224)), dim=1)
    log.info("Dummy inference P(FAKE)=%.4f", out[0, 1].item())
    log.info("DONE — rename spatial_agent_best.pth → spatial_agent.pth → veritas/backend/models/")


if __name__ == "__main__":
    main()
