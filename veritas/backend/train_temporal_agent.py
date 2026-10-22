"""
VERITAS Temporal Agent — Kaggle Training Script
================================================
Architecture : R3D-18  (torchvision 3D ResNet-18, Kinetics-400 pretrained)
Head         : model.fc = nn.Linear(512, 1)   →  sigmoid  →  P(FAKE)
Input        : (B, 3, 16, 112, 112)
Normalisation: mean=[0.43, 0.40, 0.37]  std=[0.15, 0.14, 0.14]
Loss         : BCEWithLogitsLoss (class-weighted for imbalance)

Dataset      : FakeAVCeleb v1.2  (MP4 videos)
Labels       : RealVideo-* → 0    FakeVideo-* → 1

Checkpoint saved as:
    {
        "epoch": int,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": {...},
        "model_name": "r3d_18",
        "num_classes": 1,
        "class_names": {"0": "REAL", "1": "FAKE"},
        "num_frames": 16,
        "frame_size": 112,
    }

This is 100% compatible with temporal_agent.py which loads via:
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

Usage on Kaggle
---------------
1. Add the FakeAVCeleb dataset to your notebook
2. Set DATASET_ROOT below to match your Kaggle dataset path
3. Run all cells (GPU T4/P100 recommended, ~2-4 hours total)
4. Download temporal_agent.pth from /kaggle/working/checkpoints/
5. Rename to temporal_agent_checkpoint.pth
6. Place in veritas/backend/models/
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
from torchvision.models.video import r3d_18

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these for your Kaggle environment
# ══════════════════════════════════════════════════════════════════════════════

# Path to the directory containing the 4 category folders
# (FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio, RealVideo-RealAudio)
DATASET_ROOT = Path(
    "/kaggle/input/datasets/shreyaty08/fakeavceleb/"
    "FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)

OUTPUT_DIR   = Path("/kaggle/working/checkpoints")
NUM_FRAMES   = 16
FRAME_SIZE   = (112, 112)        # H x W
MEAN         = np.array([0.43, 0.40, 0.37], dtype=np.float32)
STD          = np.array([0.15, 0.14, 0.14], dtype=np.float32)

BATCH_SIZE   = 8                 # safe for T4 (16 GB)
NUM_WORKERS  = 2                 # Kaggle gives 4 CPUs; keep 2 free
NUM_EPOCHS   = 12
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SAMPLES_PER_EPOCH = 4000         # balanced sampling per epoch
VAL_RATIO    = 0.15
SEED         = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("temporal_train")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_videos(root: Path):
    """Walk the dataset and return (list_of_paths, list_of_labels)."""
    LABEL_MAP = {
        "RealVideo-RealAudio": 0,
        "RealVideo-FakeAudio": 0,   # video is real
        "FakeVideo-RealAudio": 1,
        "FakeVideo-FakeAudio": 1,
    }
    paths, labels = [], []
    for category, label in LABEL_MAP.items():
        cat_dir = root / category
        if not cat_dir.exists():
            log.warning("Category dir missing: %s", cat_dir)
            continue
        for mp4 in sorted(cat_dir.rglob("*.mp4")):
            paths.append(str(mp4))
            labels.append(label)
    return paths, labels


def train_val_split(paths, labels, val_ratio=0.15, seed=42):
    """Stratified random split."""
    rng = random.Random(seed)
    real_idx = [i for i, l in enumerate(labels) if l == 0]
    fake_idx = [i for i, l in enumerate(labels) if l == 1]
    rng.shuffle(real_idx)
    rng.shuffle(fake_idx)

    n_real_val = max(1, int(len(real_idx) * val_ratio))
    n_fake_val = max(1, int(len(fake_idx) * val_ratio))

    val_idx  = set(real_idx[:n_real_val] + fake_idx[:n_fake_val])
    train_idx = [i for i in range(len(paths)) if i not in val_idx]
    val_idx   = sorted(val_idx)

    tp = [paths[i] for i in train_idx];  tl = [labels[i] for i in train_idx]
    vp = [paths[i] for i in val_idx];    vl = [labels[i] for i in val_idx]
    return tp, tl, vp, vl


# ══════════════════════════════════════════════════════════════════════════════
# 2. VIDEO FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, num_frames: int = 16,
                   frame_size=(112, 112), augment=False) -> Optional[np.ndarray]:
    """
    Extract num_frames evenly-spaced frames from a video.

    Returns (num_frames, 3, H, W) float32 normalised array, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        # Small random jitter during training
        if augment and total > num_frames * 2:
            jitter = np.random.randint(-2, 3, size=num_frames)
            indices = np.clip(indices + jitter, 0, total - 1)

        flip = augment and random.random() < 0.5

        frames = []
        last_good = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, bgr = cap.read()
            if ret and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, frame_size)
                last_good = rgb
            else:
                if last_good is None:
                    continue
                rgb = last_good
            if flip:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            frames.append(rgb)

        if not frames:
            return None
        while len(frames) < num_frames:
            frames.append(frames[-1])

        stack = np.stack(frames[:num_frames]).astype(np.float32) / 255.0
        stack = (stack - MEAN) / STD
        stack = stack.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        return stack
    except Exception:
        return None
    finally:
        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 3. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FakeAVCelebTemporalDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = extract_frames(
            self.paths[idx], NUM_FRAMES, FRAME_SIZE, augment=self.augment
        )
        if frames is None:
            # Return zeros on failure — the loss will still work
            frames = np.zeros((NUM_FRAMES, 3, *FRAME_SIZE), dtype=np.float32)

        # frames: (16, 3, 112, 112)  →  tensor (3, 16, 112, 112) for R3D-18
        tensor = torch.from_numpy(frames)              # (16, 3, 112, 112)
        tensor = tensor.permute(1, 0, 2, 3)            # (3, 16, 112, 112)
        label  = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    """R3D-18 with Kinetics-400 pretrained backbone, binary head."""
    try:
        from torchvision.models.video import R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        log.info("Loaded R3D-18 with Kinetics-400 pretrained weights (new API).")
    except (ImportError, AttributeError):
        model = r3d_18(pretrained=True)
        log.info("Loaded R3D-18 with Kinetics-400 pretrained weights (legacy API).")

    # Freeze stem + layer1-2 for first few epochs (fine-tune layer3-4 + fc)
    for name, param in model.named_parameters():
        if name.startswith(("stem", "layer1", "layer2")):
            param.requires_grad = False

    # Replace final FC: 512 → 1  (binary sigmoid output)
    model.fc = nn.Linear(512, 1)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    model.to(device)
    log.info("Model head: fc = Linear(512, 1)")
    log.info("Frozen: stem, layer1, layer2.  Trainable: layer3, layer4, fc")
    return model


def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning (call after warmup epochs)."""
    for param in model.parameters():
        param.requires_grad = True
    log.info("All layers unfrozen for full fine-tuning.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def make_sampler(labels, num_samples):
    """WeightedRandomSampler for class-balanced epochs."""
    counts = np.bincount(labels)
    weight_per_class = 1.0 / counts
    sample_weights = [weight_per_class[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (clips, labels) in enumerate(loader):
        clips  = clips.to(device, non_blocking=True)    # (B, 3, 16, 112, 112)
        labels = labels.to(device, non_blocking=True)    # (B,)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(clips).squeeze(-1)            # (B,)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * clips.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += clips.size(0)

        if (batch_idx + 1) % 50 == 0:
            log.info(
                "  Epoch %d  batch %d/%d  loss=%.4f  acc=%.3f",
                epoch, batch_idx + 1, len(loader),
                running_loss / total, correct / total,
            )

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(clips).squeeze(-1)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * clips.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += clips.size(0)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().long().tolist())

    acc  = correct / max(total, 1)
    loss = running_loss / max(total, 1)
    return loss, acc, all_probs, all_labels


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "scheduler":        scheduler.state_dict(),
        "metrics":          metrics,
        "model_name":       "r3d_18",
        "num_classes":      1,
        "class_names":      {"0": "REAL", "1": "FAKE"},
        "num_frames":       NUM_FRAMES,
        "frame_size":       FRAME_SIZE[0],
    }, path)
    log.info("Checkpoint saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Discover videos ───────────────────────────────────────────────────────
    log.info("Scanning dataset at %s ...", DATASET_ROOT)
    all_paths, all_labels = discover_videos(DATASET_ROOT)
    n_real = sum(1 for l in all_labels if l == 0)
    n_fake = sum(1 for l in all_labels if l == 1)
    log.info("Found %d videos  (REAL=%d  FAKE=%d)", len(all_paths), n_real, n_fake)

    if len(all_paths) == 0:
        log.error("No videos found. Check DATASET_ROOT: %s", DATASET_ROOT)
        return

    # ── Split ─────────────────────────────────────────────────────────────────
    train_p, train_l, val_p, val_l = train_val_split(
        all_paths, all_labels, val_ratio=VAL_RATIO, seed=SEED
    )
    log.info("Train: %d  Val: %d", len(train_p), len(val_p))
    log.info("Train REAL=%d  FAKE=%d", sum(1 for l in train_l if l==0), sum(1 for l in train_l if l==1))
    log.info("Val   REAL=%d  FAKE=%d", sum(1 for l in val_l if l==0), sum(1 for l in val_l if l==1))

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = FakeAVCelebTemporalDataset(train_p, train_l, augment=True)
    val_ds   = FakeAVCelebTemporalDataset(val_p,   val_l,   augment=False)

    train_sampler = make_sampler(train_l, num_samples=SAMPLES_PER_EPOCH)
    train_loader  = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(device)

    # ── Class-weighted loss ───────────────────────────────────────────────────
    # pos_weight = n_real / n_fake  (< 1 because FAKE is overrepresented)
    # This downweights FAKE relative to REAL in the loss
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info("BCEWithLogitsLoss  pos_weight=%.4f", pos_weight.item())

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    UNFREEZE_EPOCH = 4  # unfreeze all layers after this many warmup epochs

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == UNFREEZE_EPOCH:
            unfreeze_all(model)
            # Rebuild optimizer with all params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR * 0.3, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH + 1, eta_min=1e-6,
            )

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  lr=%.2e  time=%.0fs",
            epoch, NUM_EPOCHS, train_loss, train_acc,
            val_loss, val_acc, optimizer.param_groups[0]["lr"], elapsed,
        )

        # Save best
        metrics = {
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                OUTPUT_DIR / "temporal_agent_best.pth",
            )

        # Save latest every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch, metrics,
            OUTPUT_DIR / "temporal_agent_latest.pth",
        )

    log.info("Training complete.  Best val_acc = %.4f", best_val_acc)

    # ── Plot training curves ──────────────────────────────────────────────────
    if HAS_PLT:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, NUM_EPOCHS + 1)
        ax1.plot(epochs_range, history["train_loss"], label="Train")
        ax1.plot(epochs_range, history["val_loss"],   label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(epochs_range, history["train_acc"],  label="Train")
        ax2.plot(epochs_range, history["val_acc"],    label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
        log.info("Training curves saved to %s", OUTPUT_DIR / "training_curves.png")
        plt.show()

    # ── Final verification ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("VERIFICATION")
    log.info("=" * 60)
    ckpt = torch.load(str(OUTPUT_DIR / "temporal_agent_best.pth"), map_location="cpu")
    log.info("Checkpoint keys: %s", list(ckpt.keys()))
    log.info("model_name: %s", ckpt.get("model_name"))
    log.info("class_names: %s", ckpt.get("class_names"))
    log.info("num_frames: %s", ckpt.get("num_frames"))
    log.info("epoch: %s", ckpt.get("epoch"))
    log.info("metrics: %s", ckpt.get("metrics"))

    # Verify load into fresh model (same as temporal_agent.py does)
    test_model = r3d_18(pretrained=False)
    test_model.fc = nn.Linear(512, 1)
    state = ckpt.get("model_state_dict", ckpt)
    result = test_model.load_state_dict(state, strict=True)
    log.info("load_state_dict strict=True  missing=%s  unexpected=%s",
             result.missing_keys, result.unexpected_keys)
    test_model.eval()

    # Dummy inference
    dummy = torch.randn(1, 3, 16, 112, 112)
    with torch.no_grad():
        logit = test_model(dummy)
        prob  = torch.sigmoid(logit.squeeze()).item()
    log.info("Dummy inference:  logit=%.4f  sigmoid=%.4f  class=%s",
             logit.item(), prob, "FAKE" if prob >= 0.5 else "REAL")
    log.info("=" * 60)
    log.info("DONE — copy temporal_agent_best.pth to:")
    log.info("  veritas/backend/models/temporal_agent_checkpoint.pth")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
