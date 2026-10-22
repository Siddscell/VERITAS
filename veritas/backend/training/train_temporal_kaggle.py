"""
VERITAS Temporal Agent — Kaggle Training Script (corrected)
============================================================
Architecture : R3D-18  (torchvision 3D ResNet-18, Kinetics-400 pretrained)
Head         : model.fc = nn.Linear(512, 1)  →  sigmoid  →  P(FAKE)
Input        : (B, 3, 16, 112, 112)
Normalisation: mean=[0.43216, 0.39467, 0.37645]  std=[0.22803, 0.22145, 0.21699]
Loss         : BCEWithLogitsLoss (class-weighted for imbalance)

Dataset      : FakeAVCeleb v1.2  (MP4 videos)
Labels       : RealVideo-* → 0    FakeVideo-* → 1

WHY THIS SUPERSEDES backend/train_temporal_agent.py
---------------------------------------------------
The old repo script NEVER saved `normalization` or `decision_threshold`, but
temporal_agent.py reads exactly those keys at load time (falling back to a 0.5
threshold and losing the training-matched normalisation). This script saves the
FULL metadata block so inference preprocessing always matches training, and it
selects the decision threshold on the validation set (best-F1 sweep) — the
shipped checkpoint used ~0.05, reflecting FakeAVCeleb's heavy FAKE skew.

Checkpoint saved as (matches temporal_agent.py — key "model_state_dict", strict=True):
    {
        "epoch", "model_state_dict", "optimizer", "scheduler", "metrics",
        "model_name": "r3d_18", "num_classes": 1,
        "class_names": {"0": "REAL", "1": "FAKE"},
        "num_frames": 16, "frame_size": 112,
        "target_fps": 8.0, "window_seconds": 2.0,
        "normalization": {"mean": [...], "std": [...]},
        "decision_threshold": <val-selected>,
        "category_labels": {...},
    }

Usage on Kaggle
---------------
1. New Kaggle Notebook → add FakeAVCeleb dataset → enable GPU (T4/P100).
2. Verify DATASET_ROOT below.
3. Run all (~2-4 h on T4).
4. Download /kaggle/working/checkpoints/temporal_agent_best.pth
5. Rename to  temporal_agent_checkpoint.pth  → veritas/backend/models/
"""

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
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATASET_ROOT = Path(
    "/kaggle/input/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
)

OUTPUT_DIR   = Path("/kaggle/working/checkpoints")
NUM_FRAMES   = 16
FRAME_SIZE   = (112, 112)        # H x W
TARGET_FPS   = 8.0
WINDOW_SECS  = 2.0

# Checkpoint-authoritative normalisation (matches temporal_agent.py fallbacks).
MEAN = np.array([0.43216, 0.39467, 0.37645], dtype=np.float32)
STD  = np.array([0.22803, 0.22145, 0.21699], dtype=np.float32)

BATCH_SIZE   = 8                 # safe for T4 (16 GB)
NUM_WORKERS  = 2
NUM_EPOCHS   = 12
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SAMPLES_PER_EPOCH = 4000
VAL_RATIO    = 0.15
SEED         = 42
UNFREEZE_EPOCH = 4

CATEGORY_LABELS = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 0,   # video is real
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("temporal_train")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DISCOVERY (identity-aware split)
# ══════════════════════════════════════════════════════════════════════════════

def discover_videos(root: Path):
    paths, labels, idents = [], [], []
    for category, label in CATEGORY_LABELS.items():
        cat_dir = root / category
        if not cat_dir.exists():
            log.warning("Category dir missing: %s", cat_dir)
            continue
        for mp4 in sorted(cat_dir.rglob("*.mp4")):
            paths.append(str(mp4))
            labels.append(label)
            rel = mp4.relative_to(cat_dir)
            idents.append(rel.parts[0] if len(rel.parts) > 1 else mp4.stem)
    return paths, labels, idents


def identity_aware_split(paths, labels, idents, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_ident: dict[str, list[int]] = {}
    for i, ident in enumerate(idents):
        by_ident.setdefault(ident, []).append(i)
    keys = list(by_ident.keys())
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])
    tr, va = [], []
    for k, idxs in by_ident.items():
        (va if k in val_keys else tr).extend(idxs)
    return ([paths[i] for i in tr], [labels[i] for i in tr],
            [paths[i] for i in va], [labels[i] for i in va])


# ══════════════════════════════════════════════════════════════════════════════
# 2. FRAME EXTRACTION  (matches temporal_agent._extract_frames sampling)
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, augment=False) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        window = min(total, int(WINDOW_SECS * fps))
        window = max(window, NUM_FRAMES)
        indices = np.linspace(0, window - 1, NUM_FRAMES, dtype=int)
        if augment and total > NUM_FRAMES * 2:
            jitter = np.random.randint(-2, 3, size=NUM_FRAMES)
            indices = np.clip(indices + jitter, 0, total - 1)

        flip = augment and random.random() < 0.5
        frames, last = [], None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, bgr = cap.read()
            if ret and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
                last = rgb
            elif last is not None:
                rgb = last
            else:
                continue
            if flip:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            frames.append(rgb)

        if not frames:
            return None
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])

        stack = np.stack(frames[:NUM_FRAMES]).astype(np.float32) / 255.0
        stack = (stack - MEAN) / STD
        stack = stack.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        return stack
    except Exception:
        return None
    finally:
        cap.release()


class TemporalDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels, self.augment = paths, labels, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = extract_frames(self.paths[idx], augment=self.augment)
        if frames is None:
            frames = np.zeros((NUM_FRAMES, 3, *FRAME_SIZE), dtype=np.float32)
        tensor = torch.from_numpy(frames).permute(1, 0, 2, 3)  # (3, 16, 112, 112)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    try:
        from torchvision.models.video import R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        log.info("Loaded R3D-18 Kinetics-400 weights (new API).")
    except (ImportError, AttributeError):
        model = r3d_18(pretrained=True)
        log.info("Loaded R3D-18 Kinetics-400 weights (legacy API).")

    for name, p in model.named_parameters():
        if name.startswith(("stem", "layer1", "layer2")):
            p.requires_grad = False

    model.fc = nn.Linear(512, 1)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    model.to(device)
    log.info("Head fc=Linear(512,1). Frozen stem/layer1/layer2; trainable layer3/layer4/fc.")
    return model


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True
    log.info("All layers unfrozen.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / VAL
# ══════════════════════════════════════════════════════════════════════════════

def make_sampler(labels, num_samples):
    counts = np.bincount(labels)
    w = 1.0 / np.maximum(counts, 1)
    return WeightedRandomSampler([w[l] for l in labels], num_samples=num_samples, replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running, correct, total = 0.0, 0, 0
    for bi, (clips, labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(clips).squeeze(-1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * clips.size(0)
        correct += ((torch.sigmoid(logits) >= 0.5).long() == labels.long()).sum().item()
        total += clips.size(0)
        if (bi + 1) % 50 == 0:
            log.info("  Epoch %d  batch %d/%d  loss=%.4f  acc=%.3f",
                     epoch, bi + 1, len(loader), running / total, correct / total)
    return running / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running, total = 0.0, 0
    probs, labs = [], []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(clips).squeeze(-1)
            loss = criterion(logits, labels)
        running += loss.item() * clips.size(0)
        total += clips.size(0)
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
        labs.extend(labels.long().cpu().tolist())
    return running / max(total, 1), probs, labs


def best_threshold(probs, labels):
    """Sweep thresholds, return the one maximising F1 (falls back to 0.5)."""
    if not HAS_SKLEARN or len(set(labels)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = [1 if p >= t else 0 for p in probs]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 4)


def compute_metrics(probs, labels, threshold, loss):
    preds = [1 if p >= threshold else 0 for p in probs]
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / max(len(labels), 1)
    m = {"accuracy": acc, "loss": loss, "threshold": threshold}
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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, threshold, path):
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),      # ← temporal_agent.py reads this
        "optimizer":          optimizer.state_dict(),
        "scheduler":          scheduler.state_dict(),
        "metrics":            metrics,
        "model_name":         "r3d_18",
        "num_classes":        1,
        "class_names":        {"0": "REAL", "1": "FAKE"},
        "num_frames":         NUM_FRAMES,
        "frame_size":         FRAME_SIZE[0],
        "target_fps":         TARGET_FPS,
        "window_seconds":     WINDOW_SECS,
        # ── The fields the old script forgot — inference reads these ──
        "normalization":      {"mean": MEAN.tolist(), "std": STD.tolist()},
        "decision_threshold": threshold,
        "category_labels":    CATEGORY_LABELS,
        "identity_aware_split": True,
        "dataset_root":       str(DATASET_ROOT),
    }, path)
    log.info("Checkpoint saved → %s  (threshold=%.4f)", path, threshold)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
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
    log.info("Found %d videos (REAL=%d FAKE=%d) identities=%d",
             len(paths), n_real, n_fake, len(set(idents)))
    if not paths:
        log.error("No videos found. Fix DATASET_ROOT: %s", DATASET_ROOT)
        return

    tr_p, tr_l, va_p, va_l = identity_aware_split(paths, labels, idents, VAL_RATIO, SEED)
    log.info("Train=%d  Val=%d", len(tr_p), len(va_p))

    train_ds = TemporalDataset(tr_p, tr_l, augment=True)
    val_ds   = TemporalDataset(va_p, va_l, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=make_sampler(tr_l, SAMPLES_PER_EPOCH),
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(device)
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info("BCEWithLogitsLoss pos_weight=%.4f", pos_weight.item())

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == UNFREEZE_EPOCH:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.3, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - UNFREEZE_EPOCH + 1, eta_min=1e-6)

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        va_loss, probs, labs = validate(model, val_loader, criterion, device)
        scheduler.step()

        thr = best_threshold(probs, labs)
        metrics = compute_metrics(probs, labs, thr, va_loss)
        metrics.update(train_loss=tr_loss, train_acc=tr_acc)
        log.info("Epoch %d/%d  train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f f1=%.4f auc=%.4f thr=%.3f  (%.0fs)",
                 epoch, NUM_EPOCHS, tr_loss, tr_acc, va_loss, metrics["accuracy"],
                 metrics.get("f1", float("nan")), metrics.get("auc", float("nan")), thr, time.time() - t0)

        save_checkpoint(model, optimizer, scheduler, epoch, metrics, thr,
                        OUTPUT_DIR / "temporal_agent_latest.pth")
        f1 = metrics.get("f1", metrics["accuracy"])
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, thr,
                            OUTPUT_DIR / "temporal_agent_best.pth")
            log.info("  ↑ new best (f1=%.4f)", best_f1)

    log.info("Training complete. Best val f1 = %.4f", best_f1)
    _verify(OUTPUT_DIR / "temporal_agent_best.pth")


def _verify(ckpt_path):
    log.info("=" * 60); log.info("VERIFICATION"); log.info("=" * 60)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    log.info("keys: %s", list(ckpt.keys()))
    for k in ("normalization", "decision_threshold", "num_frames", "target_fps", "window_seconds"):
        assert k in ckpt, f"checkpoint missing '{k}' — inference would fall back!"
        log.info("  %s = %s", k, ckpt[k])
    m = r3d_18(weights=None)
    m.fc = nn.Linear(512, 1)
    res = m.load_state_dict(ckpt["model_state_dict"], strict=True)
    log.info("load_state_dict strict=True missing=%s unexpected=%s", res.missing_keys, res.unexpected_keys)
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.randn(1, 3, 16, 112, 112)).squeeze()).item()
    log.info("Dummy inference P(FAKE)=%.4f", p)
    log.info("DONE — rename temporal_agent_best.pth → temporal_agent_checkpoint.pth → veritas/backend/models/")


if __name__ == "__main__":
    main()
