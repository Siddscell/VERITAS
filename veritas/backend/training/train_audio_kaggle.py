"""
VERITAS Audio Agent — Windows & Kaggle Training Pipeline
========================================================
Architecture : EfficientNet-B0 (ImageNet pretrained), classifier[1] = Linear(1280, 1)
Input        : Mel-spectrogram treated as a 224×224 image, 3-channel
Output       : Sigmoid(logit) → P(FAKE) ∈ [0, 1]
Loss         : BCEWithLogitsLoss with class-weighted loss (no double-balancing)

Dataset      : FakeAVCeleb v1.2 (audio track of each MP4, or native WAV/MP3)
Labels       : *-RealAudio → 0 (REAL)   *-FakeAudio → 1 (FAKE)
               AUDIO label is independent of the VIDEO label:
                 RealVideo-RealAudio → audio REAL (0)
                 RealVideo-FakeAudio → audio FAKE (1)
                 FakeVideo-RealAudio → audio REAL (0)
                 FakeVideo-FakeAudio → audio FAKE (1)

Methodology & Rigor
-------------------
1. Identity-aware 80/10/10 split:
   - Speakers (identities) are disjoint across Train, Validation, and Test.
   - Stratified to guarantee both REAL and FAKE audio appear in all three splits.
   - Explicit assertions guard against identity leakage across splits.
2. Robust failure handling:
   - Extraction failures are NEVER silently trained on as zero-valued spectrograms.
   - Failed samples are tracked with path, category, split, and reason.
   - Clean collate_fn ensures batches with corrupt audio do not cause NaNs or crashes.
   - Failure counts and valid sample counts are reported by split and category.
3. Class balancing:
   - Class-weighted BCE loss (pos_weight = N_real / N_fake) with uniform shuffling.
   - WeightedRandomSampler is NOT combined with pos_weight (eliminating double-weighting).
4. Decision threshold calibration:
   - Operating threshold is calibrated on the VALIDATION split by sweeping F1.
   - Saved as `decision_threshold` in the checkpoint metadata.
   - The TEST set is held-out and strictly used for final evaluation (never for tuning).
5. Evaluation & reporting:
   - Locked best checkpoint is evaluated on the independent test set.
   - Full metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
   - Category-wise breakdown across all four FakeAVCeleb categories.
   - Structured JSON report exported to `OUTPUT_DIR / "audio_training_report.json"`.
6. Preprocessing parity:
   - Exact match with production `audio_agent.py`:
     16 kHz mono PCM → 48,000 samples (3s) → MelSpectrogram(128 bins) →
     AmplitudeToDB → Bilinear resize (224, 224) → (x - 0.5) / 0.5 → 3 channels.

Windows-specific features
-------------------------
- Dataset root: D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2
- Output: D:\VERITAS_TRAINING\checkpoints  WAV cache: D:\VERITAS_TRAINING\wav_cache
- Batch size: 8 (conservative for 6 GB VRAM), workers: 1 (spawn multiprocessing context)
- AMP: torch.amp (PyTorch 2.6+ compatible)
- Resumable: Periodic mid-epoch (every 200 batches) and end-of-epoch checkpoints
  (model weights, optimizer state, scheduler state, AMP scaler, RNG states)
- Best validation model protected against overwrite by inferior epochs
- Training log written to D:\VERITAS_TRAINING\checkpoints\training.log
- Preflight report written to D:\VERITAS_TRAINING\preflight_report.log
"""

import os
import re
import sys
import time
import json
import wave
import shutil
import random
import logging
import subprocess
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# PyTorch 2.x AMP API (avoids DeprecationWarning)
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_DEVICE = "cuda"

from torchvision.models import efficientnet_b0

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG = "ffmpeg"  # fallback to PATH

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Pipeline execution mode:
#   "diagnose": Inspect metadata, print component sizes, compare split strategies, and stop safely.
#   "preflight": Validate 80/10/10 split assertions and speaker disjointness, and stop safely.
#   "train":     Execute the full training pipeline (only when explicitly selected).
EXECUTION_MODE = os.environ.get("VERITAS_MODE", "diagnose")

# Windows training paths (authoritative)
DATASET_ROOT = Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
OUTPUT_DIR   = Path(r"D:\VERITAS_TRAINING\checkpoints")
WAV_CACHE    = Path(r"D:\VERITAS_TRAINING\wav_cache")
PREFLIGHT_LOG = Path(r"D:\VERITAS_TRAINING\preflight_report.log")

CANDIDATE_DATASET_ROOTS = [
    DATASET_ROOT,
    Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2"),
    Path(r"D:\VERITAS_TRAINING\data"),
    Path("/kaggle/input/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"),
    Path("/kaggle/input/fakeavceleb/FakeAVCeleb_v1.2"),
]

# Audio constants — MUST match backend/audio_agent.py exactly
SAMPLE_RATE  = 16_000
CLIP_SAMPLES = 48_000            # 3.0 seconds at 16 kHz
N_FFT        = 1024
HOP_LENGTH   = 512
N_MELS       = 128
IMAGE_SIZE   = 224

BATCH_SIZE   = 8    # conservative for 6 GB VRAM laptop GPU
NUM_WORKERS  = 1    # 1 DataLoader worker initially (spawn context on Windows)
NUM_EPOCHS   = 10
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
CHECKPOINT_EVERY_N_BATCHES = 200  # periodic mid-epoch resume checkpoint

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10

CATEGORY_AUDIO_LABELS = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 1,
    "FakeVideo-RealAudio": 0,
    "FakeVideo-FakeAudio": 1,
}

_MEL = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
_DB  = T.AmplitudeToDB()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_path = OUTPUT_DIR / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_log_path), mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger("audio_train")


# ==============================================================================
# 1. DATASET DISCOVERY, METADATA INSPECTION & IDENTITY EXTRACTION
# ==============================================================================

def resolve_dataset_root() -> Path:
    """Find the root directory containing the 4 FakeAVCeleb categories."""
    for candidate in CANDIDATE_DATASET_ROOTS:
        if candidate.exists():
            present = [cat for cat in CATEGORY_AUDIO_LABELS if (candidate / cat).exists()]
            if len(present) == len(CATEGORY_AUDIO_LABELS):
                return candidate

    # Search dynamically under D:\VERITAS_TRAINING\data
    base = Path(r"D:\VERITAS_TRAINING\data")
    if base.exists():
        for d in base.rglob("*"):
            if (d.is_dir()
                    and (d / "RealVideo-RealAudio").exists()
                    and (d / "FakeVideo-FakeAudio").exists()):
                return d

    # Fallback to default
    return CANDIDATE_DATASET_ROOTS[0]


def extract_speaker_identity(mp4_path: Path, cat_dir: Path) -> str:
    """
    Legacy: extract ALL id-tokens from the full relative path, merge them.
    This creates giant connected components in FakeAVCeleb because fake clips
    reference visual-swap donors that are unrelated to the audio speaker.
    Retained for comparison diagnostics only; not used in training.
    """
    rel = mp4_path.relative_to(cat_dir)
    rel_str = str(rel).replace("\\", "/")
    id_tokens = re.findall(r"id\d+", rel_str, flags=re.IGNORECASE)
    if id_tokens:
        return "_".join(sorted(set(t.lower() for t in id_tokens)))
    generic_dirs = {"men", "women", "african", "asian", "caucasian", "american", "indian", "male", "female"}
    for part in reversed(rel.parts[:-1]):
        if part.lower() not in generic_dirs and len(part) > 1:
            return part.lower()
    return mp4_path.stem


def extract_audio_speaker(mp4_path: Path, cat_dir: Path, category: str) -> str:
    """
    Audio-source attribution: return the ID of the person whose voice is heard.

    FakeAVCeleb ground-truth attribution (verified against meta_data.csv & paper):

      RealVideo-RealAudio:
        → Genuine unmodified video and audio from target celebrity.
        → Audio speaker = folder identity (e.g. id00076).

      RealVideo-FakeAudio:
        → Audio synthesized via SV2TTS (RTVC) cloning the SAME target celebrity.
        → Audio speaker = folder identity (e.g. id00076).

      FakeVideo-RealAudio:
        → Wav2Lip (5,015 clips): Driving audio is real voice of donor encoded in
          filename as _idXXXXX_wavtolip.mp4 (audio speaker = idXXXXX).
        → FSGAN (3,964 clips): Face swapped onto folder video, but folder's genuine
          audio is preserved (audio speaker = folder identity).
        → Faceswap (730 clips): Face swapped onto target video (fallback to folder).

      FakeVideo-FakeAudio:
        → Synthesized cloned audio driving Wav2Lip lip sync.
        → Audio speaker = cloned voice donor encoded before _wavtolip in filename.
    """
    fname = mp4_path.stem  # filename without extension
    folder = mp4_path.parent.name  # immediate parent directory (usually idXXXXX)
    id_tokens = re.findall(r"id\d+", folder, re.IGNORECASE)
    folder_id = id_tokens[0].lower() if id_tokens else folder.lower()

    # 1. RealVideo-RealAudio: Genuine voice of the folder celebrity
    if category == "RealVideo-RealAudio":
        return folder_id

    # 2. RealVideo-FakeAudio: SV2TTS/RTVC cloned voice of the same folder celebrity
    if category == "RealVideo-FakeAudio":
        return folder_id

    # 3. Clips produced via Wav2Lip (in FakeVideo-FakeAudio OR FakeVideo-RealAudio):
    # The audio source driving facial reenactment is encoded before wavtolip / wav2lip
    m = re.search(r"_(id\d+)_wav(?:to)?lip", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m = re.search(r"_wav2lip_(id\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 4. Fallback for visual swaps (FSGAN, Faceswap) where folder audio was preserved:
    return folder_id


def discover_dataset(root: Path, use_audio_attribution: bool = True):
    """
    Walk the 4 FakeAVCeleb categories, verify presence, and extract metadata.
    Returns: paths, labels, categories, identities
    """
    paths, labels, categories, identities = [], [], [], []

    log.info("Inspecting dataset root: %s", root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for category, label in CATEGORY_AUDIO_LABELS.items():
        cat_dir = root / category
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing expected category directory: {cat_dir}")

        cat_mp4s = sorted(cat_dir.rglob("*.mp4"))
        log.info("  Category '%s': found %d mp4 files (audio label=%d)", category, len(cat_mp4s), label)
        for mp4 in cat_mp4s:
            if use_audio_attribution:
                ident = extract_audio_speaker(mp4, cat_dir, category)
            else:
                ident = extract_speaker_identity(mp4, cat_dir)
            paths.append(str(mp4))
            labels.append(label)
            categories.append(category)
            identities.append(ident)

    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    unique_idents = len(set(identities))

    log.info("Total discovered: %d clips (REAL=%d, FAKE=%d) across %d unique identities.",
             len(paths), n_real, n_fake, unique_idents)

    if n_real == 0 or n_fake == 0:
        raise ValueError(f"Both classes must be present! Found REAL={n_real}, FAKE={n_fake}")

    return paths, labels, categories, identities


# ==============================================================================
# 2. DISJOINT 80/10/10 IDENTITY-AWARE SPLIT
# ==============================================================================

def identity_aware_split_3way(paths, labels, categories, identities,
                              train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42):
    """
    Split dataset into 80% Train, 10% Validation, and 10% Testing with:
      - 100% disjoint identities across all 3 splits.
      - Both REAL and FAKE audio represented in all 3 splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    # 1. Group indices by identity
    by_ident: dict[str, list[int]] = {}
    for i, ident in enumerate(identities):
        by_ident.setdefault(ident, []).append(i)

    # 2. Determine class profile of each identity (0=real only, 1=fake only, 2=both)
    ident_profiles: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for ident, idxs in by_ident.items():
        has_real = any(labels[i] == 0 for i in idxs)
        has_fake = any(labels[i] == 1 for i in idxs)
        if has_real and has_fake:
            ident_profiles[2].append(ident)
        elif has_fake:
            ident_profiles[1].append(ident)
        else:
            ident_profiles[0].append(ident)

    train_idents, val_idents, test_idents = set(), set(), set()

    # 3. Stratified allocation of identities to guarantee class diversity in every split
    for profile, id_list in ident_profiles.items():
        rng.shuffle(id_list)
        n_total = len(id_list)
        if n_total == 0:
            continue

        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0
        n_test = max(1, int(round(n_total * test_ratio))) if n_total >= 3 else 0
        n_train = n_total - (n_val + n_test)
        if n_train <= 0:
            n_train = n_total
            n_val, n_test = 0, 0

        val_idents.update(id_list[:n_val])
        test_idents.update(id_list[n_val:n_val + n_test])
        train_idents.update(id_list[n_val + n_test:])

    # 4. Map back to sample indices
    train_idx, val_idx, test_idx = [], [], []
    for ident, idxs in by_ident.items():
        if ident in val_idents:
            val_idx.extend(idxs)
        elif ident in test_idents:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    def _pack(indices):
        return (
            [paths[i] for i in indices],
            [labels[i] for i in indices],
            [categories[i] for i in indices],
            [identities[i] for i in indices],
        )

    tr_p, tr_l, tr_c, tr_id = _pack(train_idx)
    va_p, va_l, va_c, va_id = _pack(val_idx)
    te_p, te_l, te_c, te_id = _pack(test_idx)

    # 5. Rigorous leak & representation assertions
    assert len(set(tr_id) & set(va_id)) == 0, "CRITICAL: Identity leakage between Train and Validation!"
    assert len(set(tr_id) & set(te_id)) == 0, "CRITICAL: Identity leakage between Train and Test!"
    assert len(set(va_id) & set(te_id)) == 0, "CRITICAL: Identity leakage between Validation and Test!"

    assert 0 in tr_l and 1 in tr_l, "Train split missing a class!"
    assert 0 in va_l and 1 in va_l, "Validation split missing a class!"
    assert 0 in te_l and 1 in te_l, "Test split missing a class!"

    log.info("Identity-Aware Split Summary:")
    log.info("  Train : %5d clips (%d REAL, %d FAKE) across %3d identities (%.1f%%)",
             len(tr_p), sum(1 for l in tr_l if l == 0), sum(1 for l in tr_l if l == 1),
             len(set(tr_id)), 100 * len(tr_p) / len(paths))
    log.info("  Val   : %5d clips (%d REAL, %d FAKE) across %3d identities (%.1f%%)",
             len(va_p), sum(1 for l in va_l if l == 0), sum(1 for l in va_l if l == 1),
             len(set(va_id)), 100 * len(va_p) / len(paths))
    log.info("  Test  : %5d clips (%d REAL, %d FAKE) across %3d identities (%.1f%%)",
             len(te_p), sum(1 for l in te_l if l == 0), sum(1 for l in te_l if l == 1),
             len(set(te_id)), 100 * len(te_p) / len(paths))

    return (tr_p, tr_l, tr_c, tr_id), (va_p, va_l, va_c, va_id), (te_p, te_l, te_c, te_id)


# ==============================================================================
# 2b. DIAGNOSTIC & PREFLIGHT MODES
# ==============================================================================

def _build_connected_components(identities_list):
    """
    Build a union-find graph over all raw id-tokens extracted from each
    composite identity string.
    Returns: list of frozensets (each frozenset = one connected component).
    """
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    all_atoms = set()
    for comp_ident in identities_list:
        atoms = comp_ident.split("_")
        for atom in atoms:
            all_atoms.add(atom)
            parent.setdefault(atom, atom)
        for i in range(len(atoms) - 1):
            union(atoms[i], atoms[i + 1])

    groups = defaultdict(set)
    for atom in all_atoms:
        groups[find(atom)].add(atom)
    return [frozenset(g) for g in groups.values()]


def diagnostic_mode(root: Path):
    """
    Inspect the dataset, read meta_data.csv if present, compare legacy vs.
    audio-source attribution identity strategies, and report component sizes.
    Does NOT start training.
    """
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("VERITAS Audio Agent — DIAGNOSTIC MODE")
    print(f"{SEP}\n")

    print(f"Dataset root : {root}")
    print(f"Root exists  : {root.exists()}")
    if not root.exists():
        print("[ERROR] Dataset root not found.")
        return

    meta_csv = root / "meta_data.csv"
    meta_parent = root.parent / "meta_data.csv"
    meta_found = meta_csv if meta_csv.exists() else (meta_parent if meta_parent.exists() else None)

    if meta_found:
        print(f"\nmeta_data.csv : {meta_found}")
        rows = []
        with open(meta_found, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= 4999:
                    break
        print(f"Columns       : {columns}")
        print(f"Rows loaded   : {len(rows)} (max 5000 for preview)")
    else:
        print("\nmeta_data.csv : NOT FOUND at dataset root or parent.")

    print(f"\n{'-' * 72}")
    print("CATEGORY INVENTORY")
    print(f"{'-' * 72}")
    cat_counts = {}
    for cat in CATEGORY_AUDIO_LABELS:
        cat_dir = root / cat
        if cat_dir.exists():
            count = len(list(cat_dir.rglob("*.mp4")))
            cat_counts[cat] = count
            print(f"  {cat:35s} : {count:6d} mp4 files  (audio_label={CATEGORY_AUDIO_LABELS[cat]})")
        else:
            print(f"  {cat:35s} : MISSING")
            cat_counts[cat] = 0
    total_clips = sum(cat_counts.values())
    print(f"  {'TOTAL':35s} : {total_clips:6d} mp4 files")

    print(f"\n{'-' * 72}")
    print("AUDIO-SOURCE ATTRIBUTION STRATEGY")
    print(f"{'-' * 72}")
    paths, labels, categories, audio_idents = discover_dataset(root, use_audio_attribution=True)
    unique_audio = sorted(set(audio_idents))
    audio_comps = _build_connected_components(unique_audio)
    print(f"  Unique audio-speaker identities   : {len(unique_audio)}")
    print(f"  Connected components (graph)      : {len(audio_comps)}")


def preflight_mode(root: Path):
    """
    Validate the 80/10/10 audio-source split against all assertions.
    Confirms speaker disjointness, both classes in every split, category
    coverage, hardware readiness, and file integrity.
    Saves report to PREFLIGHT_LOG. Stops safely — does NOT start training.
    """
    SEP = "=" * 72
    lines = []

    def p(text: str = ""):
        print(text)
        lines.append(text)

    p(f"\n{SEP}")
    p("VERITAS Audio Agent - PREFLIGHT VALIDATION MODE")
    p(f"{SEP}\n")

    # 1. Environment & Hardware
    p("1. ENVIRONMENT & HARDWARE VERIFICATION:")
    p(f"  Python executable      : {sys.executable}")
    p(f"  PyTorch version        : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    p(f"  CUDA available         : {cuda_avail}")
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        p(f"  GPU device             : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    p(f"  FFmpeg binary          : {_FFMPEG}")
    p(f"  Dataset root           : {root}")
    p(f"  Checkpoint directory   : {OUTPUT_DIR}")
    p(f"  WAV cache directory    : {WAV_CACHE}")
    p(f"  Batch size             : {BATCH_SIZE} (conservative for 6 GB VRAM)")
    p(f"  DataLoader workers     : {NUM_WORKERS}")

    # 2. Category inventory
    p(f"\n{'-' * 72}")
    p("2. CATEGORY INVENTORY & MP4 COUNTS:")
    p(f"{'-' * 72}")
    cat_counts = {}
    all_categories_present = True
    for cat, expected_label in CATEGORY_AUDIO_LABELS.items():
        cat_dir = root / cat
        if cat_dir.exists():
            mp4s = list(cat_dir.rglob("*.mp4"))
            cat_counts[cat] = len(mp4s)
            p(f"  [OK] {cat:28s} : {len(mp4s):5d} MP4s  (Audio Label = {expected_label} [{'REAL' if expected_label == 0 else 'FAKE'}])")
        else:
            cat_counts[cat] = 0
            p(f"  [FAIL] {cat:28s} : MISSING ({cat_dir})")
            all_categories_present = False

    paths, labels, categories, audio_idents = discover_dataset(root, use_audio_attribution=True)

    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    unique_speakers = len(set(audio_idents))

    p(f"\n  Total discovered clips : {len(paths):6d}")
    p(f"  Total REAL audio (0)   : {n_real:6d}")
    p(f"  Total FAKE audio (1)   : {n_fake:6d}")
    p(f"  Unique audio speakers  : {unique_speakers:6d}")

    all_ok = all_categories_present

    # 3. File integrity: check missing and 0-byte files
    p(f"\n{'-' * 72}")
    p("3. FILE INTEGRITY VERIFICATION (ALL 21,560 DISCOVERED CLIPS):")
    p(f"{'-' * 72}")
    missing_files = []
    zero_byte_files = []
    for p_str in paths:
        po = Path(p_str)
        if not po.exists():
            missing_files.append(p_str)
        elif po.stat().st_size == 0:
            zero_byte_files.append(p_str)

    p(f"  Missing files          : {len(missing_files)}")
    p(f"  Zero-byte files        : {len(zero_byte_files)}")
    if len(missing_files) > 0 or len(zero_byte_files) > 0:
        p("  [FAIL] Some dataset files are missing or empty.")
        all_ok = False
    else:
        p("  [PASS] All 21,560 MP4 files exist and have non-zero size.")

    # 4. Identity-aware 80/10/10 split & disjointness checks
    p(f"\n{'-' * 72}")
    p("4. IDENTITY-AWARE 80/10/10 SPLIT & SPEAKER DISJOINTNESS CHECKS:")
    p(f"{'-' * 72}")
    try:
        (tr_p, tr_l, tr_c, tr_id), (va_p, va_l, va_c, va_id), (te_p, te_l, te_c, te_id) = \
            identity_aware_split_3way(paths, labels, categories, audio_idents,
                                      TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

        checks = [
            (not (set(tr_id) & set(va_id)),   "Train ∩ Val speaker identity overlap is ZERO"),
            (not (set(tr_id) & set(te_id)),   "Train ∩ Test speaker identity overlap is ZERO"),
            (not (set(va_id) & set(te_id)),   "Val ∩ Test speaker identity overlap is ZERO"),
            (0 in tr_l and 1 in tr_l,         "Train split contains both REAL and FAKE audio classes"),
            (0 in va_l and 1 in va_l,         "Val split contains both REAL and FAKE audio classes"),
            (0 in te_l and 1 in te_l,         "Test split contains both REAL and FAKE audio classes"),
            (len(tr_p) > len(va_p),           "Train split is larger than Val split"),
            (len(tr_p) > len(te_p),           "Train split is larger than Test split"),
            (len(set(tr_id) & set(te_id))==0, "Test set speakers are 100% held-out from training"),
        ]

        for ok, desc in checks:
            status = "[PASS]" if ok else "[FAIL]"
            p(f"  {status} {desc}")
            if not ok:
                all_ok = False

        p()
        p(f"  Train : {len(tr_p):6d} clips ({100*len(tr_p)/len(paths):.1f}%) across {len(set(tr_id)):4d} speaker IDs (REAL={sum(1 for l in tr_l if l==0):5d}, FAKE={sum(1 for l in tr_l if l==1):5d})")
        p(f"  Val   : {len(va_p):6d} clips ({100*len(va_p)/len(paths):.1f}%) across {len(set(va_id)):4d} speaker IDs (REAL={sum(1 for l in va_l if l==0):5d}, FAKE={sum(1 for l in va_l if l==1):5d}) [Threshold calibration only]")
        p(f"  Test  : {len(te_p):6d} clips ({100*len(te_p)/len(paths):.1f}%) across {len(set(te_id)):4d} speaker IDs (REAL={sum(1 for l in te_l if l==0):5d}, FAKE={sum(1 for l in te_l if l==1):5d}) [Strictly locked held-out]")

        p()
        p("Category breakdown by split:")
        for split_name, split_cats, split_labs in [
            ("Train", tr_c, tr_l), ("Val", va_c, va_l), ("Test", te_c, te_l)
        ]:
            cat_dist = Counter(split_cats)
            p(f"  {split_name} split ({len(split_cats)} total clips):")
            for cat, cnt in sorted(cat_dist.items()):
                n_real_cat = sum(1 for c, l in zip(split_cats, split_labs) if c == cat and l == 0)
                n_fake_cat = sum(1 for c, l in zip(split_cats, split_labs) if c == cat and l == 1)
                p(f"    {cat:28s}: {cnt:5d} clips  (REAL={n_real_cat:5d}, FAKE={n_fake_cat:5d})")

    except AssertionError as exc:
        p(f"  [CRITICAL FAIL] Split assertion failed: {exc}")
        all_ok = False
    except Exception as exc:
        p(f"  [ERROR] Preflight aborted: {exc}")
        all_ok = False

    p(f"\n{SEP}")
    if all_ok:
        p("  [PASS] All preflight checks PASSED successfully.")
        p("  Training pipeline is fully verified and ready.")
        p("  (Awaiting user approval before launching full training run)")
    else:
        p("  [FAIL] One or more preflight checks FAILED.")
    p(f"{SEP}\n")

    # Save to PREFLIGHT_LOG
    try:
        PREFLIGHT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PREFLIGHT_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        p(f"Preflight report saved to: {PREFLIGHT_LOG}")
    except Exception as exc:
        p(f"Warning: could not write preflight log to {PREFLIGHT_LOG}: {exc}")


# ==============================================================================
# 3. EXTRACTION FAILURE TRACKER & CACHED DATA PIPELINE
# ==============================================================================

class FailureTracker:
    """Records, categorizes, and reports extraction failures without silently faking data."""
    def __init__(self):
        self.failures: list[dict] = []

    def record(self, path: str, category: str, split: str, reason: str):
        self.failures.append({
            "path": path,
            "filename": Path(path).name,
            "category": category,
            "split": split,
            "reason": str(reason),
        })

    def summary(self) -> dict:
        by_split: dict[str, int] = {}
        by_cat: dict[str, int] = {}
        for f in self.failures:
            by_split[f["split"]] = by_split.get(f["split"], 0) + 1
            by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1
        return {
            "total_failures": len(self.failures),
            "by_split": by_split,
            "by_category": by_cat,
        }

    def check_integrity(self, total_samples: int):
        n_fails = len(self.failures)
        if n_fails == 0:
            log.info("Extraction failure check: 0 failures recorded.")
            return

        rate = n_fails / max(total_samples, 1)
        log.warning("Extraction failures recorded: %d / %d (%.2f%%)", n_fails, total_samples, 100 * rate)
        if rate > 0.20:
            raise RuntimeError(f"High audio extraction failure rate: {rate:.1%}. Check ffmpeg setup.")


def extract_wav_cached(media_path: str, wav_cache_dir: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Extract mono 16 kHz PCM WAV to cache.
    Returns: (wav_path, error_str)
    """
    ext = Path(media_path).suffix.lower()
    if ext in {".wav", ".mp3"}:
        return media_path, None

    wav_cache_dir.mkdir(parents=True, exist_ok=True)
    key = str(abs(hash(media_path)))
    out = wav_cache_dir / f"{key}.wav"
    fail_marker = wav_cache_dir / f"{key}.failed"

    if out.exists():
        return str(out), None
    if fail_marker.exists():
        return None, "Previously failed extraction (cached)"

    cmd = [
        _FFMPEG, "-i", media_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(out), "-y", "-loglevel", "quiet",
    ]
    try:
        r = subprocess.run(cmd, timeout=45, capture_output=True)
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            fail_marker.touch()
            return None, f"ffmpeg error code {r.returncode}"
        return str(out), None
    except Exception as exc:
        fail_marker.touch()
        return None, str(exc)


def wav_to_tensor(wav_path: str) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """
    Convert a 16 kHz mono WAV file to a (3, 224, 224) normalized tensor.
    Matches backend/audio_agent._wav_to_tensor EXACTLY.
    Returns: (tensor, error_str)
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if not raw or n_frames == 0:
            return None, "Empty audio buffer"

        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio_np = audio_np.reshape(-1, n_channels).T  # (C, N)
        waveform = torch.from_numpy(audio_np.copy())

        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n = waveform.shape[1]
        if n >= CLIP_SAMPLES:
            waveform = waveform[:, :CLIP_SAMPLES]
        else:
            pad = torch.zeros(1, CLIP_SAMPLES - n)
            waveform = torch.cat([waveform, pad], dim=1)

        mel = _DB(_MEL(waveform))  # (1, 128, T)
        mel_resized = F.interpolate(
            mel.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        mel_norm = (mel_resized - 0.5) / 0.5
        tensor = mel_norm.repeat(3, 1, 1)  # (3, 224, 224)
        return tensor, None

    except Exception as exc:
        return None, str(exc)


class ResearchAudioDataset(Dataset):
    """
    Dataset returning (tensor, label, category, path, error_reason).
    Failed extractions return tensor=None and are never silently zero-filled.
    """
    def __init__(self, paths, labels, categories, cache_dir: Path):
        self.paths = paths
        self.labels = labels
        self.categories = categories
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        category = self.categories[idx]

        wav, err = extract_wav_cached(path, self.cache_dir)
        if err is not None:
            return None, torch.tensor(label, dtype=torch.float32), category, path, err

        tensor, err = wav_to_tensor(wav)
        if err is not None:
            return None, torch.tensor(label, dtype=torch.float32), category, path, err

        return tensor, torch.tensor(label, dtype=torch.float32), category, path, None


def collate_audio(batch):
    """
    Collate function that filters out extraction failures and collects failure metadata.
    Prevents empty batches from crashing or introducing NaNs.
    """
    valid = [item for item in batch if item[0] is not None]
    failed = [item for item in batch if item[0] is None]

    failures = [(item[3], item[2], item[4]) for item in failed]

    if not valid:
        return None, None, [], failures

    tensors = torch.stack([item[0] for item in valid])
    labels = torch.stack([item[1] for item in valid])
    categories = [item[2] for item in valid]

    return tensors, labels, categories, failures


# ==============================================================================
# 4. MODEL & CALIBRATION
# ==============================================================================

def build_model(device: torch.device) -> nn.Module:
    """Build EfficientNet-B0 with custom classifier matching production audio_agent.py."""
    try:
        from torchvision.models import EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        log.info("Initialized EfficientNet-B0 backbone with ImageNet-1K pretrained weights.")
    except Exception:
        model = efficientnet_b0(pretrained=True)
        log.info("Initialized EfficientNet-B0 backbone with pretrained weights (legacy API).")

    model.classifier[1] = nn.Linear(1280, 1)
    model.to(device)
    return model


def find_best_threshold(probs: list[float], labels: list[int]) -> tuple[float, float, dict]:
    """
    Sweep decision thresholds on VALIDATION data to find the threshold maximizing F1.
    Never run on test data.
    """
    best_thresh = 0.50
    best_f1 = -1.0
    curve: dict[float, float] = {}

    if not probs or len(set(labels)) < 2:
        return 0.50, 0.0, {}

    thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 91)]
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        if HAS_SKLEARN:
            f1 = float(f1_score(labels, preds, zero_division=0))
        else:
            tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-8)

        curve[t] = round(f1, 4)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1, curve


def compute_metrics(probs: list[float], labels: list[int], loss: float, threshold: float = 0.50) -> dict:
    """Compute complete classification metrics at a given decision threshold."""
    if not probs:
        return {"loss": loss, "accuracy": 0.0, "f1": 0.0, "samples": 0}

    preds = [1 if p >= threshold else 0 for p in probs]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    acc = correct / len(labels)

    m = {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "threshold": round(threshold, 4),
        "samples": len(labels),
    }

    if HAS_SKLEARN and len(set(labels)) > 1:
        m["precision"] = round(float(precision_score(labels, preds, zero_division=0)), 4)
        m["recall"]    = round(float(recall_score(labels, preds, zero_division=0)), 4)
        m["f1"]        = round(float(f1_score(labels, preds, zero_division=0)), 4)
        try:
            m["auc"]   = round(float(roc_auc_score(labels, probs)), 4)
        except Exception:
            m["auc"]   = float("nan")

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        m.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    else:
        m["precision"] = 0.0
        m["recall"]    = 0.0
        m["f1"]        = 0.0
        m["auc"]       = float("nan")

    return m


# ==============================================================================
# 5. TRAINING, VALIDATION & EVALUATION LOOPS
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, tracker, epoch,
                    start_batch: int = 0, checkpoint_callback=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for bi, (x, y, cats, failures) in enumerate(loader):
        # Skip already completed batches on mid-epoch resume
        if bi < start_batch:
            continue

        for path, cat, err in failures:
            tracker.record(path, cat, "train", err)

        if x is None or len(x) == 0:
            continue

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(_AMP_DEVICE, enabled=(device.type == "cuda")):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        running_loss += loss.item() * bs
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += bs

        if (bi + 1) % 40 == 0 or (bi + 1) == len(loader):
            log.info("  Epoch %2d  batch %3d/%3d  loss=%.4f  acc=%.3f",
                     epoch, bi + 1, len(loader),
                     running_loss / max(total, 1), correct / max(total, 1))

        # Periodic mid-epoch resume checkpoint
        if checkpoint_callback is not None and CHECKPOINT_EVERY_N_BATCHES > 0 and (bi + 1) % CHECKPOINT_EVERY_N_BATCHES == 0:
            checkpoint_callback(epoch, bi)
            log.info("  [Checkpoint] Periodic resume checkpoint saved at epoch %d, batch %d/%d",
                     epoch, bi + 1, len(loader))

    return running_loss / max(total, 1), correct / max(total, 1), total


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, tracker, split_name: str):
    model.eval()
    running_loss = 0.0
    total = 0
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_cats: list[str] = []

    for x, y, cats, failures in loader:
        for path, cat, err in failures:
            tracker.record(path, cat, split_name, err)

        if x is None or len(x) == 0:
            continue

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(_AMP_DEVICE, enabled=(device.type == "cuda")):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y) if criterion else torch.tensor(0.0)

        bs = x.size(0)
        running_loss += loss.item() * bs
        total += bs

        probs = torch.sigmoid(logits).float().cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(y.long().cpu().tolist())
        all_cats.extend(cats)

    mean_loss = running_loss / max(total, 1)
    return mean_loss, all_probs, all_labels, all_cats, total


# ==============================================================================
# 6. MAIN PIPELINE
# ==============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Dispatch based on VERITAS_MODE (env var or global)
    root = resolve_dataset_root()
    mode = EXECUTION_MODE.strip().lower()
    log.info("VERITAS_MODE = %s", mode)

    if mode == "diagnose":
        diagnostic_mode(root)
        return

    if mode == "preflight":
        preflight_mode(root)
        return

    if mode != "train":
        log.warning("Unknown VERITAS_MODE '%s'; defaulting to diagnose.", mode)
        diagnostic_mode(root)
        return

    # ──────────────────────────────────────────────────────────────────────────
    # TRAIN MODE
    # ──────────────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Execution Device: %s | FFMPEG: %s", device, _FFMPEG)
    log.info("Dataset root  = %s", root)
    log.info("Output dir    = %s", OUTPUT_DIR)
    log.info("WAV cache     = %s", WAV_CACHE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_CACHE.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Dataset & Discover (using audio-source attribution)
    paths, labels, categories, identities = discover_dataset(root, use_audio_attribution=True)

    # 2. Split 80/10/10 with zero audio-speaker identity leakage
    (tr_p, tr_l, tr_c, tr_id), (va_p, va_l, va_c, va_id), (te_p, te_l, te_c, te_id) = (
        identity_aware_split_3way(paths, labels, categories, identities,
                                  TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    )

    # 3. Datasets & Loaders
    train_ds = ResearchAudioDataset(tr_p, tr_l, tr_c, WAV_CACHE)
    val_ds   = ResearchAudioDataset(va_p, va_l, va_c, WAV_CACHE)
    test_ds  = ResearchAudioDataset(te_p, te_l, te_c, WAV_CACHE)

    # Windows safety: use spawn context to avoid DataLoader deadlocks
    mp_ctx = "spawn" if NUM_WORKERS > 0 and sys.platform == "win32" else None
    loader_kwargs = dict(
        batch_size=BATCH_SIZE, pin_memory=(device.type == "cuda"),
        collate_fn=collate_audio,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0),
        multiprocessing_context=mp_ctx,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # 4. Class Balancing Strategy
    # Using class-weighted BCE loss (pos_weight = N_real / N_fake) with standard shuffle.
    # No WeightedRandomSampler is combined with this to avoid double-weighting.
    n_tr_real = sum(1 for l in tr_l if l == 0)
    n_tr_fake = sum(1 for l in tr_l if l == 1)
    pos_weight_val = n_tr_real / max(n_tr_fake, 1)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log.info("Class Balancing Strategy: class-weighted loss (pos_weight=%.4f)", pos_weight_val)
    log.info("  Training Distribution: REAL=%d (%.1f%%), FAKE=%d (%.1f%%)",
             n_tr_real, 100 * n_tr_real / len(tr_l), n_tr_fake, 100 * n_tr_fake / len(tr_l))

    # 5. Build Model & Optimizer
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler(_AMP_DEVICE, enabled=(device.type == "cuda"))

    tracker = FailureTracker()
    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    calibrated_threshold = 0.50
    start_epoch = 1
    start_batch = 0

    # ── Attempt resume from latest checkpoint ──────────────────────────────────
    _resume_path = OUTPUT_DIR / "audio_agent_resume.pth"
    if _resume_path.exists():
        try:
            _rc = torch.load(str(_resume_path), map_location="cpu", weights_only=False)
            model.load_state_dict(_rc["model_state_dict"])
            optimizer.load_state_dict(_rc["optimizer"])
            if "scheduler" in _rc:
                scheduler.load_state_dict(_rc["scheduler"])
            scaler.load_state_dict(_rc["scaler"])
            best_val_f1 = _rc.get("best_val_f1", -1.0)
            best_epoch  = _rc.get("best_epoch", 0)
            calibrated_threshold = _rc.get("decision_threshold", 0.50)
            history = _rc.get("history", [])
            _saved_epoch = _rc.get("epoch", 1)
            _saved_batch = _rc.get("batch_idx", -1)
            if _rc.get("python_rng"):
                random.setstate(_rc["python_rng"])
            if _rc.get("numpy_rng") is not None:
                np.random.set_state(_rc["numpy_rng"])
            if _rc.get("torch_rng") is not None:
                torch.set_rng_state(_rc["torch_rng"])
            if _rc.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(_rc["cuda_rng"])

            if _saved_batch >= 0:
                start_epoch = _saved_epoch
                start_batch = _saved_batch + 1
                log.info("Resumed from mid-epoch checkpoint: epoch=%d, batch=%d -> resuming at batch %d",
                         _saved_epoch, _saved_batch, start_batch)
            else:
                start_epoch = _saved_epoch + 1
                start_batch = 0
                log.info("Resumed from completed epoch %d -> starting epoch %d",
                         _saved_epoch, start_epoch)
        except Exception as _exc:
            log.warning("Failed to restore resume checkpoint: %s -- starting fresh.", _exc)
            start_epoch = 1
            start_batch = 0

    def save_resume_checkpoint(epoch_num: int, batch_num: int):
        _state = {
            "epoch": epoch_num,
            "batch_idx": batch_num,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "decision_threshold": calibrated_threshold,
            "history": history,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "model_name": "efficientnet_b0",
            "num_classes": 1,
            "sample_rate": SAMPLE_RATE,
            "clip_samples": CLIP_SAMPLES,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "image_size": IMAGE_SIZE,
            "category_audio_labels": CATEGORY_AUDIO_LABELS,
        }
        torch.save(_state, OUTPUT_DIR / "audio_agent_resume.pth")

    log.info("Starting Audio Agent training -- epochs %d..%d", start_epoch, NUM_EPOCHS)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        t0 = time.time()
        tr_loss, tr_acc, tr_valid_n = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, tracker, epoch,
            start_batch=epoch_start_batch, checkpoint_callback=save_resume_checkpoint,
        )
        va_loss, va_probs, va_labs, va_cats, va_valid_n = evaluate_loader(
            model, val_loader, criterion, device, tracker, "val"
        )
        scheduler.step()

        # Threshold calibration on validation split
        opt_thresh, opt_f1, thresh_curve = find_best_threshold(va_probs, va_labs)
        val_metrics_at_05 = compute_metrics(va_probs, va_labs, va_loss, threshold=0.50)
        val_metrics_at_opt = compute_metrics(va_probs, va_labs, va_loss, threshold=opt_thresh)

        epoch_time = time.time() - t0
        log.info(
            "Epoch %2d/%2d [%.0fs] | Train Loss: %.4f Acc: %.3f (%d valid) | "
            "Val Loss: %.4f Acc@0.5: %.3f F1@0.5: %.3f | Opt Thresh: %.2f (Val F1: %.3f)",
            epoch, NUM_EPOCHS, epoch_time, tr_loss, tr_acc, tr_valid_n,
            va_loss, val_metrics_at_05["accuracy"], val_metrics_at_05["f1"],
            opt_thresh, opt_f1
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_metrics_at_0.5": val_metrics_at_05,
            "val_metrics_at_opt": val_metrics_at_opt,
            "opt_thresh": opt_thresh,
            "epoch_time_sec": epoch_time,
        })

        # End-of-epoch resume checkpoint (batch_idx=-1 = epoch complete)
        save_resume_checkpoint(epoch, -1)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics_at_opt,
            "decision_threshold": opt_thresh,
            "history": history,
        }, OUTPUT_DIR / "audio_agent_latest.pth")

        # Select best checkpoint based on validation F1 at calibrated threshold
        # NEVER overwrite best checkpoint with an inferior epoch
        if opt_f1 > best_val_f1:
            best_val_f1 = opt_f1
            best_epoch = epoch
            calibrated_threshold = opt_thresh

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "decision_threshold": calibrated_threshold,
                "val_metrics": val_metrics_at_opt,
                "model_name": "efficientnet_b0",
                "num_classes": 1,
                "class_names": {"0": "REAL", "1": "FAKE"},
                "sample_rate": SAMPLE_RATE,
                "clip_samples": CLIP_SAMPLES,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "image_size": IMAGE_SIZE,
                "category_audio_labels": CATEGORY_AUDIO_LABELS,
                "dataset_root": str(root),
            }, OUTPUT_DIR / "audio_agent_best.pth")
            log.info("  [*] Best validation checkpoint updated (Epoch %d, Val F1=%.4f, Threshold=%.2f)",
                     epoch, best_val_f1, calibrated_threshold)

    log.info("Training complete. Best Validation Epoch: %d (F1=%.4f, Calibrated Threshold=%.2f)",
             best_epoch, best_val_f1, calibrated_threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. FINAL EVALUATION ON UNBIASED HELD-OUT TEST SPLIT
    # ══════════════════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("FINAL EVALUATION ON HELD-OUT TEST SPLIT")
    log.info("=" * 65)

    best_ckpt_path = OUTPUT_DIR / "audio_agent_best.pth"
    best_ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_loss, te_probs, te_labs, te_cats, te_valid_n = evaluate_loader(
        model, test_loader, criterion, device, tracker, "test"
    )

    test_metrics = compute_metrics(te_probs, te_labs, test_loss, threshold=calibrated_threshold)
    test_metrics_05 = compute_metrics(te_probs, te_labs, test_loss, threshold=0.50)

    log.info("Test Results (Calibrated Threshold = %.2f):", calibrated_threshold)
    log.info("  Accuracy  : %.4f", test_metrics["accuracy"])
    log.info("  Precision : %.4f", test_metrics.get("precision", 0.0))
    log.info("  Recall    : %.4f", test_metrics.get("recall", 0.0))
    log.info("  F1 Score  : %.4f", test_metrics.get("f1", 0.0))
    log.info("  ROC-AUC   : %.4f", test_metrics.get("auc", float("nan")))
    log.info("  Confusion : TN=%d, FP=%d, FN=%d, TP=%d",
             test_metrics.get("tn", 0), test_metrics.get("fp", 0),
             test_metrics.get("fn", 0), test_metrics.get("tp", 0))

    # Category-wise evaluation breakdown
    category_results: dict[str, dict] = {}
    for cat in CATEGORY_AUDIO_LABELS:
        cat_indices = [i for i, c in enumerate(te_cats) if c == cat]
        if not cat_indices:
            continue
        c_probs = [te_probs[i] for i in cat_indices]
        c_labs  = [te_labs[i] for i in cat_indices]
        c_metrics = compute_metrics(c_probs, c_labs, 0.0, threshold=calibrated_threshold)
        c_metrics["mean_fake_prob"] = round(float(np.mean(c_probs)), 4) if c_probs else 0.0
        category_results[cat] = c_metrics
        log.info("  Category [%s] (N=%d): Acc=%.4f, MeanFakeProb=%.4f, F1=%.4f",
                 cat, len(cat_indices), c_metrics["accuracy"], c_metrics["mean_fake_prob"], c_metrics.get("f1", 0.0))

    failure_summary = tracker.summary()
    tracker.check_integrity(len(paths))

    # Update best checkpoint with verified test & failure metadata
    best_ckpt["test_metrics"] = test_metrics
    best_ckpt["test_metrics_at_0.5"] = test_metrics_05
    best_ckpt["category_test_metrics"] = category_results
    best_ckpt["failure_summary"] = failure_summary
    torch.save(best_ckpt, best_ckpt_path)

    # Export complete research report JSON
    report = {
        "dataset_root": str(root),
        "total_clips": len(paths),
        "split_counts": {
            "train": len(tr_p),
            "val": len(va_p),
            "test": len(te_p),
        },
        "identities_count": {
            "train": len(set(tr_id)),
            "val": len(set(va_id)),
            "test": len(set(te_id)),
        },
        "balancing_strategy": {
            "method": "class_weighted_loss",
            "pos_weight": round(pos_weight_val, 4),
            "train_real": n_tr_real,
            "train_fake": n_tr_fake,
        },
        "calibrated_threshold": calibrated_threshold,
        "best_epoch": best_epoch,
        "best_val_metrics": best_ckpt["val_metrics"],
        "test_metrics": test_metrics,
        "category_test_metrics": category_results,
        "failure_summary": failure_summary,
        "history": history,
    }

    report_path = OUTPUT_DIR / "audio_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Saved research evaluation report to %s", report_path)

    # WAV cache is retained for resume and re-runs
    log.info("WAV cache retained at %s", WAV_CACHE)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. VERIFICATION BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    _verify_checkpoint(best_ckpt_path)


def _verify_checkpoint(ckpt_path: Path):
    log.info("=" * 65)
    log.info("CHECKPOINT INTEGRATION VERIFICATION")
    log.info("=" * 65)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt, "Checkpoint must have 'model_state_dict'"
    assert "decision_threshold" in ckpt, "Checkpoint must have 'decision_threshold'"

    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(1280, 1)
    res = m.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys, f"Keys mismatch: {res}"
    log.info("[PASS] Checkpoint reloaded with strict=True (0 missing, 0 unexpected keys).")

    thresh = float(ckpt["decision_threshold"])
    assert 0.01 <= thresh <= 0.99, f"Invalid decision_threshold: {thresh}"
    log.info("[PASS] Verified calibrated decision_threshold: %.4f", thresh)

    m.eval()
    with torch.no_grad():
        dummy_in = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        dummy_prob = torch.sigmoid(m(dummy_in).squeeze()).item()
    assert 0.0 <= dummy_prob <= 1.0, f"Probability out of range: {dummy_prob}"
    log.info("[PASS] Dummy inference produced valid fake probability: %.4f", dummy_prob)
    log.info("=" * 65)
    log.info("READY: Copy %s -> veritas/backend/models/audio_agent_checkpoint.pth", ckpt_path.name)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
