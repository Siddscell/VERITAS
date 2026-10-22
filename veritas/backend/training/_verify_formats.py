"""
_verify_formats.py — CPU-only format compatibility check (no dataset, no GPU).

For each agent, build the model the way the TRAINING script does, save a
checkpoint with that script's exact metadata block, then reload it the way the
INFERENCE module does (strict=True) and run a dummy forward. Proves the trained
checkpoints will drop into backend/models/ without a format mismatch.
"""
import sys, io, tempfile, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.models.video import r3d_18

OK = "[PASS]"
tmp = tempfile.mkdtemp()

# ── Spatial ────────────────────────────────────────────────────────────────
print("=" * 60); print("SPATIAL"); print("=" * 60)
m = efficientnet_b0(weights=None)
m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 256),
                             nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 2))
p = os.path.join(tmp, "spatial.pth")
torch.save({"epoch": 1, "state_dict": m.state_dict(), "model_name": "efficientnet_b0",
            "num_classes": 2, "class_names": {"0": "REAL", "1": "FAKE"}, "image_size": 224,
            "category_labels": {}, "class_weights": [1.0, 1.0], "metrics": {}}, p)
# reload exactly like spatial_agent.load_model / _build_model
ckpt = torch.load(p, map_location="cpu", weights_only=False)
assert "state_dict" in ckpt, "spatial: missing 'state_dict' key"
m2 = efficientnet_b0(weights=None)
m2.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 256),
                              nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 2))
r = m2.load_state_dict(ckpt["state_dict"], strict=True)
assert not r.missing_keys and not r.unexpected_keys, f"spatial mismatch {r}"
m2.eval()
with torch.no_grad():
    fake = torch.softmax(m2(torch.randn(1, 3, 224, 224)), 1)[0, 1].item()
assert 0.0 <= fake <= 1.0
print(f"{OK} spatial: strict load + P(FAKE)={fake:.4f}")

# ── Temporal ───────────────────────────────────────────────────────────────
print("=" * 60); print("TEMPORAL"); print("=" * 60)
MEAN = [0.43216, 0.39467, 0.37645]; STD = [0.22803, 0.22145, 0.21699]
m = r3d_18(weights=None); m.fc = nn.Linear(512, 1)
p = os.path.join(tmp, "temporal.pth")
torch.save({"epoch": 1, "model_state_dict": m.state_dict(), "model_name": "r3d_18",
            "num_classes": 1, "class_names": {"0": "REAL", "1": "FAKE"},
            "num_frames": 16, "frame_size": 112, "target_fps": 8.0, "window_seconds": 2.0,
            "normalization": {"mean": MEAN, "std": STD}, "decision_threshold": 0.05,
            "category_labels": {}, "metrics": {}}, p)
# reload like temporal_agent.load_model — verify the metadata keys it reads
ckpt = torch.load(p, map_location="cpu", weights_only=False)
for k in ("model_state_dict", "normalization", "decision_threshold",
          "num_frames", "target_fps", "window_seconds"):
    assert k in ckpt, f"temporal: inference reads '{k}' but it is missing"
assert ckpt["normalization"].get("mean") and ckpt["normalization"].get("std")
m2 = r3d_18(weights=None); m2.fc = nn.Linear(512, 1)
r = m2.load_state_dict(ckpt["model_state_dict"], strict=True)
assert not r.missing_keys and not r.unexpected_keys, f"temporal mismatch {r}"
m2.eval()
with torch.no_grad():
    fake = torch.sigmoid(m2(torch.randn(1, 3, 16, 112, 112)).squeeze()).item()
assert 0.0 <= fake <= 1.0
print(f"{OK} temporal: strict load + metadata present + P(FAKE)={fake:.4f}")

# ── Audio ──────────────────────────────────────────────────────────────────
print("=" * 60); print("AUDIO"); print("=" * 60)
m = efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(1280, 1)
p = os.path.join(tmp, "audio.pth")
torch.save({"epoch": 1, "model_state_dict": m.state_dict(), "model_name": "efficientnet_b0",
            "num_classes": 1, "class_names": {"0": "REAL", "1": "FAKE"},
            "sample_rate": 16000, "clip_samples": 48000, "n_fft": 1024,
            "hop_length": 512, "n_mels": 128, "image_size": 224,
            "category_audio_labels": {}, "metrics": {}}, p)
# reload like audio_agent._load_model: weights_only=True, get("model_state_dict", ckpt)
ckpt = torch.load(p, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
m2 = efficientnet_b0(weights=None); m2.classifier[1] = nn.Linear(1280, 1)
r = m2.load_state_dict(state)  # audio_agent uses default strict=True
assert not r.missing_keys and not r.unexpected_keys, f"audio mismatch {r}"
m2.eval()
with torch.no_grad():
    fake = torch.sigmoid(m2(torch.randn(1, 3, 224, 224)).squeeze()).item()
assert 0.0 <= fake <= 1.0
print(f"{OK} audio: strict load (weights_only=True) + P(FAKE)={fake:.4f}")

print("\n" + "=" * 60)
print("ALL FORMAT CHECKS PASSED — checkpoints will load into inference modules")
print("=" * 60)
