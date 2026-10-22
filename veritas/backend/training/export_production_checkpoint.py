"""
export_production_checkpoint.py
===============================
Task 2: Create and verify a sanitized, production-ready checkpoint for VERITAS Audio Agent:
1. Load trusted local checkpoint (audio_agent_best.pth) using weights_only=False solely for conversion.
2. Strip optimizer state, scheduler state, numpy scalars, and evaluation metadata.
3. Export pure model_state_dict, validated decision_threshold (0.50), and primitive audio constants.
4. Save to veritas/backend/models/audio_agent_checkpoint.pth.
5. Verify safe loading with torch.load(..., weights_only=True) in PyTorch 2.6.
6. Test strict loading into EfficientNet-B0 with Linear(1280, 1): verify 0 missing / unexpected keys.
"""

import sys
from pathlib import Path

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

SRC_CKPT = Path(r"D:\VERITAS_TRAINING\checkpoints\audio_agent_best.pth")
PROD_DST = Path(r"d:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\backend\models\audio_agent_checkpoint.pth")
BACKUP_DST = Path(r"D:\VERITAS_TRAINING\checkpoints\audio_agent_checkpoint_clean.pth")

VALIDATED_THRESHOLD = 0.50


def main():
    print("=" * 78)
    print("TASK 2: CREATE & VERIFY CLEAN PRODUCTION CHECKPOINT")
    print("=" * 78)

    assert SRC_CKPT.exists(), f"Source checkpoint not found: {SRC_CKPT}"

    # 1. Inspect destination path
    print(f"Target production path : {PROD_DST}")
    PROD_DST.parent.mkdir(parents=True, exist_ok=True)
    if PROD_DST.exists():
        backup_path = PROD_DST.with_suffix(".pth.bak")
        print(f"  Existing file found! Creating backup at: {backup_path}")
        import shutil
        shutil.copy2(PROD_DST, backup_path)
    else:
        print("  Destination is clean (no pre-existing audio checkpoint).")

    # 2. Load trusted local source checkpoint
    print(f"\nLoading source checkpoint from: {SRC_CKPT} (weights_only=False for conversion)")
    raw_ckpt = torch.load(str(SRC_CKPT), map_location="cpu", weights_only=False)
    raw_keys = sorted(list(raw_ckpt.keys()))
    print(f"  Source keys ({len(raw_keys)}): {raw_keys}")

    # Extract state dict and verify types
    raw_state_dict = raw_ckpt["model_state_dict"]
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        assert isinstance(v, torch.Tensor), f"Non-tensor in state_dict: {k} ({type(v)})"
        clean_state_dict[k] = v.clone()

    # 3. Construct sanitized dictionary with strictly primitive types
    clean_ckpt = {
        "model_state_dict": clean_state_dict,
        "decision_threshold": float(VALIDATED_THRESHOLD),
        "model_name": "efficientnet_b0",
        "sample_rate": int(16000),
        "clip_samples": int(48000),
        "n_fft": int(1024),
        "hop_length": int(512),
        "n_mels": int(128),
        "image_size": int(224),
        "epoch": int(raw_ckpt.get("epoch", 1)),
        "num_classes": int(1),
        "class_names": {"0": "REAL", "1": "FAKE"},
    }

    # Verify no numpy objects exist in the sanitized dict
    for k, v in clean_ckpt.items():
        if k != "model_state_dict":
            assert not hasattr(v, "__module__") or "numpy" not in getattr(v, "__module__", ""), f"Numpy type found in {k}: {type(v)}"

    # 4. Save to production path and backup path
    print(f"\nSaving sanitized production checkpoint to: {PROD_DST}")
    torch.save(clean_ckpt, str(PROD_DST))
    print(f"Saving copy to: {BACKUP_DST}")
    torch.save(clean_ckpt, str(BACKUP_DST))

    file_size_bytes = PROD_DST.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Exported checkpoint size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")

    # 5. Verify safe loading with weights_only=True
    print("\n" + "-" * 78)
    print("VERIFICATION: TESTING SAFE LOADING WITH weights_only=True")
    print("-" * 78)
    try:
        reloaded = torch.load(str(PROD_DST), map_location="cpu", weights_only=True)
        print("  [PASS] Successfully loaded with torch.load(..., weights_only=True)!")
        print(f"  Reloaded keys: {list(reloaded.keys())}")
        print(f"  Reloaded threshold: {reloaded['decision_threshold']} (type: {type(reloaded['decision_threshold']).__name__})")
        print(f"  Reloaded epoch    : {reloaded['epoch']} (type: {type(reloaded['epoch']).__name__})")
    except Exception as exc:
        print(f"  [FAIL] weights_only=True raised: {exc}")
        raise exc

    # 6. Test strict loading into EfficientNet-B0
    print("\n" + "-" * 78)
    print("VERIFICATION: STRICT MODEL ARCHITECTURE LOADING")
    print("-" * 78)
    prod_model = efficientnet_b0(weights=None)
    prod_model.classifier[1] = nn.Linear(1280, 1)

    load_res = prod_model.load_state_dict(reloaded["model_state_dict"], strict=True)
    print(f"  Missing keys   : {len(load_res.missing_keys)}")
    print(f"  Unexpected keys: {len(load_res.unexpected_keys)}")
    assert len(load_res.missing_keys) == 0, f"Missing keys: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys: {load_res.unexpected_keys}"
    print("  [PASS] Strict state_dict compatibility verified (0 missing, 0 unexpected keys)!")

    # 7. Dummy forward pass test
    prod_model.eval()
    with torch.no_grad():
        dummy_in = torch.randn(1, 3, 224, 224)
        logit = prod_model(dummy_in).squeeze()
        prob = torch.sigmoid(logit).item()
    print(f"  Dummy forward pass output logit: {logit.item():.4f}, sigmoid probability: {prob:.4f}")
    assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"
    print("  [PASS] Dummy forward pass produces valid probability in [0, 1]!")

    print("\n" + "=" * 78)
    print("TASK 2 COMPLETE: CLEAN PRODUCTION CHECKPOINT DEPLOYED & VERIFIED")
    print("=" * 78)


if __name__ == "__main__":
    main()
