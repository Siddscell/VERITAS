# VERITAS — Base Detector Leakage-Controlled Training Report (v3)

**Date:** October 21, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140.5 MB VRAM), Intel Core i7 16-vCPU, 24 GB RAM  
**CUDA Version:** 12.4 (PyTorch 2.6.0+cu124)  
**Dataset:** FakeAVCeleb v1.2 (`D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`)  
**Split Manifest:** [`leakage_controlled_split_manifest.json`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/leakage_controlled_split_manifest.json)  

---

## 1. Executive Summary

Prior iterations of the VERITAS multi-agent deepfake detector relied on visual models (Spatial and Temporal) trained with demographic race-level directory partitioning (`rel.parts[0]`), exposing them to celebrity identities across 4 out of 5 demographic groups. Although earlier tests reported high accuracy, identity familiarity could not be ruled out.

This milestone establishes **100% Leakage-Controlled (LC) Base Detectors** trained strictly on designated `detector_train` identities (270 celebrity identities) with model selection on `detector_val` (30 identities), preserving zero identity, source, or donor overlap with `fusion_train` (100 identities), `fusion_val` (50 identities), or `final_test` (50 identities).

### Checkpoint Protection Summary
The following existing production checkpoints and logs remain untouched and preserved:
* `veritas/backend/models/spatial_agent.pth` (Legacy Spatial)
* `veritas/backend/models/temporal_agent_checkpoint.pth` (Legacy Temporal)
* `veritas/backend/models/audio_agent_checkpoint.pth` (Legacy Audio)
* `veritas/backend/models/ppo_fusion_policy.pth` (Legacy PPO v1)
* `veritas/backend/models/ppo_fusion_policy_v2.pth` (Legacy PPO v2)

All new models are saved under distinct, versioned filenames:
* [`spatial_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/spatial_agent_lc.pth)
* [`audio_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/audio_agent_lc.pth)
* [`temporal_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/temporal_agent_lc.pth)

---

## 2. Manifest & Identity Partitioning Audit

Verification was executed via [`verify_five_way_manifest.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/verify_five_way_manifest.py).

### 5-Way Balanced Partitioning Overview

| Partition | Celebrity Identities | Stratum Balance | Retained Clips | Excluded Clips | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`detector_train`** | **270** | 27 / stratum (54.0%) | **5,056** | 9,933 | Training Base Detectors |
| **`detector_val`** | **30** | 3 / stratum (6.0%) | **108** | 817 | Model Selection & Threshold Sweeping |
| **`fusion_train`** | **100** | 10 / stratum (20.0%) | **802** | 2,741 | PPO Fusion Training |
| **`fusion_val`** | **50** | 5 / stratum (10.0%) | **220** | 823 | PPO Model Selection & Checkpointing |
| **`final_test`** | **50** | 5 / stratum (10.0%) | **217** | 849 | Frozen Out-of-Sample Final Evaluation |
| **Total** | **500** | 50 / stratum (100%) | **6,403** | **15,163** | 100% Identity Disjoint Guarantee |

### Disjointness & Constraint Verification
1. **Zero Primary Identity Overlap:** Checked across all 5 partitions:
   $$\text{detector\_train} \cap \text{detector\_val} \cap \text{fusion\_train} \cap \text{fusion\_val} \cap \text{final\_test} = \emptyset$$
2. **Donor & Source Isolation:** All 6,403 clips audited against `meta_data.csv`. Exactly 0 cross-partition donor or source leaks exist.
3. **Disk Existence:** Exactly 6,403 / 6,403 clips exist on disk (0 missing files).
4. **Validation Representativeness:** `detector_val` contains 108 clips across 30 identities evenly sampled across all 10 demographic strata (African, East Asian, South Asian, Caucasian American, Caucasian European $\times$ men/women).
   - Visual Manipulation: 60 Real vs 48 Fake.
   - Audio Manipulation: 63 Real vs 45 Fake.

---

## 3. Spatial Detector Training (`spatial_agent_lc.pth`)

* **Script:** [`train_spatial_lc.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/train_spatial_lc.py)
* **Architecture:** `efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`
* **Classifier Head:** `Dropout(0.3) -> Linear(1280, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 2)`
* **Input Resolution:** $224 \times 224 \times 3$, normalized with ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
* **Target Mapping:** RealVideo-* = 0 (Real), FakeVideo-* = 1 (Fake).
* **Class Weights:** Inverse frequency weighting: Real = $4.681$, Fake = $0.560$ (540 real vs 4,516 fake clips).
* **Optimizer & Scheduler:** AdamW ($\text{lr} = 3\times 10^{-4}$, weight decay $1\times 10^{-4}$), CosineAnnealingLR ($T_{\max}=5$).
* **Hardware & Acceleration:** Mixed precision (`torch.amp.autocast("cuda")`), batch size 32 on RTX 4050 GPU. In-memory uint8 frame caching for epochs 2–5.
* **Total Training Duration:** 314.1 seconds (~5.2 minutes).

### Loss Curves and Epoch Metrics

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.1405 | **0.0185** | **98.15%** | **100.0%** | **95.83%** | **0.9787** | 0.9948 | **Saved New Best Checkpoint** |
| 2 | 0.0460 | 0.1875 | 87.96% | 79.55% | 72.92% | 0.8434 | 0.9913 | — |
| 3 | 0.0360 | 0.0154 | 98.15% | 100.0% | 95.83% | 0.9787 | **0.9962** | Preserved (tied F1) |
| 4 | 0.0233 | 0.0158 | 98.15% | 100.0% | 95.83% | 0.9787 | 0.9955 | Preserved |
| 5 | **0.0155** | 0.0151 | 98.15% | 100.0% | 95.83% | 0.9787 | 0.9958 | Preserved |

* **Final Selected Checkpoint:** Epoch 1 (Val F1: **0.9787**, Val Accuracy: **98.15%**, Val ROC-AUC: **0.9948**).
* **Saved Path:** [`veritas/backend/models/spatial_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/spatial_agent_lc.pth).

---

## 4. Audio Detector Training (`audio_agent_lc.pth`)

* **Script:** [`train_audio_lc.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/train_audio_lc.py)
* **Architecture:** `efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` with `classifier[1] = Linear(1280, 1)`.
* **Preprocessing Pipeline:** 16 kHz mono WAV -> Mel-spectrogram ($128$ mels, $N_{\text{fft}}=1024$, $\text{hop}=512$) -> AmplitudeToDB -> resize ($224 \times 224$) -> normalise $(x - 0.5) / 0.5$ -> 3 channels. Identical to `audio_agent.py` production interface.
* **Target Mapping:** *-RealAudio = 0 (Real), *-FakeAudio = 1 (Fake).
* **Class Weights:** `pos_weight = 1.4145` (Real: 2,962, Fake: 2,094).
* **Optimizer & Scheduler:** AdamW ($\text{lr} = 3\times 10^{-4}$, weight decay $1\times 10^{-4}$), CosineAnnealingLR ($T_{\max}=5$).
* **Hardware & Acceleration:** Mixed precision (`torch.amp.autocast("cuda")`), batch size 32 on RTX 4050 GPU. In-memory mel-tensor caching.
* **Total Training Duration:** 375.7 seconds (~6.2 minutes).

### Loss Curves and Epoch Metrics

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Optimal Thresh | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.3293 | 0.2400 | **95.37%** | **97.62%** | **91.11%** | **0.9425** | **0.9831** | **0.50** | **Saved New Best Checkpoint** |
| 2 | 0.2466 | **0.2202** | 95.37% | 100.0% | 88.89% | 0.9412 | 0.9727 | 0.25 | — |
| 3 | 0.2215 | 0.3031 | 95.37% | 100.0% | 88.89% | 0.9412 | 0.9681 | 0.30 | — |
| 4 | 0.1974 | 0.2307 | 95.37% | 97.62% | 91.11% | 0.9425 | 0.9753 | 0.25 | Preserved (tied F1, higher loss) |
| 5 | **0.1773** | 0.2261 | 95.37% | 100.0% | 88.89% | 0.9412 | 0.9774 | 0.20 | — |

* **Final Selected Checkpoint:** Epoch 1 (Val F1: **0.9425**, Val Accuracy: **95.37%**, Val ROC-AUC: **0.9831**, Threshold: **0.50**).
* **Saved Path:** [`veritas/backend/models/audio_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/audio_agent_lc.pth).

---

## 5. Temporal Detector Training (`temporal_agent_lc.pth`)

* **Script:** [`train_temporal_lc.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/train_temporal_lc.py)
* **Architecture:** `r3d_18(weights=R3D_18_Weights.DEFAULT)` with `fc = Linear(512, 1)`.
* **Preprocessing Pipeline:** 16 frames sampled at 8 FPS ($2.0$s window), resized to $112 \times 112 \times 3$, normalized with mean $[0.43216, 0.39467, 0.37645]$ and std $[0.22803, 0.22145, 0.21699]$. Input tensor shape: $(3, 16, 112, 112)$.
* **Target Mapping:** RealVideo-* = 0 (Real), FakeVideo-* = 1 (Fake).
* **Class Weights:** `pos_weight = 0.1196` (Real: 540, Fake: 4,516).
* **Optimizer & Scheduler:** AdamW ($\text{lr} = 2\times 10^{-4}$, weight decay $1\times 10^{-4}$), CosineAnnealingLR ($T_{\max}=4$).
* **Hardware & Acceleration:** Mixed precision (`torch.amp.autocast("cuda")`), batch size 8 on RTX 4050 GPU. Disk-cached 16-frame uint8 `.npy` arrays in `D:\VERITAS_TRAINING\temporal_cache`.

### Loss Curves and Epoch Metrics

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Optimal Thresh | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 0.0684 | 0.0638 | 95.37% | — | 93.75% | 0.9474 | 0.9878 | 0.30 | Saved New Best Checkpoint |
| 2 | 0.0262 | 0.0194 | 97.22% | — | 93.75% | 0.9677 | 0.9951 | 0.15 | Saved New Best Checkpoint |
| **3** | **0.0098** | **0.0456** | **97.22%** | **97.87%** | **95.83%** | **0.9684** | **0.9938** | **0.35** | **Saved New Best Checkpoint ← Selected** |
| 4 | 0.0062 | 0.0168 | 97.22% | — | 93.75% | 0.9677 | 0.9944 | 0.15 | Preserved (lower F1 than epoch 3) |

* **Total Training Duration:** 1,084.6 seconds (~18.1 minutes). Frame extraction was fully disk-cached after epoch 1.
* **Final Selected Checkpoint:** Epoch 3 (Val F1: **0.9684**, Val Accuracy: **97.22%**, Val Precision: **97.87%**, Val Recall: **95.83%**, Val ROC-AUC: **0.9938**, Decision Threshold: **0.35**).
* **Saved Path:** [`veritas/backend/models/temporal_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/temporal_agent_lc.pth).

---

## 6. LC Detector Summary

All three leakage-controlled base detectors are trained and saved. Zero identity, source, or donor overlap exists between any detector training data and fusion/test splits.

| Detector | Checkpoint | Val F1 | Val ROC-AUC | Decision Threshold | Best Epoch |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Spatial** | `spatial_agent_lc.pth` | **0.9787** | **0.9948** | 0.50 (softmax) | 1 |
| **Audio** | `audio_agent_lc.pth` | **0.9425** | **0.9831** | **0.50** | 1 |
| **Temporal** | `temporal_agent_lc.pth` | **0.9684** | **0.9938** | **0.35** | 3 |

**Next Step:** Run `extract_fusion_features_lc_v3.py` to generate 100% out-of-sample feature representations for `fusion_train` (802), `fusion_val` (220), and `final_test` (217) splits, then train `ppo_fusion_policy_v3.pth` via `train_ppo_v3.py`.
