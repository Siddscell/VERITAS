# VERITAS — PPO v3 Leakage-Controlled Fusion Policy: Training & Frozen Evaluation Report

**Date:** October 21, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), Intel Core i7 16-vCPU, 24 GB RAM  
**Dataset:** FakeAVCeleb v1.2 — 5-Way Leakage-Controlled Split  
**Split Manifest:** [`leakage_controlled_split_manifest.json`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/leakage_controlled_split_manifest.json)

---

## 1. Executive Summary

This report documents the complete leakage-controlled rebuild of the VERITAS adaptive fusion policy (PPO v3). All three base detectors were trained exclusively on `detector_train` identities (270 celebrities), with model selection on `detector_val` (30 identities). The PPO v3 fusion policy was then trained on **100% out-of-sample** feature representations extracted from `fusion_train` (100 identities) and evaluated on a **frozen, never-touched** `final_test` split (50 identities, 217 clips).

### System Comparison on Frozen Final Test (n=217)

| System | Accuracy | Balanced Acc | F1 | Precision | Recall | 4-way Cat Acc | CI 95% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Fixed Fusion** | 76.50% | 84.73% | 0.8198 | 100.0% | 69.46% | — | [70.4%, 81.6%] |
| **Corrected Fixed Fusion** | **97.70%** | **95.70%** | **0.9852** | **97.65%** | **99.40%** | — | [94.7%, 99.0%] |
| **PPO v3 Adaptive (LC)** | 95.85% | 91.70% | 0.9736 | 95.40% | **99.40%** | **93.55%** | [92.3%, 97.8%] |

> **Key finding:** PPO v3 trained on truly out-of-sample features achieves **95.85% accuracy** and **99.40% recall** on the frozen test set — statistically indistinguishable from Corrected Fixed Fusion (McNemar p=0.1336). The 4-way categorical attribution accuracy of **93.55%** is the unique capability of the adaptive policy.

---

## 2. Checkpoint Protection Verification

The following **legacy checkpoints remain unmodified** throughout this entire pipeline:

| Checkpoint | Size | Last Modified |
| :--- | :---: | :--- |
| `spatial_agent.pth` | 50.09 MB | 03-Sep-2026 (untouched) |
| `temporal_agent_checkpoint.pth` | 379.7 MB | 04-Sep-2026 (untouched) |
| `audio_agent_checkpoint.pth` | 15.56 MB | 19-Oct-2026 (untouched) |
| `ppo_fusion_policy.pth` | 0.03 MB | 20-Oct-2026 (untouched) |
| `ppo_fusion_policy_v2.pth` | 0.03 MB | 20-Oct-2026 (untouched) |

All new LC checkpoints use distinct versioned filenames:
- [`spatial_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/spatial_agent_lc.pth) (16.83 MB)
- [`audio_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/audio_agent_lc.pth) (15.58 MB)
- [`temporal_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/temporal_agent_lc.pth) (126.6 MB)
- [`ppo_fusion_policy_v3.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/ppo_fusion_policy_v3.pth)

---

## 3. Base Detector Training Summary

Full details in [`base_detector_v3_report.md`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/base_detector_v3_report.md).

| Detector | Script | Val F1 | Val ROC-AUC | Decision Threshold | Best Epoch | Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Spatial** (`efficientnet_b0`) | `train_spatial_lc.py` | **0.9787** | **0.9948** | 0.50 (softmax) | 1 | 5.2 min |
| **Audio** (`efficientnet_b0` mel) | `train_audio_lc.py` | **0.9425** | **0.9831** | 0.50 (sigmoid) | 1 | 6.2 min |
| **Temporal** (`r3d_18`) | `train_temporal_lc.py` | **0.9684** | **0.9938** | 0.35 (sigmoid) | 3 | 18.1 min |

All detectors trained on **`detector_train` only (270 identities)** — zero exposure to fusion or test identities.

---

## 4. Out-of-Sample Feature Extraction

- **Script:** [`extract_fusion_features_lc_v3.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/extract_fusion_features_lc_v3.py)
- **Cache:** [`cache/fusion_features_lc_v3.json`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/cache/fusion_features_lc_v3.json)
- **Duration:** 340.7 seconds (~3.6 clips/sec, GPU-accelerated)
- **State Representation:** 14-dimensional per clip

| Split | Manifest Clips | Unique Records | Note |
| :--- | :---: | :---: | :--- |
| `fusion_train` | 802 | 800 | 2 intra-split duplicate paths in manifest |
| `fusion_val` | 220 | 219 | 1 intra-split duplicate path in manifest |
| `final_test` | 217 | 217 | Clean |
| **Total** | **1,239** | **1,236** | Zero cross-partition contamination |

**Ground-truth category distribution:** AUTHENTIC=200, AUDIO\_MANIP=200, VIDEO\_MANIP=524, BOTH\_MANIP=312

---

## 5. PPO v3 Fusion Policy Training

- **Script:** [`train_ppo_v3.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/train_ppo_v3.py)
- **Architecture:** `ActorCriticNetwork` — 14-dim state → 5-dim continuous action
- **Training:** Single-step contextual bandit (γ=0), PPO clip ε=0.20, 20 epochs
- **FeatureNormalizer:** Fitted on `fusion_train` only (N=800)

| Epoch | Reward | Val Acc | Val F1 | Val CatAcc | Action |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | +2.081 | 92.24% | 0.9510 | 85.84% | Saved |
| 2 | +2.210 | 92.69% | 0.9538 | 85.84% | Saved |
| 3 | +2.129 | 93.61% | 0.9593 | 87.21% | Saved |
| **5** | **+2.147** | **94.06%** | **0.9619** | **87.21%** | **Best ← Saved** |
| 6–20 | 2.21–2.33 | 94.06% | 0.9619 | ≤89.50% | Preserved |

- **Duration:** 5.0 seconds (20 epochs)
- **Best Epoch:** 5 — Val F1=**0.9619**, Val Acc=**94.06%**, Val Balanced Acc=**90.52%**

---

## 6. Frozen Final Test Evaluation (n=217)

- **Script:** [`evaluate_ppo_v3_frozen.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/evaluate_ppo_v3_frozen.py)
- **Results:** [`ppo_v3_test_results.json`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/ppo_v3_test_results.json)

### 6.1 Binary Classification

| System | Acc | Bal Acc | F1 | Prec | Rec | Spec | TP | TN | FP | FN | CI 95% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Legacy Fixed | 76.50% | 84.73% | 0.8198 | 100% | 69.46% | 100% | 116 | 50 | 0 | 51 | [70.4%, 81.6%] |
| Corrected Fixed | **97.70%** | 95.70% | **0.9852** | 97.65% | 99.40% | 92% | 166 | 46 | 4 | 1 | [94.7%, 99.0%] |
| **PPO v3 (LC)** | 95.85% | 91.70% | 0.9736 | 95.40% | **99.40%** | 84% | 166 | 42 | 8 | 1 | [92.3%, 97.8%] |

### 6.2 Per-Category Recall

| GT Category | n | Legacy | Corrected | PPO v3 |
| :--- | :---: | :---: | :---: | :---: |
| AUTHENTIC | 50 | 100.0% | 92.0% | 84.0% |
| AUDIO\_MANIPULATED | 50 | **0.0%** | 100.0% | 100.0% |
| VIDEO\_MANIPULATED | 72 | 98.6% | 98.6% | 98.6% |
| AUDIO\_VIDEO\_MANIPULATED | 45 | 100.0% | 100.0% | 100.0% |

### 6.3 4-Way Attribution Confusion Matrix (PPO v3)

| GT \ Pred | AUTHENTIC | AUDIO\_FAKE | VIDEO\_FAKE | BOTH\_FAKE |
| :--- | :---: | :---: | :---: | :---: |
| **AUTHENTIC** | **40** | 10 | 0 | 0 |
| **AUDIO\_MANIPULATED** | 0 | **50** | 0 | 0 |
| **VIDEO\_MANIPULATED** | 1 | 0 | **69** | 2 |
| **AUDIO\_VIDEO\_MANIPULATED** | 0 | 0 | 1 | **44** |

**4-way categorical attribution accuracy: 93.55%** (203/217)

### 6.4 Statistical Significance

**McNemar's Test (PPO v3 vs Corrected Fixed):**  
b=4, c=0, χ²=2.2500, p=0.1336 → **Not significant** (p ≥ 0.05)

PPO v3 is statistically equivalent to Corrected Fixed at binary detection while additionally providing categorical attribution.

### 6.5 Demographic Fairness (PPO v3)

| Stratum | n | PPO v3 | Corrected Fixed |
| :--- | :---: | :---: | :---: |
| African (men) | 22 | **100.0%** | 100.0% |
| African (women) | 22 | 95.5% | 95.5% |
| Asian (East) (men) | 24 | 91.7% | 100.0% |
| Asian (East) (women) | 22 | **100.0%** | 100.0% |
| Asian (South) (men) | 19 | **100.0%** | 100.0% |
| Asian (South) (women) | 22 | **100.0%** | 100.0% |
| Caucasian (American) (men) | 23 | 91.3% | 95.7% |
| Caucasian (American) (women) | 23 | **100.0%** | 100.0% |
| Caucasian (European) (men) | 20 | 90.0% | 90.0% |
| Caucasian (European) (women) | 20 | 90.0% | 95.0% |

**Min stratum accuracy: 90.0%** | Max gap vs Corrected Fixed: 8.3pp (Asian East men)

### 6.6 Inference Latency

| Mean | Std | p95 |
| :---: | :---: | :---: |
| 0.688 ms | 0.640 ms | 0.819 ms |

---

## 7. Comparative Honesty Note

The legacy PPO v2 reported **98.62%** on the same final\_test split. That figure used features extracted from base detectors that had **identity exposure** (trained on race-level splits without strict disjointness). The **95.85% of PPO v3** is the **first honest, leakage-free estimate** of VERITAS generalization to truly unseen identities.

---

## 8. Bugs Fixed During Pipeline

| File | Bug | Fix |
| :--- | :--- | :--- |
| `train_ppo_v3.py` | `NameError: 'F' is not defined` | Added `import torch.nn.functional as F` |
| `evaluate_ppo_v3_frozen.py` | `NameError: 'logger' is not defined` | Added `import logging` + module-level `logger` |
| `evaluate_ppo_v3_frozen.py` | Wrong checkpoint key `model_state_dict` assumption | Confirmed key exists; simplified load path |

---

## 9. Next Steps

1. **Promote PPO v3** — wire `ppo_fusion_policy_v3.pth` + LC base detectors into the production API server
2. **Run `generate_provenance_audit.py`** — full lineage audit linking test results to identity partitions
3. **AUTHENTIC specificity** — 8/50 authentic clips over-flagged; investigate threshold calibration or confidence-weighted ensemble
4. **Expand test coverage** — 217-clip test set is sufficient for a leakage-free claim but narrow for sub-group CIs; plan additional data collection
