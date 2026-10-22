# VERITAS PPO v2 — Comprehensive Data-Provenance, Exposure, and Generalization Audit Report

**Date:** 2026-10-20  
**Audit Scope:** Data provenance, cross-split identity/donor leakage, base-detector training exposure, frozen test evaluation reproduction, live runtime process verification, and statistical significance analysis.  
**Auditor:** Independent RL & ML Systems Architecture Review  
**Final Status Determination:** `PPO PROMISING — GENERALIZATION NOT YET ESTABLISHED`

---

## 1. Executive Summary & Audit Status

This independent audit evaluated the VERITAS PPO v2 adaptive fusion system across all layers of the machine learning pipeline: feature caches, dataset metadata, detector training provenance, frozen evaluation reproduction, and live process state.

### Final Determination: `PPO PROMISING — GENERALIZATION NOT YET ESTABLISHED`

**Status Justification:**
1. **Mathematical & Architectural Soundness:** PPO v2 completely resolves the dimensional defect of v1 (ACTION_DIM reduced from 6 to 5; all 5 continuous actions actively modulate fusion parameters). The policy achieves 99.50% binary accuracy and 0.0% False Negative Rate on the 199 unique-source test records.
2. **Failure of Full Generalization Proof (Why not `PPO VERIFIED FOR DEMO`):**
   - **Cross-Split Celebrity Identity Overlap:** Detailed audit against FakeAVCeleb's ground-truth `meta_data.csv` revealed that while file paths do not leak, **49 of the 95 source celebrity identities** (51.6%) and **22 donor identities** appearing in the test set also appear in the training pool.
   - **Base-Detector Exposure:** Historical training manifests for the Spatial (EfficientNet-B0) and Temporal (R3D-18) detectors are **NOT VERIFIED**. Source code inspection of `train_spatial_kaggle.py` and `train_temporal_kaggle.py` reveals that splitting was performed on top-level race folder names (`rel.parts[0]`) rather than individual celebrity identities, exposing the visual models to the test identities during pre-training.
   - **Statistical Non-Significance on n=199:** PPO v2 correctly classifies 5 more samples than Semantically Corrected Fixed fusion (198 vs 193). However, a continuity-corrected McNemar test yields $\chi^2 = 2.2857$ ($p = 0.1306 > 0.05$), failing to establish statistical superiority over the non-learned baseline at standard significance thresholds.
   - **Stale Active Backend Process:** A live probe of the running FastAPI server (`http://127.0.0.1:8000/analyse`) proved that the active uvicorn process has been running for >2h45m without `--reload` and is still executing the legacy fixed-weight linear average (returning `verdict: AUTHENTIC, conf: 0.2994` on `00014_fake.mp4`).

---

## 2. Dataset Audit & Count Inconsistency Resolution

### 2.1 Resolution of Reported vs. Actual Sample Counts

Previous reports documented the expanded fusion cache as having:
- *Reported:* 800 raw clips, 957 augmented records (480 train, 238 val, 239 test).

An inspection of `veritas/backend/training/cache/fusion_dataset_cache_expanded.json` resolves the exact counts:

| Split | Raw Unique Clips | Modality Augmentations | Total Records | Reported Raw (Approx.) | Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 400 | 80 | 480 | 400 | 0 |
| **Validation** | 198 | 40 | 238 | 200 | -2 |
| **Test** | 199 | 40 | 239 | 200 | -1 |
| **Total** | **797** | **160** | **957** | **800** | **-3 raw clips** |

**Root Cause of Discrepancy:**
`extract_expanded_cache.py` targeted 50 clips per category for validation and test. However:
- In Validation: `FakeVideo-FakeAudio` only yielded 48 valid extracted clips (due to 2 unparseable/corrupt video streams), giving $50 + 50 + 50 + 48 = 198$ raw clips ($198 + 40 \text{ aug} = 238$).
- In Test: `FakeVideo-FakeAudio` yielded 49 valid clips (due to 1 unparseable stream), giving $50 + 50 + 50 + 49 = 199$ raw clips ($199 + 40 \text{ aug} = 239$).
- Previous summaries rounded $198 \to 200$ and $199 \to 200$, erroneously claiming 800 raw clips. The verified ground truth is **797 raw clips** and **160 modality-stripped augmentations** ($797 + 160 = 957$).

### 2.2 Nature of the Augmentations
The 160 augmented records are strictly derivative variants of existing raw clips:
- `_aug_no_audio`: Sets audio probability to 0.50, `au_valid = 0.0`, `has_audio = 0.0` (simulating silent or non-audio media).
- `_aug_no_video`: Sets spatial and temporal probabilities to 0.50, `valid = 0.0` (simulating audio-only media).
- **Rule:** Augmented records evaluate policy resilience to missing modalities; **they do NOT represent independent test samples** and are excluded from generalization claims.

---

## 3. Unique-Source Split Integrity & Provenance Audit

### 3.1 Path-Level and Content-Hash Verification
We computed SHA-256 cryptographic hashes for all 797 unique files referenced in the cache across `D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`:

| Check | Expected | Actual | Audit Finding |
| :--- | :---: | :---: | :--- |
| Train $\cap$ Val File Paths | 0 | 0 | ✅ Zero path leakage |
| Train $\cap$ Test File Paths | 0 | 0 | ✅ Zero path leakage |
| Val $\cap$ Test File Paths | 0 | 0 | ✅ Zero path leakage |
| Cross-Split SHA-256 Collisions | 0 | 0 | ✅ Zero identical video files across splits |
| Intra-Dataset Hash Collisions | 0 | 1 pair | ⚠️ Anomaly in FakeAVCeleb: `id00592/00017.mp4` identical in RealVideo and FakeVideo |

*Note on Intra-Dataset Collision:* The file `RealVideo-RealAudio/African/women/id00592/00017.mp4` and `FakeVideo-RealAudio/African/women/id00592/00017.mp4` share the exact hash `c821d65d578c4aa51fdaa37316c69d96b9701dbd21611da9782a6e7eead5d289`. Both instances were assigned to the Training split; neither leaked into Validation or Test.

### 3.2 Deep Provenance: Visual vs. Audio Donor Identity Overlap
In FakeAVCeleb v1.2, deepfakes are generated by combining a **visual base source** (`source`) with a **driving expression/face donor** (`target1`) and/or **synthesized voice donor** (`target2`).

We cross-referenced every cache record against the complete 21,544 rows of `meta_data.csv`:

```
Split train: 400 unique files | Sources: 225 | Donors: 188 | Total Involved IDs: 327
Split val:   198 unique files | Sources:  94 | Donors:  79 | Total Involved IDs: 120
Split test:  199 unique files | Sources:  95 | Donors:  81 | Total Involved IDs: 123
```

**Cross-Split Identity Contamination Audit:**
- **Source Celebrities Overlap (Train $\cap$ Test):** **49 identities** (out of 95 in Test, 51.6%).
- **Donor Celebrities Overlap (Train $\cap$ Test):** **22 identities** (out of 81 in Test, 27.2%).
- **Total Combined Involved IDs (Train $\cap$ Test):** **74 identities** (out of 123 in Test, 60.2%).

**Why Did Identity Contamination Occur?**
`extract_expanded_cache.py` partitioned identities by extracting a single token from the file path using a regular expression:
```python
def _extract_identity_from_path(p: Path, cat: str) -> str:
    folder = p.parent.name
    id_tokens = re.findall(r"id\d+", folder, re.IGNORECASE)
    folder_id = id_tokens[0].lower() if id_tokens else folder.lower()
    if cat in ("RealVideo-RealAudio", "RealVideo-FakeAudio"):
        return folder_id
    fname = p.stem
    m = re.search(r"_(id\d+)_wav(?:to)?lip", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    ...
```
This heuristic extracted the *donor identity* (`target1`) for Wav2Lip clips, but the *source identity* (`source`) for real videos. Consequently, a celebrity assigned to the Test split as a Wav2Lip donor (e.g. `id00018`) simultaneously appeared in the Training split as a visual source for real videos.

**Impact on Generalization:**
The visual appearance (face) and/or acoustic profile of 74 out of 123 celebrities in the Test split were present in the Training split. **The evaluation split is not strictly group-disjoint at the celebrity level.**

---

## 4. Base-Detector Training Exposure Analysis

The fusion policy takes probability predictions from the Spatial, Temporal, and Audio base detectors. If a base detector was trained on a celebrity, its predictions on that celebrity in the fusion test set are *in-sample memorizations*, not genuine out-of-sample generalizations.

| Detector | Backbone | Dataset | Training Manifest | Split Strategy in Code | Exposure Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Audio Agent** | EfficientNet-B0 | FakeAVCeleb v1.2 | `audio_training_report.json` | 80/10/10 Stratified Identity Split | **VERIFIED DISJOINT** |
| **Spatial Agent** | EfficientNet-B0 | FakeAVCeleb v1.2 | **NONE (Missing)** | Split on `rel.parts[0]` (Race group) | **NOT VERIFIED / LIKELY EXPOSED** |
| **Temporal Agent** | R3D-18 | FakeAVCeleb v1.2 | **NONE (Missing)** | Split on `rel.parts[0]` (Race group) | **NOT VERIFIED / LIKELY EXPOSED** |

### 4.1 Audio Detector Exposure (Verified)
- Training report verified at `D:\VERITAS_TRAINING\checkpoints\audio_training_report.json`.
- Trained on 17,310 clips across 400 identities; validated on 2,152 clips (50 IDs); tested on 2,098 clips (50 IDs).
- Code verified in `train_audio_kaggle.py` uses `identity_aware_split_3way` with strict assertion gates.

### 4.2 Spatial & Temporal Detector Exposure (Flaw Identified)
In both `train_spatial_kaggle.py` (lines 116-125) and `train_temporal_kaggle.py` (lines 116-121):
```python
rel = mp4.relative_to(cat_dir)
idents.append(rel.parts[0] if len(rel.parts) > 1 else mp4.stem)
```
In FakeAVCeleb, paths follow `Category/<Race>/<Gender>/<idXXXXX>/<file>.mp4`.
- `rel.parts[0]` evaluated to the **race group string** (e.g., `'Asian (East)'`, `'African'`), not the celebrity identity!
- With only 5 race groups, an 85/15 split assigned 4 race groups to training and 1 race group to validation.
- **Consequence:** Celebrities of all training race groups were seen during detector training. The visual detectors were never evaluated on held-out celebrity identities.
- Therefore, the high confidence of Spatial and Temporal detectors on the fusion test set partially reflects in-sample memorization.

---

## 5. Independent Frozen Test Evaluation Reproduction

We executed an independent frozen reproduction using only the **199 unique original test videos** (zero augmentations) across all three fusion mechanisms, without altering any weights, code, or thresholds:

### 5.1 Binary Classification Benchmark (n = 199 unique original videos)

| Fusion Method | Accuracy | Balanced Acc | Precision | Recall | F1-Score | FNR | FPR | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Fixed** | 70.85% | 80.54% | 1.0000 | 61.07% | 0.7583 | 38.93% | 0.00% | 91 / 50 / 0 / 58 |
| **Corrected Fixed** | 96.98% | 97.99% | 1.0000 | 95.97% | 0.9795 | 4.03% | 0.00% | 143 / 50 / 0 / 6 |
| **PPO v2 Adaptive** | **99.50%** | **99.00%** | **0.9933** | **100.00%** | **0.9967** | **0.00%** | **2.00%** | **149 / 49 / 1 / 0** |

*Confidence Intervals (Wilson Score Interval, 95% Confidence):*
- PPO v2 Accuracy: **99.50%** — 95% CI: **[97.22%, 99.91%]** ($\pm 1.35\%$).
- PPO v2 Recall: **100.00%** — 95% CI: **[97.49%, 100.00%]**.
- PPO v2 FNR: **0.00%** (zero missed deepfakes out of 149 manipulated videos).

### 5.2 PPO v2 4-Category Attribution Matrix (n = 199)

The 4-category prediction is derived directly from the actor's continuous action distribution modulating modality trust and cross-modal mismatch (not from ground-truth labels):

| Ground Truth Category | Predicted: AUTHENTIC | Predicted: AUDIO_MANIP | Predicted: VIDEO_MANIP | Predicted: AV_MANIP | Total Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AUTHENTIC** | **49** | 0 | 1 | 0 | 50 |
| **AUDIO_MANIPULATED** | 0 | **50** | 0 | 0 | 50 |
| **VIDEO_MANIPULATED** | 0 | 0 | **50** | 0 | 50 |
| **AUDIO_VIDEO_MANIPULATED**| 0 | 0 | 0 | **49** | 49 |
| **Total Predicted** | 49 | 50 | 51 | 49 | 199 |

*Per-Category Metrics:*
- `AUTHENTIC`: Precision = 1.0000, Recall = 0.9800, F1 = 0.9899
- `AUDIO_MANIPULATED`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000
- `VIDEO_MANIPULATED`: Precision = 0.9804, Recall = 1.0000, F1 = 0.9901
- `AUDIO_VIDEO_MANIPULATED`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000
- **Macro-F1:** **0.9950**

### 5.3 Forensic Analysis of the Single PPO Error
Out of 199 unique test videos, PPO v2 made exactly one error:
- **File:** `RealVideo-RealAudio/Caucasian (American)/men/id00971/00253.mp4`
- **Ground Truth:** `RealVideo-RealAudio` (AUTHENTIC)
- **Agent Outputs:**
  - Spatial: confidence = 0.0000, `is_fake = False` (status: trained)
  - Audio: confidence = 0.0000, `is_fake = False` (status: trained)
  - **Temporal Agent:** confidence = **0.9981**, `is_fake = True` (status: trained)
- **PPO v2 Prediction:** `overall_confidence = 0.7123`, `verdict = DEEPFAKE`, `predicted_category = VIDEO_MANIPULATED`.
- **Finding:** The failure was driven by a false alarm from the base Temporal detector (R3D-18 assigned 99.8% fake probability to authentic facial movements). PPO appropriately prioritized this overwhelming temporal signal.

### 5.4 Statistical Significance: PPO v2 vs. Corrected Fixed Fusion
We conducted a paired comparison between PPO v2 and Semantically Corrected Fixed fusion on all 199 unique test samples:
- Both correct: **192 samples**
- Both incorrect: **0 samples**
- PPO correct, Corrected Fixed incorrect: **6 samples** (6 `FakeVideo-RealAudio` clips missed by fixed fusion but caught by PPO's adaptive threshold)
- Corrected Fixed correct, PPO incorrect: **1 sample** (the `id00971/00253.mp4` false positive)

**McNemar's Test (with continuity correction):**
$$b = 1, \quad c = 6$$
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|1 - 6| - 1)^2}{1 + 6} = \frac{4^2}{7} = \frac{16}{7} \approx 2.2857$$
$$p\text{-value} = P(\chi^2_1 \ge 2.2857) \approx 0.1306$$

**Conclusion on Statistical Significance:**
Because $p = 0.1306 > 0.05$, the performance difference between PPO v2 and Semantically Corrected Fixed fusion is **not statistically significant** on the 199-sample held-out set. A larger test sample ($n \ge 600$) is required to achieve statistical separation.

---

## 6. Active Backend Runtime Selection & Live Process Audit

### 6.1 Empirical Live Probe
We dispatched a live HTTP `POST /analyse` request directly to the active FastAPI process (`http://127.0.0.1:8000`) with diagnostic sample `00014_fake.mp4`:

```json
{
  "HTTP Status": 200,
  "verdict": "AUTHENTIC",
  "overall_confidence": 0.2994,
  "fusion_method": null,
  "predicted_category": null,
  "decision_threshold": null,
  "dynamic_weights": null,
  "agents": [
    {"agent": "Spatial Analysis", "confidence": 0.0008, "is_fake": false},
    {"agent": "Temporal Analysis", "confidence": 0.0001, "is_fake": false},
    {"agent": "Audio Analysis", "confidence": 1.0, "is_fake": true},
    {"agent": "Context Verification", "confidence": 0.99, "is_fake": true}
  ]
}
```

### 6.2 Stale Process Finding
- **Expected PPO Output:** `verdict: DEEPFAKE`, `overall_confidence: ~0.93`, `predicted_category: AUDIO_MANIPULATED`, `fusion_method: ppo`.
- **Actual Runtime Output:** `verdict: AUTHENTIC`, `overall_confidence: 0.2994` (the legacy formula: $0.40 \times 0.0008 + 0.30 \times 0.0001 + 0.20 \times 1.0 + 0.10 \times 0.99 = 0.2994$).
- **Root Cause:** The terminal command `python -m uvicorn main:app --host 127.0.0.1 --port 8000` was started over 2.5 hours prior without `--reload`. Consequently, the running Python process still retains the un-reloaded legacy code in memory.
- **Remediation:** The uvicorn process must be restarted before running any production demo.

---

## 7. Plan for Full-Scale Provenance-Safe Evaluation

To resolve the remaining exposure risks and establish true generalization across all 21,560 FakeAVCeleb videos:

### 7.1 Required Identity-Disjoint Protocol
1. **Disjoint Partitioning on All Celebrity Roles:**
   Partition the 500 celebrities such that if celebrity $X$ is assigned to Test:
   - No clip where $X$ is the visual source (`source`) can appear in Train or Val.
   - No clip where $X$ is the audio donor (`target1` or `target2`) can appear in Train or Val.
   - No clip where $X$ is the reenactment driver can appear in Train or Val.
2. **Base Detector Retraining:**
   - Retrain Spatial (EfficientNet-B0) and Temporal (R3D-18) detectors using this celebrity-disjoint split.
   - Export explicit training manifests (`spatial_manifest.json`, `temporal_manifest.json`) recording every training clip.
3. **Out-of-Sample Feature Extraction:**
   - Run inference across the truly held-out test split ($\approx 2,100$ videos).
   - Estimated resource cost: 2,100 videos $\times$ 0.45s = ~15.7 minutes on GPU (or ~1.8 hours on CPU).

---

## 8. Test Suite & Regression Verification

All 5 core test suites pass cleanly:

| Test Suite | Scope | Result |
| :--- | :--- | :---: |
| `test_ppo_fusion_v2.py` | 5D action space, continuous actor-critic distribution, normalizer roundtrip | **7 / 7 PASS** |
| `test_ppo_fusion.py` | State extraction, missing modality masking, reward function, regression | **51 / 51 PASS** |
| `test_integration_routing_fusion.py` | Media detection, agent applicability contracts, legacy formula verification | **44 / 44 PASS** |
| `test_audio_agent_production_integration.py` | Production audio checkpoint, strict key loading, preprocessing parity | **46 / 46 PASS** |
| `test_audio_agent_readiness.py` | Fallback handling, corrupt audio recovery, 80/10/10 split integrity | **44 / 44 PASS** |
| **Total** | | **192 / 192 PASS** |

---

## 9. Final Audit Synthesis

| Criterion | Target | Verified Status |
| :--- | :--- | :---: |
| **PPO v2 Architecture** | 5D continuous action space, all outputs active | ✅ **PASS** |
| **Evaluation Metrics (Raw Clips)** | Accuracy $\ge 99\%$, FNR $\le 1\%$ | ✅ **PASS** (99.50% Acc, 0.00% FNR) |
| **Category Attribution** | Model-derived 4-way classification | ✅ **PASS** (Macro-F1 = 0.9950) |
| **Path-Level Leakage** | Zero path overlap across splits | ✅ **PASS** (0 collisions) |
| **Identity-Level Disjointness** | Zero source or donor celebrity overlap | ❌ **FAIL** (74 overlapping IDs) |
| **Visual Detector Manifests** | Verified held-out splits for base models | ❌ **NOT VERIFIED** (Split by race, not ID) |
| **Statistical Separation** | $p < 0.05$ vs. Corrected Fixed baseline | ⚠️ **INCONCLUSIVE** ($p = 0.1306$) |
| **Active Process State** | Live uvicorn serving PPO v2 | ❌ **FAIL** (Stale process serving legacy) |

### **CONCLUSION: `PPO PROMISING — GENERALIZATION NOT YET ESTABLISHED`**
PPO v2 is an empirically high-performing and architecturally sound fusion system. However, full generalization across unseen individuals cannot be formally claimed until the base visual detectors are retrained with explicit celebrity-disjoint splits and evaluated on a larger, donor-disjoint test set.