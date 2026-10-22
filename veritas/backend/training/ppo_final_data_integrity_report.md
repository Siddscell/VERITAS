# VERITAS — Comprehensive Data Provenance, Detector Exposure, and Final PPO Validation Report

**Date:** 2026-10-20  
**Authors:** Senior RL Engineer & ML Systems Architecture Review  
**Dataset:** FakeAVCeleb v1.2 (`D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`)  
**Scope:** Reconstruct complete dataset provenance, construct role-independent connected groups, audit base-detector training exposure, evaluate defensible frozen protocols, resolve live backend synchronization, and determine formal generalization status.

---

### **Executive Determination: `PPO PROMISING — GENERALIZATION NOT YET ESTABLISHED`**

**Summary of Status:**
1. **Mathematical Architecture & Policy Verified:** PPO v2 operates on a clean, active 5-dimensional action space with continuous Gaussian distributions. On all tested subsets (the 199 unique legacy test clips and the 112 pure held-out clips), PPO v2 achieves **100% recall on deepfakes (0.0% False Negative Rate)**, resolving the legacy fixed-fusion failure mode on partial/audio deepfakes.
2. **Safe Production Default Restored:** In accordance with instructions, `main.py` has been updated so that **Semantically Corrected Fixed Fusion is the safe production default**, while PPO v2 is explicitly selectable via `VERITAS_FUSION_MODE=ppo` or runtime parameter `fusion_mode="ppo"`. The stale live uvicorn server (PID 29544) was terminated and restarted; live HTTP testing confirmed correct mode negotiation and metadata generation.
3. **Reason Full Generalization Is Not Yet Established:**
   - **Identity Graph Structure:** Analysis of all 21,566 rows in `meta_data.csv` revealed that cross-pairing between visual sources and driving donors forms a **single connected component of 578 identities**. A valid split requires strictly dropping cross-split contaminated clips rather than naive partitioning.
   - **Base-Detector Training Exposure:** The Audio detector (EfficientNet-B0) was verified to have an 80/10/10 identity-stratified split (`KNOWN_HELD_OUT` for test). However, the Spatial (EfficientNet-B0) and Temporal (R3D-18) training scripts split data on `rel.parts[0]` (top-level race directory) rather than celebrity identity, exposing them to all identities within those race categories (`TRAINED_ON_RELATED_IDENTITY / UNKNOWN`). True out-of-sample visual generalization cannot be proven without base detector retraining.
   - **Statistical Significance:** On the 199-sample legacy test set, McNemar's test yielded $\chi^2 = 2.2857$ ($p = 0.1306 > 0.05$). On the 112 pure held-out clips, PPO and Corrected Fixed tie at 99.11% accuracy. The empirical difference is directionally sound but not statistically distinguishable from the non-learned baseline at standard significance levels.

---

## 1. Full Dataset Provenance & Graph Analysis

### 1.1 Complete Dataset Audit (`meta_data.csv` vs. Disk)
We conducted an exhaustive audit of `D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`:

| Metric | Count | Details |
| :--- | :---: | :--- |
| **Total Rows in `meta_data.csv`** | 21,566 | 1 header + 21,566 data records |
| **Unique Relative File Paths** | 21,544 | Deduplicated path strings |
| **MP4 Video Files on Disk** | 21,560 | 21,544 unique CSV paths + 16 collision copies (`(1).mp4`) |
| **Unique Primary Celebrity Sources** | 500 | 500 unique celebrity identities (VoxCeleb2 derived) |
| **Total Involved Celebrity IDs** | 578 | 500 primary sources + 78 external driving donors |
| **Generation Methods** | 7 | `real` (500), `rtvc` (500), `faceswap` (730), `fsgan` (3,964), `wav2lip` (9,602), `faceswap-wav2lip` (2,717), `fsgan-wav2lip` (3,553) |

**Category Distribution Across the Complete Dataset:**
- `RealVideo-RealAudio` (Category A): **500 clips** (1 authentic clip per celebrity)
- `RealVideo-FakeAudio` (Category B): **500 clips** (1 SV2TTS/RTVC voice-cloned clip per celebrity)
- `FakeVideo-RealAudio` (Category C): **9,709 clips** (Visual face-swap / lip-sync with original voice)
- `FakeVideo-FakeAudio` (Category D): **10,857 clips** (Visual face-swap / lip-sync with cloned voice)

*Data Quality Anomaly Discovered in FakeAVCeleb:*
A SHA-256 hash collision exists between two files on disk:
- `RealVideo-RealAudio/African/women/id00592/00017.mp4`
- `FakeVideo-RealAudio/African/women/id00592/00017.mp4`
Both share hash `c821d65d578c4aa51fdaa37316c69d96b9701dbd21611da9782a6e7eead5d289`. In FakeAVCeleb, this exact file was duplicated across two categories with contradictory labels (Real in A, Fake in C). Both belong to the Training split.

---

### 1.2 Connected Components & Role-Independent Grouping

In deepfakes, individuals participate across multiple roles:
- **Visual Source (`source`):** The person whose background video / body is used.
- **Visual Target Donor (`target1`):** The person whose face or facial motion is transferred.
- **Audio Voice Donor (`target2`):** The person whose voice or speech drives the clip.

#### Graph Connectivity Finding
We constructed an undirected graph $G = (V, E)$ where vertices $V$ are all 578 celebrity identities, and an edge $(u, v) \in E$ is formed whenever identities $u$ and $v$ co-occur in the same video:

$$\text{Total Vertices: } 578 \qquad \text{Connected Components: } 1 \qquad \text{Component Size: } 578$$

> [!CRITICAL]
> **Graph Contamination Limit:**
> Because the creators of FakeAVCeleb cross-paired identities extensively across 20,566 manipulated clips, the entire dataset forms a **single giant connected component**. It is mathematically impossible to partition the graph into disjoint connected components without discarding edges.

#### Resolution: The Pure Group-Disjoint Protocol
To establish a rigorous, leak-free partition, we partition the 500 primary identities into:
- $I_{\text{train}}$: 400 identities (80%)
- $I_{\text{val}}$: 50 identities (10%)
- $I_{\text{test}}$: 50 identities (10%)

We define a video $V$ with involved identities $\text{IDs}(V) = \{\text{source}, \text{target1}, \text{target2}\} \setminus \{'-', \text{'fake.mp4'}\}$ as **Pure** if and only if all its involved identities belong strictly to a single split:
$$\text{Pure Train: } \text{IDs}(V) \subseteq I_{\text{train}} \qquad \text{Pure Val: } \text{IDs}(V) \subseteq I_{\text{val}} \qquad \text{Pure Test: } \text{IDs}(V) \subseteq I_{\text{test}}$$

Any video where an identity crosses partition boundaries is designated **Cross-Split Contaminated** and discarded from clean evaluation.

**Resulting Pure Split Yields:**

| Category | Pure Train ($I_{\text{train}}$) | Pure Val ($I_{\text{val}}$) | Pure Test ($I_{\text{test}}$) | Contaminated (Dropped) | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `RealVideo-RealAudio` | 400 | 50 | 50 | 0 | 500 |
| `RealVideo-FakeAudio` | 400 | 50 | 50 | 0 | 500 |
| `FakeVideo-RealAudio` | 5,950 | 113 | 108 | 3,538 | 9,709 |
| `FakeVideo-FakeAudio` | 4,630 | 60 | 58 | 6,109 | 10,857 |
| **Total Clips** | **11,380** | **273** | **266** | **9,647** | **21,566** |

This protocol produces **266 Pure Test Clips** across 50 held-out identities with **zero source, donor, or target identity leakage** into Train or Val.

---

## 2. Base-Detector Training Exposure Audit

We audited the training scripts, checkpoints, and reports for all three underlying agents:

| Agent | Architecture | Training Dataset | Exposure Classification | Evidence & Provenance |
| :--- | :--- | :--- | :---: | :--- |
| **Audio Analysis** | EfficientNet-B0 | FakeAVCeleb v1.2 | `KNOWN_HELD_OUT` (for Test/Val)<br>`TRAINED_ON_EXAMPLE` (for Train) | Verified via `audio_training_report.json` and `train_audio_kaggle.py`. Strict 80/10/10 stratified split on 500 IDs (400 train, 50 val, 50 test). Our Pure Test split is 100% aligned with the audio held-out test split. |
| **Spatial Analysis** | EfficientNet-B0 | FakeAVCeleb v1.2 | `TRAINED_ON_RELATED_IDENTITY`<br>`UNKNOWN` | Verified flaw in `train_spatial_kaggle.py` lines 116-125: `idents.append(rel.parts[0])` partitioned on the top-level race directory string, assigning 4 of 5 race categories to training. All identities in those 4 races were exposed during training. No sample-level manifest exists. |
| **Temporal Analysis** | R3D-18 | FakeAVCeleb v1.2 | `TRAINED_ON_RELATED_IDENTITY`<br>`UNKNOWN` | Verified flaw in `train_temporal_kaggle.py` lines 116-121: identically partitioned on `rel.parts[0]` (race string). Checkpoint metadata records `identity_aware_split: True`, but the underlying identity keys were race strings, exposing the model to the test identities. |

### Implications of Exposure Classifications:
1. **Audio Agent Predictions are Genuine Out-of-Sample:** The Audio detector was never exposed to the voices of the 50 Pure Test identities during its training. Its predictions on `test_ids` represent true generalization.
2. **Visual Detector Predictions Carry Memorization Risk:** Because Spatial and Temporal agents were trained on videos from 4 of the 5 race groups without celebrity identity isolation, test-set faces and temporal expressions were likely seen during base detector training. Consequently, high visual confidence cannot be attributed solely to fusion reasoning.

---

## 3. Independent Empirical Evaluation & Baseline Comparisons

### 3.1 Evaluation on Legacy Test Subset ($n = 199$ Unique Original Videos)

| System | Accuracy | Balanced Acc | Precision | Recall | F1-Score | FNR | FPR | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Fixed** | 70.85% | 80.54% | 1.0000 | 61.07% | 0.7583 | 38.93% | 0.00% | 91 / 50 / 0 / 58 |
| **Corrected Fixed** | 96.98% | 97.99% | 1.0000 | 95.97% | 0.9795 | 4.03% | 0.00% | 143 / 50 / 0 / 6 |
| **PPO v2 Adaptive** | **99.50%** | **99.00%** | **0.9933** | **100.00%** | **0.9967** | **0.00%** | **2.00%** | **149 / 49 / 1 / 0** |

- **Wilson 95% Confidence Interval (PPO v2):** Accuracy = 99.50% [97.22%, 99.91%], Recall = 100.00% [97.49%, 100.00%].
- **McNemar Paired Test:** PPO was correct on 6 samples where Fixed failed; Fixed was correct on 1 sample where PPO failed. Continuity-corrected $\chi^2 = \frac{(|1 - 6| - 1)^2}{7} = 2.2857 \implies p = 0.1306$. The difference is not statistically significant at $\alpha = 0.05$.

---

### 3.2 Evaluation on the Pure Held-Out Subset ($n = 112$ Extracted Clips)
We evaluated all three methods on the 112 pure held-out clips currently extracted in `interim_feature_extraction.json` (50 `RealVideo-RealAudio`, 50 `RealVideo-FakeAudio`, 5 `FakeVideo-RealAudio`, 7 `FakeVideo-FakeAudio`):

| System | Accuracy | Precision | Recall | F1-Score | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Legacy Fixed** | 54.46% | 1.0000 | 17.74% | 0.3014 | 11 / 50 / 0 / 51 |
| **Corrected Fixed** | **99.11%** | **1.0000** | 98.39% | **0.9919** | 61 / 50 / 0 / 1 |
| **PPO v2 Adaptive** | **99.11%** | 0.9841 | **100.00%** | 0.9920 | 62 / 49 / 1 / 0 |

**PPO v2 4-Category Attribution Matrix (Row = Ground Truth, Col = Prediction):**
```
                    Pred: AUTH   Pred: AUD_MANIP   Pred: VID_MANIP   Pred: AV_MANIP
GT: AUTHENTIC           49              0                 1                 0
GT: AUDIO_MANIP          0             50                 0                 0
GT: VIDEO_MANIP          0              0                 5                 0
GT: AV_MANIP             0              0                 0                 7
```
- PPO correctly attributed 50 out of 50 audio-manipulated deepfakes (100% recall).
- Legacy Fixed failed completely on audio fakes (missed 51 out of 62 deepfakes, FNR = 82.26%).
- PPO v2 and Corrected Fixed tie in overall accuracy (99.11%), with PPO achieving superior recall (100% vs 98.39%) and Fixed achieving superior specificity (100% vs 98.00%).

---

## 4. Resolution of the Stale Live Backend

### 4.1 Root Cause & Live Process Inspection
- **Process Identified:** PID 29544 was executing `C:\Python314\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000`.
- **Finding:** The process had been running continuously for >4.5 hours without `--reload`. It retained the initial in-memory module state, executing legacy fixed-weight averaging.

### 4.2 Code & Configuration Updates
1. **Safe Default Restored:** In `main.py`, the default fusion mode was changed from `"ppo"` to `"corrected_fixed"`.
2. **Explicit Parameter Selection:** Updated `_compute_verdict`, `_stream_agents`, `/analyse`, and `/analyse/stream` to accept an explicit `fusion_mode` parameter (`Form(default="")`).
3. **Graceful Fallback:** If `VERITAS_FUSION_MODE=ppo` is requested but the checkpoint is absent, the system logs a warning and falls back to `corrected_fixed`.

### 4.3 Process Restart & Live HTTP Probe Verification
We terminated PID 29544 and restarted uvicorn via `C:\Python314\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000`.
We then submitted live HTTP `POST /analyse` requests using diagnostic sample `00014_fake.mp4`:

#### Default Mode (`corrected_fixed`):
```json
{
  "HTTP Status": 200,
  "verdict": "DEEPFAKE",
  "overall_confidence": 1.0,
  "fusion_method": "corrected_fixed",
  "summary": "VERITAS has analysed the submitted content. Multi-agent consensus reached. The framework concludes with 100% confidence that the verdict is DEEPFAKE."
}
```

#### Explicit PPO Mode (`fusion_mode="ppo"`):
```json
{
  "HTTP Status": 200,
  "verdict": "DEEPFAKE",
  "overall_confidence": 0.9324,
  "fusion_method": "ppo",
  "predicted_category": "AUDIO_MANIPULATED",
  "dynamic_weights": {"spatial": 0.5028, "temporal": 0.4972, "audio": 0.5052},
  "decision_threshold": 0.4983,
  "summary": "VERITAS multi-agent fusion detected synthetic audio manipulation on authentic video. The framework concludes with 93% confidence that the verdict is DEEPFAKE."
}
```
**Conclusion:** The live API is fully operational, correctly serving Semantically Corrected Fixed Fusion by default, and actively serving PPO v2 when selected.

---

## 5. Artifact Preservation & New Manifests

The following new immutable manifest was created without modifying or overwriting existing artifacts:
- [`veritas/backend/training/pure_identity_disjoint_manifest.json`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/pure_identity_disjoint_manifest.json)
  - Records the 500 primary identities, the 578 involved identities, graph component properties, and exact split allocations.
  - Lists the 11,380 Pure Train clips, 273 Pure Val clips, and 266 Pure Test clips.
- Existing files preserved:
  - `models/spatial_agent.pth`, `models/temporal_agent_checkpoint.pth`, `models/audio_agent_checkpoint.pth`
  - `models/ppo_fusion_policy.pth`, `models/ppo_fusion_policy_v2.pth`
  - `training/ppo_split_manifest.json`, `training/cache/fusion_dataset_cache_expanded.json`
  - `training/ppo_v2_final_verification_report.md`

---

## 6. Commands Executed During Audit

```powershell
# 1. Graph and metadata analysis
python -c "import csv, os; ..."

# 2. Reconstruct audio detector identity partition
python -c "import ppo_v2_verification_audit as audit; audit._reconstruct_audio_partition(seed=42)"

# 3. Simulate pure non-contaminated yields
python -c "import csv, sys; ..."

# 4. Generate immutable pure split manifest
python -c "import json; ... json.dump(manifest_data, open('training/pure_identity_disjoint_manifest.json', 'w'))"

# 5. Evaluate pure held-out subset (n=112)
python -c "import json, numpy as np; ..."

# 6. Check and terminate stale process
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id 29544 -Force

# 7. Restart uvicorn backend
C:\Python314\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# 8. Verify live API probe
python scratch/test_live_restarted_backend.py
```

---

## 7. Next Steps for Full Generalization Validation

To transition status from `PPO PROMISING — GENERALIZATION NOT YET ESTABLISHED` to `PPO VERIFIED FOR DEMO`:

1. **Extract Remaining Pure Test Features:**
   Run feature extraction on the remaining 154 pure test clips (103 `FakeVideo-RealAudio`, 51 `FakeVideo-FakeAudio`). Estimated compute time: ~2.5 minutes on CPU.
2. **Retrain Spatial & Temporal Base Detectors with Celebrity-Level Splits:**
   Fix the bug in `train_spatial_kaggle.py` and `train_temporal_kaggle.py` so that `idents` extracts the actual celebrity folder ID (`idXXXXX`) rather than the race directory. Train on the 400 `audio_train` identities and evaluate on the 50 `audio_test` identities.
3. **Achieve Statistical Significance:**
   With all 266 Pure Test clips extracted, evaluate the paired McNemar test between PPO and Corrected Fixed fusion to establish whether PPO's adaptive threshold achieves statistically significant separation ($p < 0.05$).
