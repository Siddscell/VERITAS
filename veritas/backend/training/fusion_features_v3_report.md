# VERITAS — Out-of-Sample Fusion Feature Extraction Report (v3)

**Date:** October 21, 2026  
**Auditor / Engineering Agent:** VERITAS Autonomous Verification Engine  
**Dataset:** FakeAVCeleb v1.2 (`D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`)  
**Feature Cache Path:** [`veritas/backend/training/cache/fusion_features_lc_v3.json`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/cache/fusion_features_lc_v3.json)  
**Extraction Script:** [`veritas/backend/training/extract_fusion_features_lc_v3.py`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/extract_fusion_features_lc_v3.py)  

---

## 1. Executive Summary

A core scientific flaw in prior deepfake fusion benchmarks is **feature leakage**: fusion classifiers were trained on base detector predictions that were generated either on base detector training samples (in-sample overfitting) or on identities to which the detectors were exposed during visual pretraining.

In VERITAS v3, **100% of the predictions used to train and evaluate the decision fusion policy originate from newly trained, leakage-controlled base detectors**:
* **Spatial Agent:** [`spatial_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/spatial_agent_lc.pth) (trained strictly on `detector_train`, 270 IDs)
* **Temporal Agent:** [`temporal_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/temporal_agent_lc.pth) (trained strictly on `detector_train`, 270 IDs)
* **Audio Agent:** [`audio_agent_lc.pth`](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/models/audio_agent_lc.pth) (trained strictly on `detector_train`, 270 IDs)

None of the clips or identities in `fusion_train` (100 identities, 802 clips), `fusion_val` (50 identities, 220 clips), or `final_test` (50 identities, 217 clips) were ever observed by any of the base detectors during training or checkpoint selection.

---

## 2. Partition Inventory and Sample Counts

| Partition | Celebrity Identities | Pure Clips Extracted | Unreadable / Skipped | Modality Availability | Role |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`fusion_train`** | 100 | 802 | 0 | Video (802), Audio (802) | Training PPO v3 Policy & Fitting Normalizer |
| **`fusion_val`** | 50 | 220 | 0 | Video (220), Audio (220) | Model Selection & Early Stopping for PPO v3 |
| **`final_test`** | 50 | 217 | 0 | Video (217), Audio (217) | Strictly Frozen Out-of-Sample Benchmark |
| **Total** | **200** | **1,239** | **0** | **100% Readable** | **Zero Exposure Guarantee** |

### Detailed Breakdown by Manipulation Category

| Partition | Total Clips | Authentic (`RealVideo-RealAudio`) | Audio Manipulated (`RealVideo-FakeAudio`) | Video Manipulated (`FakeVideo-RealAudio`) | Audio-Video Manipulated (`FakeVideo-FakeAudio`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`fusion_train`** | 802 | 100 | 100 | 369 | 233 |
| **`fusion_val`** | 220 | 50 | 50 | 83 | 37 |
| **`final_test`** | 217 | 50 | 50 | 72 | 45 |
| **Combined** | **1,239** | **200** | **200** | **524** | **315** |

---

## 3. 14-Dimensional Continuous State Vector Specification

Every extracted record preserves the exact 14-dimensional continuous feature representation required by `fusion_policy.py`:

```
s = [
    s[0]:  p_sp      — Spatial fake probability in [0, 1] (or 0.5 if unavailable)
    s[1]:  u_sp      — Spatial uncertainty u = 1 - 2*|p_sp - 0.5| in [0, 1]
    s[2]:  v_sp      — Spatial valid flag in {0, 1}
    s[3]:  p_te      — Temporal fake probability in [0, 1] (or 0.5 if unavailable)
    s[4]:  u_te      — Temporal uncertainty u = 1 - 2*|p_te - 0.5| in [0, 1]
    s[5]:  v_te      — Temporal valid flag in {0, 1}
    s[6]:  p_au      — Audio fake probability in [0, 1] (or 0.5 if unavailable)
    s[7]:  u_au      — Audio uncertainty u = 1 - 2*|p_au - 0.5| in [0, 1]
    s[8]:  v_au      — Audio valid flag in {0, 1}
    s[9]:  m_ctx     — Cross-modal mismatch |p_vis - p_au| in [0, 1]
    s[10]: v_ctx     — Context verification valid flag in {0, 1}
    s[11]: d_au      — Audio duration normalized min(duration / 30.0, 1.0)
    s[12]: has_audio — Container has_audio flag in {0, 1}
    s[13]: has_video — Container has_video flag in {0, 1}
]
```

### Preprocessing and Normalization Discipline
* **Fitting Isolation:** `FeatureNormalizer` is fitted strictly on the 802 samples of `fusion_train`.
* **Frozen Constants:** The mean $\mu \in \mathbb{R}^{14}$ and standard deviation $\sigma \in \mathbb{R}^{14}$ are frozen and packaged directly inside `ppo_fusion_policy_v3.pth`.
* **Zero Contamination:** `fusion_val` and `final_test` feature vectors are transformed using the frozen training constants, preventing any statistical normalization leakage.

---

## 4. Audit of Historical 266-Clip Inventory vs 217 Pure Held-Out Clips

An audit of the earlier 3-way partition (`pure_identity_disjoint_manifest.json`) reported 266 test clips, whereas the 5-way partition (`leakage_controlled_split_manifest.json`) retains 217 test clips.

### Full Provenance Reconciliation:
1. **Source Identity Retention:** Both manifests designate the exact same 50 test celebrity identities (5 men and 5 women across all 10 demographic strata).
2. **Authentic Clips:** Exactly 50 clips in both (100% retained).
3. **Audio-Only Fake Clips:** Exactly 50 clips in both (100% retained).
4. **Visual & Joint Manipulations:**
   - In the 3-way split, 166 manipulation clips were retained.
   - When moving to a 5-way split that isolates `detector_train` (270 IDs) and `fusion_train` (100 IDs), 49 manipulation clips were found to have donor identities belonging to `detector_train` or `fusion_train`.
   - To guarantee **0% donor identity leakage** into the base detectors or fusion training, those 49 clips were excluded from the pure evaluation benchmark.
   - All 49 excluded clips are documented in the audit database.

---

## 5. Diagnostic Clip (`00014_fake.mp4`) Documentation

* **File:** `RealVideo-FakeAudio/African/men/id00014/00014_fake.mp4`
* **Provenance Status:** Identity `id00014` is assigned to `fusion_train` (100 identities).
* **Usage Policy:** Strictly excluded from test reporting. Evaluated only as a qualitative diagnostic trace.
