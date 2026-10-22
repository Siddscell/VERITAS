# VERITAS PPO v3 — Binary / Four-Category Consistency Audit

**Date:** 2026-10-21 03:50:49  
**Status:** `METRICS RECONCILED — PPO NOT READY FOR PROMOTION`  
**Checkpoint:** `D:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\backend\models\ppo_fusion_policy_v3.pth`  
**Features:** `D:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\backend\training\cache\fusion_features_lc_v3.json`  
**Manifest:** `D:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\backend\training\leakage_controlled_split_manifest.json`  
**Evaluated samples:** 217  

Original report/JSON were **not** overwritten. PPO v3 was **not** promoted.

---

## 1. Root Cause

Original evaluator derived binary metrics from verdict (UNCERTAIN counted as not-fake) while the 4-way matrix used predicted_category. Two AUTHENTIC clips were UNCERTAIN + AUDIO_MANIPULATED, so category-implied FP=10/TN=40 while verdict-based FP=8/TN=42. Per-category '84%' was binary accuracy on AUTHENTIC (42/50), not categorical recall (40/50).

### Dual decision surfaces (intentional policy design)

`apply_action_to_state` emits two related but not identical decisions:

1. **Binary verdict** with a deadband: `DEEPFAKE` if `p_fake >= τ+0.05`,
   `AUTHENTIC` if `p_fake <= τ-0.05`, else `UNCERTAIN / INSUFFICIENT EVIDENCE`.
2. **Four-category `predicted_category`**: hard AUTHENTIC / AUDIO / VIDEO / BOTH.
   In the UNCERTAIN band, category uses `p_cat[0] >= 0.5` independently, so a
   sample can be **UNCERTAIN** (binary not-fake) while `predicted_category` is a
   manipulation class.

This is intentional (abstention on binary verdict while still attributing a
diagnostic category for reward / analytics). It is **not** a silent bug in the
policy. The evaluation bug was **mixing surfaces** in one report without labeling them.

Invariant `binary_fake == (predicted_category != AUTHENTIC)`:
- Satisfied on all samples: **False**
- Discrepant samples: **2** (all in UNCERTAIN zone)

| sample_id | rel_path | GT | verdict | pred_cat | p_fake | τ |
| ---: | :--- | :--- | :--- | :--- | ---: | ---: |
| 13 | `RealVideo-RealAudio\Caucasian (American)\men\id04216\00470.mp4` | AUTHENTIC | UNCERTAIN / INSUFFICIENT EVIDENCE | AUDIO_MANIPULATED | 0.5089 | 0.5405 |
| 14 | `RealVideo-RealAudio\Caucasian (American)\men\id04221\00053.mp4` | AUTHENTIC | UNCERTAIN / INSUFFICIENT EVIDENCE | AUDIO_MANIPULATED | 0.5327 | 0.5398 |

---

## 2. Evaluator Trace (same forward pass per sample)

| Quantity | Source |
| :--- | :--- |
| `predicted_category` | `apply_action_to_state` → argmax / UNCERTAIN rule |
| `verdict` | same call → three-zone threshold on `p_fake` |
| `overall_confidence` | `round(clip(p_fake))` from same call |
| Binary CM (original report) | `pred_fake = (verdict == DEEPFAKE)` |
| 4-way CM | `CATEGORY_TO_IDX[predicted_category]` |
| Per-category “acc” in original | **binary** accuracy within each GT class |
| Checkpoint | `models/ppo_fusion_policy_v3.pth` (`model_state_dict` + normalizer) |
| Split / features | `leakage_controlled_split_manifest.json` → `final_test`; `cache/fusion_features_lc_v3.json` |

Deterministic inference uses the actor mean action (no sampling).

---

## 3. Recomputed Metrics (n=217, same videos & labels)

### 3.1 Binary classification

| System | Acc | Bal Acc | Prec | Rec | Spec | F1 | TP | TN | FP | FN | CI 95% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Legacy Fixed | 76.50% | 84.73% | 100.00% | 69.46% | 100.00% | 0.8198 | 116 | 50 | 0 | 51 | [70.4%, 81.6%] |
| Corrected Fixed | 97.70% | 95.70% | 97.65% | 99.40% | 92.00% | 0.9852 | 166 | 46 | 4 | 1 | [94.7%, 99.0%] |
| PPO v3 (verdict-based) | 95.85% | 91.70% | 95.40% | 99.40% | 84.00% | 0.9736 | 166 | 42 | 8 | 1 | [92.3%, 97.8%] |
| PPO v3 (category-based) | 94.93% | 89.70% | 94.32% | 99.40% | 80.00% | 0.9679 | 166 | 40 | 10 | 1 | [91.2%, 97.1%] |

Fixed fusion **four-category metrics: unavailable** (binary-only emitters).

Category-based binary (TP/TN/FP/FN) is exactly implied by the 4-way matrix margins:
`TN=cm[0,0]=40`, `FP=sum(cm[0,1:])=10`, `FN=sum(cm[1:,0])=1`, `TP=sum(cm[1:,1:])=166` → accuracy 94.93%.

### 3.2 Four-category confusion matrix (PPO v3)

Rows = ground truth; columns = prediction. Code labels / report aliases:

| GT \ Pred | AUTHENTIC | AUDIO_FAKE | VIDEO_FAKE | BOTH_FAKE |
| :--- | :---: | :---: | :---: | :---: |
| **AUTHENTIC** (AUTHENTIC) | 40 | 10 | 0 | 0 |
| **AUDIO_MANIPULATED** (AUDIO_FAKE) | 0 | 50 | 0 | 0 |
| **VIDEO_MANIPULATED** (VIDEO_FAKE) | 1 | 0 | 69 | 2 |
| **AUDIO_VIDEO_MANIPULATED** (BOTH_FAKE) | 0 | 0 | 1 | 44 |

**4-way accuracy:** 93.55% (203/217)  
**4-way macro-F1:** 0.9318

### 3.3 Per-category precision / recall / F1 (from same 4-way matrix)

| Category | n | Precision | Recall | F1 | Diagonal |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AUTHENTIC | 50 | 97.56% | 80.00% | 0.8791 | 40 |
| AUDIO_MANIPULATED | 50 | 83.33% | 100.00% | 0.9091 | 50 |
| VIDEO_MANIPULATED | 72 | 98.57% | 95.83% | 0.9718 | 69 |
| AUDIO_VIDEO_MANIPULATED | 45 | 95.65% | 97.78% | 0.9670 | 44 |

### 3.4 Per-GT binary accuracy vs categorical recall (AUTHENTIC example)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| Original report “AUTHENTIC 84%” | 84.0% | Actually PPO **verdict** binary acc = 84.0% (42/50) |
| Categorical AUTHENTIC recall | 80.00% | 40/50 = CM diagonal |
| Category-based binary AUTHENTIC acc | 80.00% | Equals categorical recall when mapping non-AUTHENTIC→fake |

### 3.5 McNemar (Corrected Fixed vs PPO)

| Comparison | b (corr✓ ppo✗) | c (ppo✓ corr✗) | χ² | p | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Verdict-based PPO | 4 | 0 | 2.2500 | 0.1336 | False |
| Category-based PPO | 6 | 0 | 4.1667 | 0.0412 | True |

> Non-significant McNemar does **not** prove equivalence. Corrected fixed still has fewer false positives and higher accuracy on this frozen set.

---

## 4. Production-readiness conclusion

**Do not promote PPO v3 to production default.** Keep corrected fixed fusion.

- Corrected fixed binary accuracy 97.70% > PPO verdict-based 95.85% > PPO category-based 94.93%.
- False positives: corrected fixed FP=4, PPO verdict FP=8, PPO category FP=10 (on n=217 authentic-capable binary labels; authentic n=50).
- Recall is tied at 99.40% (verdict) / 99.40% (category) vs corrected 99.40% — high recall alone does not justify promotion.
- McNemar vs corrected fixed: verdict-based p=0.1336 (not significant); category-based p=0.0412 (significant). Non-significance is not equivalence.
- 4-way categorical accuracy 93.55% with AUTHENTIC categorical recall 80.00% (40/50) shows material authentic over-attribution to AUDIO_MANIPULATED.
- 2 sample(s) violate binary_fake==(predicted_category!='AUTHENTIC') because the UNCERTAIN band allows manip categories under a non-DEEPFAKE verdict.

PPO v3 remains valuable for continued evaluation of 4-way attribution, but
authentic false positives / UNCERTAIN–category mismatches must be addressed
before any default switch. No threshold retuning on the test set was performed.

---

## 5. Deliverables & assertions

| File | Role |
| :--- | :--- |
| `ppo_v3_record_level_predictions.csv` | Record-level dual-surface table |
| `ppo_v3_test_results_recomputed.json` | Machine-readable recomputed metrics |
| `ppo_v3_consistency_audit.md` | This document |
| `ppo_v3_test_results.json` | Preserved original (verdict-based) JSON |
| `ppo_v3_lc_evaluation_report.md` | Preserved original narrative report |

Assertions passed:

- `category_binary_matches_4way_margins`: **True**
- `per_category_diagonal_matches_4way`: **True**
- `all_confusion_matrices_sum_to_n`: **True**
- `verdict_based_matches_original_json`: **True**

**Final status:** `METRICS RECONCILED — PPO NOT READY FOR PROMOTION`
