# VERITAS — Independent Evaluation Protocol (v1)

**Document type:** Feasibility assessment and limitations only  
**Protocol version:** `independent_evaluation_protocol_v1`  
**Audit date:** 2026-10-21  
**Phases executed:** Phase 1 only  
**Phases not executed:** 2 (protocol freeze for a live cohort), 3 (paired evaluation), 4 (metrics / failure deliverables), 5 (promotion-style decision on new data)

> This document records **why** a new leakage-free independent evaluation cohort cannot be constructed from FakeAVCeleb v1.2 under the project’s existing identity and donor-isolation rules.  
> It does **not** redefine operating points, retrain models, generate new performance claims, or promote PPO.

---

## 1. Objective (Phase 1)

Determine whether FakeAVCeleb v1.2 contains enough **eligible** clips and identities for a genuinely independent evaluation of:

- Production baseline: `corrected_fixed`
- Experimental system: PPO v3 (`ppo_fusion_policy_v3.pth`)

An eligible cohort must **not** have been used for any of:

1. Base-detector training or validation  
2. Fusion training or validation  
3. Existing frozen `final_test` evaluation  
4. Threshold selection, calibration, debugging-driven tuning, or model / checkpoint selection  

Reuse the project’s established leakage-control rules: primary-identity disjointness across partitions, plus source and donor (`target1` / `target2`) isolation so that no clip spans multiple partitions.

---

## 2. Non-negotiable constraints (unchanged)

| Constraint | Status at audit |
|---|---|
| `corrected_fixed` remains production default | Confirmed in `main.py` (`VERITAS_FUSION_MODE` default) |
| PPO v3 disabled as production default | Confirmed; prior status `METRICS RECONCILED — PPO NOT READY FOR PROMOTION` |
| No modify / overwrite / retrain of existing checkpoints | Observed; this archive writes only under `independent_evaluation_v1/` |
| No change to frozen final-test predictions, metrics, manifests, or prior reports | Observed |
| Never use existing `final_test` for threshold / calibration / fusion / model selection | Binding for any future work |
| No sample-specific rules, identity shortcuts, or hardcoded dataset corrections | Binding |
| Do not claim PPO is better unless a **new** evaluation supports it | Binding |

---

## 3. Evidence artifacts inspected

Paths are relative to `veritas/backend/` unless noted. SHA-256 hashes were computed on **2026-10-21** at archive time.

### 3.1 Verified on disk (hash recorded)

| Artifact | Path | SHA-256 |
|---|---|---|
| LC 5-way split manifest | `training/leakage_controlled_split_manifest.json` | `86F84CE7948F9E0D1A11B4B90D8CCCB00E2A08456ABF88227C82C7F9ABA932D4` |
| Historical 3-way pure manifest | `training/pure_identity_disjoint_manifest.json` | `180AF500CA8A2DAA210AB4D1690AFA6D4EBB8D3FCB38D4EDEE808471DA663C7C` |
| LC fusion feature cache | `training/cache/fusion_features_lc_v3.json` | `6030C5AA754A3D53E4D13CA636EB59EF5DB26505DFADA64948F3BD88964BD6E8` |
| Frozen record-level predictions | `training/ppo_v3_record_level_predictions.csv` | `9A2865F599052296B2FD93AD4336A8D485E8F61B0628E0F1DA6BA057E45D1D7A` |
| Recomputed frozen metrics | `training/ppo_v3_test_results_recomputed.json` | `81825BA94FB70F3D3C67C689888FC30150914664CAEF646965A21A16D9CF8CE6` |
| Consistency audit | `training/ppo_v3_consistency_audit.md` | `481FE1A77B79C8B83B45EC4B696159CFD2945158359DF6FDB009085A32EDB235` |
| Decision-contract report | `training/ppo_decision_contract_report.md` | `F19B0482FE831718A278DAF98F92A3C60BD321D033EAF35817EDA3B1BE9E9901` |
| Decision-contract schema | `training/ppo_decision_contract.schema.json` | `7BF4A40665DC4E803123B5430D7420B46C033B42AB149E05CF8C45A57EB4B268` |
| Decision-contract module | `decision_contract.py` | `B0FCE22622B41E57B357EC57EADB78FD0C159624224F8A1FF7296042036E8D2A` |
| Data integrity report | `training/ppo_final_data_integrity_report.md` | `8BC3B12958B8D2F0FE8F26F15BA2DD6EDE60F3F0C113D1CE9CFD17AB2FAB4104` |
| LC validation report | `training/ppo_leakage_controlled_validation_report.md` | `6DBAB5D444F7214C0128D36869F5FE29754DA4E50A2FA354CADC54FCB489A9B9` |

### 3.2 Dataset root (verified present)

| Item | Path | Notes |
|---|---|---|
| FakeAVCeleb v1.2 root | `D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2` | Present on audit host |
| `meta_data.csv` | same root | **21,566** data rows verified by count on 2026-10-21 |

### 3.3 Checkpoint presence (hashes snapshotted; files not modified)

Prefix SHA-256 (first 16 hex chars) as of 2026-10-21 — for integrity awareness only; **not** an invitation to retrain or retarget production:

| Checkpoint | SHA-256 prefix |
|---|---|
| `models/spatial_agent_lc.pth` | `A1BABDC2581A4DCC` |
| `models/temporal_agent_lc.pth` | `3C7DD64BB9501F29` |
| `models/audio_agent_lc.pth` | `CC47D83BCFF5F119` |
| `models/ppo_fusion_policy_v3.pth` | `BB52B769DF2874D4` |
| `models/ppo_fusion_policy_v2.pth` | `2F775904D13148FB` |

### 3.4 Unverified / not re-derived in this archive

| Claim | Source | Status |
|---|---|---|
| Exact “578 involved identities” / “78 external donors” graph construction | Narrative in `ppo_final_data_integrity_report.md` / pure-manifest metadata | **Cited from prior reports**; Phase 1 re-verified that FakeAVCeleb forms one heavily cross-paired identity graph and that LC drops cross-partition clips. A full graph rebuild was not re-run for this archive. |
| Generator script that originally wrote `leakage_controlled_split_manifest.json` | Not found in-repo (only `verify_five_way_manifest.py` as consumer) | **Unverified reproducibility of split generation**; manifest contents and disjointness were verified. |
| Demographic strata assignment details beyond manifest metadata | Manifest `partition_proportions` text | Accepted as metadata; strata membership lists not independently re-audited. |

---

## 4. Leakage-controlled five-way split (verified)

**Manifest:** `leakage_controlled_split_manifest_v1`  
**Manifest `created_at`:** 2026-10-20  
**Dataset:** FakeAVCeleb v1.2  
**Seed:** 42  
**`total_videos_evaluated` (metadata):** 21,566  
**`excluded_cross_partition_videos_count` (metadata):** **15,163** (verified field value)

### 4.1 Partition allocations

| Partition | Primary identities | Listed clips (`summary`) | Unique paths (Phase 1 recount) | Prior use |
|---|---:|---:|---:|---|
| `detector_train` | 270 | 5,056 | 5,048 | Base-detector training |
| `detector_val` | 30 | 108 | 108 | Detector selection |
| `fusion_train` | 100 | 802 | 800 | PPO train + normalizer fit |
| `fusion_val` | 50 | 220 | 219 | PPO checkpoint selection |
| `final_test` | 50 | 217 | 217 | Frozen paired evaluation |
| **Total** | **500** | **6,403** | **6,392** | — |

**Primary-identity pairwise overlaps across all five partitions:** 0 (verified).  
**Strict `idXXXXX` source/target role overlaps across partitions (within listed LC clips):** 0 (verified).

### 4.2 Feature cache alignment (verified)

`fusion_features_lc_v3.json` records: `fusion_train` 800, `fusion_val` 219, `final_test` 217 (total 1,236). Detectors are LC checkpoints referenced in cache metadata.

### 4.3 Decision contract (context only; not retuned)

Canonical module: `decision_contract.py`.  
Verdict classes include `AUTHENTIC`, `DEEPFAKE`, `UNCERTAIN` (abstention — never silently scored as authentic).  
Operating points for comparison remain those already frozen for prior work (`corrected_fixed` τ=0.50 with 0.45/0.55 cuts; PPO deadband architecture as documented). **This Phase 1 archive does not select or change thresholds.**

---

## 5. Full-corpus eligibility recount (2026-10-21)

Using `meta_data.csv` (21,566 rows) and strict celebrity IDs matching `^id\d+$`:

| Pool | Count | Eligible as **new** independent cohort? |
|---|---:|---|
| Unallocated primary identities (of 500) | **0** | No — all assigned to LC partitions |
| Clips already listed in LC pure manifest | 6,403 | No |
| Leftover clips that are pure w.r.t. a **single** LC partition but not listed in the LC manifest | **1,880** | **No** — same identities as consumed partitions |
| Of which leftover under `final_test` identities | 96 | **No** — same identity pool as frozen evaluation |
| Cross-partition contaminated (roles in ≥2 LC partitions) | **13,283** | **No** — violates isolation |
| Cross + leftover + in-manifest | 15,163 + 6,403 = 21,566 | Exhausts meta rows |

**Phase 1 conclusion on eligibility:**

- **Eligible primary identities:** **0**  
- **Eligible clips:** **0**

Unused files are **not** assumed independent. Leftover same-partition clips share identities already used for training, selection, or frozen testing. Contaminated clips reintroduce donor/source leakage by construction.

---

## 6. Why candidate reuse pools fail

### 6.1 Leftover same-identity clips (1,880)

These clips are “pure” only relative to an **already allocated** partition. They are additional examples of identities already used for detector training, fusion training/validation, or frozen `final_test`. Labeling them “prospective” or “independent” would be false.

### 6.2 Historical three-way pure-test partition

`pure_identity_disjoint_manifest.json` (`pure_identity_disjoint_v1`, created 2026-10-20) uses a **different** 400/50/50 identity partition than the LC five-way split.

Verified identity overlap of historical `pure_test` (50 IDs) with LC partitions:

| LC partition | Overlap with historical `pure_test` |
|---|---:|
| `detector_train` | 27 |
| `detector_val` | 3 |
| `fusion_train` | 8 |
| `fusion_val` | 6 |
| `final_test` | 6 |
| Outside all LC IDs | **0** |

Therefore historical pure-test **cannot** serve as a new independent LC cohort. Any documentation implying LC `final_test` and pure `test` share the same 50 IDs is **incorrect** (only 6 IDs in common).

### 6.3 `fusion_val`

Used for PPO v3 checkpoint selection. Reusing it as an independent evaluation set would violate the exclusion of selection / model-choice data.

### 6.4 Frozen `final_test` (50 IDs / 217 clips)

Already used for record-level paired comparison (`ppo_v3_record_level_predictions.csv`, recomputed metrics, consistency audit). It remains a **frozen** artifact. It must **not** be reused for threshold selection, calibration, fusion changes, or a newly claimed “independent” model decision.

### 6.5 Cross-partition contaminated videos (15,163)

Dropped by the LC protocol specifically to enforce source/donor isolation. Restoring them would reintroduce leakage.

### 6.6 Single connected co-occurrence component

Prior integrity reporting states that visual sources and driving donors form **one** connected co-occurrence component over the involved identity set (cited figure: 578 identities including external donors — **not re-derived** in this archive; see §3.4). Practical implication, verified operationally by the LC drop count: FakeAVCeleb cannot be split into identity-and-donor-disjoint partitions **without discarding** cross edges. There is no hidden free primary-identity pool left after the five-way allocation.

---

## 7. Distinction: three-way pure vs five-way LC

| Aspect | Historical 3-way pure | Leakage-controlled 5-way |
|---|---|---|
| Manifest | `pure_identity_disjoint_manifest.json` | `leakage_controlled_split_manifest.json` |
| Identity split | train 400 / val 50 / test 50 | detector_train 270 / detector_val 30 / fusion_train 100 / fusion_val 50 / final_test 50 |
| Purpose | Role-pure train/val/test under older detector-exposure narrative | Staged OOS: detectors → fusion → frozen final test |
| Pure clip yields (manifest summary) | 11,380 / 273 / 266; 9,647 contaminated dropped | 6,403 pure across five parts; 15,163 contaminated dropped |
| Test ID set vs LC `final_test` | Different (6 shared) | Canonical for PPO v3 frozen eval |
| Role in this protocol | Historical reference only | Binding partition for eligibility |

Do **not** treat a re-partition or unused same-identity subset of FakeAVCeleb as prospective data collection.

---

## 8. Separated conclusions (do not conflate)

### 8.1 Existing frozen evaluation (`final_test`, n=217)

Evidence already recorded in `ppo_v3_test_results_recomputed.json` / `ppo_v3_consistency_audit.md` **supports retaining `corrected_fixed` as the production default** on that frozen set (higher accuracy / fewer false positives than PPO v3 on the reconciled surfaces; PPO not promoted).  
This archive **does not** recompute or extend those metrics.

### 8.2 New independent evaluation

**Inconclusive** because **no eligible leakage-free cohort exists** inside FakeAVCeleb v1.2 under the rules in §1–§2. Absence of a cohort is a **data limitation**, not a finding that PPO improved or regressed on fresh independent data.

### 8.3 PPO v3 production status

Remains **experimental** and **disabled** as the production default. Opt-in / training-wrapper evaluation paths may exist; they are out of scope for this archive.

### 8.4 External validation requirement

**External validation is required** before claiming generalization beyond the LC FakeAVCeleb protocol or before promoting PPO to production.

---

## 9. Only acceptable next evaluation sources

Any future independent evaluation **must** use one of:

1. **Newly collected clips** with documented provenance, labels, and identities **disjoint** from:
   - the existing **500** FakeAVCeleb primary identities, and  
   - all other identities involved in prior training, validation, selection, and frozen testing (including donor/source roles appearing in those pipelines); **or**

2. **A second public corpus** evaluated under an appropriately documented **source and donor isolation** protocol, with a new versioned manifest and leakage audit.

Unacceptable (explicitly rejected):

- Re-splitting FakeAVCeleb and calling it prospective  
- Using leftover same-identity clips as an “independent” set  
- Using `fusion_val`, contaminated drops, or burned `final_test` for selection or a new promotion claim  
- Silent `UNCERTAIN` → `AUTHENTIC` scoring in any later report  

---

## 10. Mandatory freeze before any later evaluation (Phases 2+)

Before examining new evaluation labels or performance results, freeze and version:

1. Eligible population definition and exclusion rules  
2. Exact identity-disjoint evaluation manifest (+ leakage audit)  
3. Checkpoint file hashes and model versions (detectors + fusion)  
4. Fusion modes under test and decision thresholds / deadbands  
5. Decision-contract version  
6. Primary comparison metric and secondary metrics  
7. Confidence-interval method and paired statistical comparison procedure  

Where calibration or threshold selection is necessary, use **validation data only**, freeze operating points, then access the new evaluation set once.

Reserve the word **prospective** for subsequently collected or otherwise genuinely future data — not for newly partitioned historical FakeAVCeleb material.

---

## 11. Phase 1 decision under the project rubric

Exactly one evidence-based outcome applies to **new independent evaluation** at this time:

> **Evaluation is inconclusive because of sample size, leakage risk, or data limitations.**

Specifically: **data limitations / zero eligible identities and clips** under identity and donor-isolation rules.

This is **not**:

- “PPO is supported for further controlled testing” on the basis of a new independent cohort (none exists here), nor  
- “PPO shows no demonstrated improvement over `corrected_fixed`” on **new** independent data (that claim would require an executed Phase 3 on an eligible cohort).

Separate, already-frozen evidence continues to support keeping **`corrected_fixed`** as production default and keeping PPO v3 out of production — without treating that frozen set as a fresh independent cohort.

---

## 12. Archive integrity statement

On **2026-10-21**, this directory was created solely to preserve Phase 1 findings.

**Created:**

- `veritas/backend/training/independent_evaluation_v1/README.md`
- `veritas/backend/training/independent_evaluation_v1/independent_evaluation_protocol.md`

**Not created / not executed:** evaluation scripts, new prediction files, new metrics JSON, failure-analysis reports, or operating-point freezes for a live cohort.

**Not modified:** model code, training scripts, checkpoints (`.pth`), LC / pure manifests, feature caches, frozen prediction CSV/JSON, prior evaluation reports, production default, or PPO enablement.
