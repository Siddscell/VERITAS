# VERITAS — PPO Decision Contract & Evaluation Architecture

**Date:** 2026-10-21  
**Status:** `DECISION CONTRACT VERIFIED`  
**Production default:** `corrected_fixed` (unchanged; PPO v3 remains disabled)  
**Prior audit:** `ppo_v3_consistency_audit.md` (`METRICS RECONCILED — PPO NOT READY FOR PROMOTION`)

---

## 1. Purpose

Unify binary-verdict and four-category decision semantics, document a single API/UI contract, and correct evaluation so `UNCERTAIN` is never silently counted as `AUTHENTIC`. No thresholds, weights, or PPO checkpoints were retuned on the frozen test set. No filename-specific overrides were added.

---

## 2. Output contract

Canonical module: `veritas/backend/decision_contract.py`  
Machine-readable schema: `veritas/backend/training/ppo_decision_contract.schema.json`

| Field | Semantics |
| :--- | :--- |
| `category_probabilities` | Normalized 4-way mass over AUTHENTIC / AUDIO_MANIPULATED / VIDEO_MANIPULATED / AUDIO_VIDEO_MANIPULATED. **Model-implied** (softmax / modality joint), not externally calibrated (`category_probabilities_calibrated=false`). |
| `predicted_category` | Leading category under policy rules. Under abstention this is a **hypothesis** (`leading_category_is_hypothesis=true`), not a definitive authenticity claim. |
| `fake_probability` | `P(fake) = P(AUDIO)+P(VIDEO)+P(BOTH)` when the 4-way distribution is valid (= `1 − P(AUTHENTIC)`). |
| `overall_confidence` | Backward-compatible alias of the fused fake score (same numeric role as `fake_probability`). |
| `verdict` / `verdict_class` | Operating decision: `DEEPFAKE` \| `AUTHENTIC` \| `UNCERTAIN` \| `UNVERIFIED`. `UNCERTAIN`/`UNVERIFIED` are abstentions. |
| `decision_confidence` | Strength of the **binary** call; `0` inside the deadband. Distinct from fake probability. |
| `decision_thresholds` | `{tau, tau_low, tau_high, deadband}` — frozen operating points used for the binary verdict. |
| `mismatch_score` | Cross-modal inconsistency; **not** folded into fake probability as evidence of fakery by itself. |

### Consistency invariant

When a valid normalized 4-way distribution exists:

```
fake_probability == sum(non-AUTHENTIC category_probabilities)  (± numerical tolerance)
```

Outside the abstention band:

```
(verdict_class == DEEPFAKE) ⇔ (predicted_category ≠ AUTHENTIC)
```

Inside the band, abstention may coexist with a manipulative **hypothesis** category. That must not be rendered as a definitive authentic verdict.

---

## 3. Uncertainty operating point

| System | τ (center) | Band | Source |
| :--- | :--- | :--- | :--- |
| PPO (`apply_action_to_state`) | `clip(0.50 + 0.15·tanh(a[4]), 0.35, 0.65)` | ±`0.05` | Architectural constant in fusion math; `a[4]` learned on **train**; checkpoint selected on **val**. Deadband **not** re-selected on `final_test`. |
| `corrected_fixed` | `0.50` | cuts at `0.45` / `0.55` | Fixed product-rule baseline |

**Documented trade-off (frozen, not test-tuned):**  
Prefer abstention on borderline scores over forced binary calls. Binary false negatives are heavily penalized in the PPO reward (`−3.0`); false positives (`−2.0`); abstention (`−0.3`). This encodes a recall-oriented preference with an explicit opt-out band — not a claim of post-hoc calibrated probability.

**Calibration status:** Four-category outputs are **not** reported as externally calibrated. Prospective work may fit a calibration map on validation only, freeze it, then evaluate on test.

---

## 4. PPO output architecture audit (`apply_action_to_state`)

| Claim | Result |
| :--- | :--- |
| `a[0], a[1]` affect visual weights / `p_visual` | Yes (softmax gate when both visual agents valid) |
| `a[2]` affects audio trust / `p_audio` | Yes (`sigmoid` trust toward 0.5 prior) |
| `a[3]` affects category logits via mismatch gain | Yes (`g_ctx` on joint softmax) |
| `a[4]` affects `decision_threshold` | Yes (`tanh` offset into τ) |
| Four category outputs coherent | Yes — joint independent base + mismatch-modulated softmax, or modality-only marginals |
| Normalization valid | Softmax / explicit 2-mass marginals sum to 1 |
| Missing agents fabricate evidence | No — invalid modalities excluded; audio/video-only paths zero impossible classes |
| Context mismatch ≠ fake probability | Yes — `mismatch_score` separate; modulates category logits only |
| Binary fake ↔ category sum | Yes when 4-way distribution present (`P_fake = 1 − p_cat[0]`) |

**Smallest architectural correction applied (no retrain):** expose contract fields (`fake_probability`, `verdict_class`, `decision_confidence`, `decision_thresholds`, hypothesis flags) and align `corrected_fixed` to emit the same 4-way joint decomposition for API/UI parity. Action→computation wiring was already sound; no disconnected action was left as display-only metadata.

**Retrain?** Not required for contract compliance. A new training run is optional for future calibration / operating-point selection on validation — preserve `ppo_fusion_policy_v3.pth` and train a new version if pursued.

Production PPO loader remains **v2** when `VERITAS_FUSION_MODE=ppo`; evaluated v3 stays opt-in via training wrappers. Default mode remains **`corrected_fixed`**.

---

## 5. Evaluation contract

Implemented in `decision_contract.py`:

1. **Binary definitive CM** — only `AUTHENTIC`/`DEEPFAKE`; abstentions excluded (never scored as authentic).
2. **Coverage** = definitive / n; **abstention rate** = abstained / n; **selective error** among definitive.
3. **Three-outcome** table: authentic/fake × {AUTHENTIC, DEEPFAKE, ABSTAIN}.
4. **Disclosed legacy binary** — optional `UNCERTAIN→not-fake` mapping only when labeled as such.
5. **Four-category** — `predicted_category` vs GT; CM, macro-F1, per-class P/R/F1.
6. Binary selective and four-category metrics derived from the **same** frozen records (`metrics_from_frozen_records`).

---

## 6. McNemar reconciliation (from prior audit)

Source: `reconcile_ppo_v3_evaluation.py` on n=217 paired `final_test` records.

| Comparison | What was compared | b (corr✓ ppo✗) | c (ppo✓ corr✗) | χ² | p |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Verdict-based | Corrected_fixed binary (`DEEPFAKE` vs not) vs PPO verdict binary (`DEEPFAKE` vs not; UNCERTAIN→not-fake) | 4 | 0 | 2.2500 | 0.1336 |
| Category-based | Same corrected_fixed binary vs PPO **binary implied by** `predicted_category ≠ AUTHENTIC` | 6 | 0 | 4.1667 | 0.0412 |

**Clarifications:**

- Both tests are **binary** McNemar tests, not four-category tests.
- Corrected fixed fusion historically emitted **no** four-category predictions; there is **no** paired 4-way McNemar.
- Calling the second row a “significant four-category comparison” is incorrect. It is a binary comparison on category-implied fake labels.
- Non-significant p=0.1336 does **not** prove equivalence; corrected_fixed still had fewer FPs on this frozen set.
- Discordant pattern `c=0` means PPO never uniquely corrected a corrected_fixed error on those pairs.

---

## 7. Frontend contract (`Console.tsx`)

| Verdict | Presentation |
| :--- | :--- |
| Definitive `AUTHENTIC` | Green + CheckCircle; “AUTHENTIC MEDIA” |
| Definitive `DEEPFAKE` | Red + AlertTriangle; “DEEPFAKE DETECTED” |
| `UNCERTAIN` | Amber + HelpCircle; “INCONCLUSIVE — UNCERTAIN”; explicit “ABSTENTION — NOT AUTHENTIC”; category shown as **HYPOTHESIS** |
| `UNVERIFIED` | Amber; no authenticity claim |
| Leading category | Always distinguishable; under abstention labeled as hypothesis |
| Unavailable agents | Surfaced in subtitle; no fabricated consensus claim when modalities missing |

Streaming behavior and existing styling language preserved.

---

## 8. Tests & commands executed

```text
cd veritas/backend
python -m unittest test_decision_contract.py -v          # 11 passed
python -m unittest test_ppo_v3_integration -v            # 6 passed
python -m unittest test_ppo_fusion_v2 -v                 # 7 passed
python test_ppo_fusion.py                                # 51 passed
python test_fusion.py                                    # ALL FUSION TESTS PASSED
python -m unittest test_integration_routing_fusion -v    # 44 passed (script)
python test_audio_agent_production_integration.py        # 46 passed
cd ../frontend && npx tsc --noEmit                       # OK
```

---

## 9. Modified / created files

**Created**

- `veritas/backend/decision_contract.py`
- `veritas/backend/test_decision_contract.py`
- `veritas/backend/training/ppo_decision_contract.schema.json`
- `veritas/backend/training/ppo_decision_contract_report.md` (this file)

**Modified**

- `veritas/backend/fusion_policy.py` — contract enrichment; corrected_fixed 4-way emission
- `veritas/backend/main.py` — summary abstention language; stream fallback `corrected_fixed`; contract payload fields
- `veritas/frontend/src/pages/Console.tsx` — UNCERTAIN ≠ AUTHENTIC UI semantics
- `veritas/backend/test_fusion.py` — pin legacy mode for legacy-weight expectations; abstention summary check

**Preserved (untouched)**

- All detector / PPO checkpoints (`ppo_fusion_policy_v3.pth`, LC detectors, etc.)
- Historical eval reports / record-level CSVs / recomputed JSON from the consistency audit
- Production default `corrected_fixed`

---

## 10. Recommendation

- **Do not enable PPO v3 as default** solely because the contract is repaired.
- Contract + selective evaluation are ready for a **separate prospective evaluation** (new frozen run with disclosed surfaces).
- **Optional** next training run only if pursuing validation-set probability calibration or a formally selected operating point; keep v3 checkpoint immutable and version a successor.
- Do **not** claim PPO outperforms corrected_fixed without new supporting evidence beyond the reconciled frozen metrics.

---

## Final status

**`DECISION CONTRACT VERIFIED`**
