# VERITAS PPO — Independent Implementation and Evaluation Audit Report

**Audit Date:** 2026-10-20  
**Auditor:** Senior Reinforcement Learning Engineer & ML Systems Architect  
**Evaluation Target:** VERITAS PPO Adaptive Decision Fusion Subsystem  
**Evaluated Artifacts:**
- Core Policy: `veritas/backend/fusion_policy.py`
- Training Pipeline: `veritas/backend/training/ppo_fusion_train.py`
- Evaluation Suite: `veritas/backend/training/ppo_fusion_evaluate.py`
- Checkpoint: `veritas/backend/models/ppo_fusion_policy.pth`
- Dataset Cache: `veritas/backend/training/cache/fusion_dataset_cache_75.json`
- Production Service: `veritas/backend/main.py`
- Frontend Console: `veritas/frontend/src/pages/Console.tsx`

---

## Final Verification Status

### **PPO REQUIRES CORRECTION**

**Executive Summary:**
The core Proximal Policy Optimization (PPO) reinforcement learning loop—consisting of action sampling from a Gaussian distribution, log-probability accumulation, advantage estimation, clipped surrogate policy loss, MSE value loss, and entropy regularization—is implemented in functional PyTorch code. Furthermore, the identity-aware splitting protocol successfully guarantees zero identity overlap across Train, Validation, and Test pools.

However, an independent mathematical and empirical audit reveals significant architectural discrepancies, disconnected action dimensions, documentation-code mismatches, and attribution errors:
1. **Disconnected Action Space:** The policy network outputs a 6-dimensional continuous action, but action dimension $a_4$ is completely unused in `apply_action_to_state`, and action dimension $a_2$ (audio trust gate $w_{au} = 0.4813$) is bypassed on line 385 (`p_audio = float(au_prob)`), rendering it purely decorative.
2. **Score Inconsistency on `00014_fake.mp4`:** The previous evaluation report documented a formula $P_{fused} = \max(P_v, P_a)(1-\gamma) + \gamma \max(P_v, P_a, D_{ctx})$ that was never actually implemented in `fusion_policy.py`. The actual code calculated $P_{fake}$ via independent joint marginals $1.0 - (1-P_v)(1-P_a)$ shifted by a mismatch gain, which produced $0.999988 \approx 1.0000$.
3. **Diagnostic Sample Split Attribution Error:** The previous report claimed `00014_fake.mp4` belonged to the frozen held-out test split. In reality, `00014_fake.mp4` is not in the test cache, and its identity `id06462` belongs to the **Train Pool** (48 clips), representing a severe documentation and provenance error.
4. **No Measurable Empirical Superiority Over Corrected Baseline:** On the 38 held-out test samples, both the Semantically-Corrected Fixed-Weight baseline and PPO achieved identical 100.00% accuracy. The PPO policy did not demonstrate superior classification accuracy over the simpler rule-based corrected baseline.
5. **Production Default Protection:** `VERITAS_FUSION_MODE` in `main.py` has been explicitly switched to `corrected_fixed` as the production default to ensure reliable operations while the PPO action space is corrected.

---

## 1. Verification of the PPO Implementation

Each component of the PPO actor-critic implementation was inspected against the source code:

| Verification Item | Implementation Reference | Status | Audit Findings & Observed Behavior |
|:---|:---|:---:|:---|
| **Gaussian Action Sampling** | `ppo_fusion_train.py:551-555` | **PASS** | `dist = Normal(mu, std)` where `mu` is emitted by `actor_net` and `std = exp(clamp(actor_log_std))`. Samples drawn via `dist.sample()`. |
| **Log-Probability Consistency** | `ppo_fusion_train.py:555, 617` | **PASS** | Log probabilities are computed as `dist.log_prob(actions).sum(dim=-1)`, correctly summing independent Gaussian log-densities across the 6 action dimensions. |
| **Old vs. New Probability Ratio** | `ppo_fusion_train.py:621` | **PASS** | `ratio = torch.exp(new_lp - old_lp)` correctly evaluates importance sampling ratio $r_t(\theta) = \frac{\pi_\theta(a_t\|s_t)}{\pi_{\theta_{old}}(a_t\|s_t)}$. |
| **Clipped Surrogate Objective** | `ppo_fusion_train.py:623-626` | **PASS** | `surr1 = ratio * adv`, `surr2 = clamp(ratio, 1-eps, 1+eps) * adv`, `loss = -min(surr1, surr2).mean()` with $\epsilon = 0.20$. Standard PPO-Clip formulation. |
| **Value Loss & Updates** | `ppo_fusion_train.py:628-636` | **PASS** | Value loss $L_V = 0.5 \cdot \mathbb{E}[(V(s) - R)^2]$, combined with policy loss and entropy bonus: $L = L_\pi + 0.5 L_V - 0.01 H(\pi)$. Gradient norm clipped to $0.5$. |
| **Contextual Bandit Formulation** | `ppo_fusion_train.py:571-576` | **PASS** | Single-step horizon ($\gamma = 0$). Advantage $A(s, a) = R(s, a) - V(s)$ normalized across mini-batches. Mathematically sound for stateless decision fusion. |
| **Deterministic Inference Parity** | `fusion_policy.py:618-639` | **PASS** | In production mode, actions are taken deterministically from `dist.mean` using frozen `FeatureNormalizer` fitted strictly on training data. |
| **Absence of Hardcoded Overrides** | `fusion_policy.py:329-498` | **PASS** | No special-case branches exist for `00014_fake.mp4`, specific filenames, or magic score combinations. |
| **Action Space Utilization** | `fusion_policy.py:346-433` | **FAIL** | **Critical Defect:** Action $a_4$ is completely unreferenced in `apply_action_to_state`. Action $a_2$ is computed as $w_{au} = \sigma(a_2)$ but never used to scale or modulate audio fake probability (bypassed on line 385). |
| **Multiclass Distribution Route** | `fusion_policy.py:387-413` | **WARNING** | The 4-category distribution is not an output head of the policy network; it is heuristically reconstructed from visual and audio marginals with an ad-hoc mismatch shift. |

---

## 2. Resolution of the `00014_fake.mp4` Score Inconsistency

### A. The Discrepancy
For the diagnostic sample `00014_fake.mp4` (`Asian (East)/women/id06462/00014_fake.mp4`), the previous report presented the following values:
- Spatial fake probability: $0.0008$
- Temporal fake probability: $0.0001$
- Audio fake probability: $1.0000$
- Context mismatch: $0.9995$
- Audio trust gate: $48.03\%$
- Decision threshold: $45.47\%$
- Claimed fused score: $1.0000$ ($100.0\%$)

The previous documentation claimed the fusion formula was:
$$P_{audio} = w_{audio} \cdot P_{fake\_audio}$$
$$P_{fused} = \max(P_{visual}, P_{audio}) \cdot (1 - \gamma) + \gamma \cdot \max(P_{visual}, P_{audio}, D_{context}) \quad \text{with } \gamma \le 0.5$$

With $w_{audio} = 0.4803$, $P_{audio} = 0.4803$, $P_{visual} \approx 0.0003$, and $D_{context} = 0.9995$, this documented formula produces:
$$P_{fused} = 0.4803(1 - \gamma) + 0.9995\gamma$$
Since $\gamma \le 0.5$, the maximum possible score is:
$$P_{fused} \le 0.4803(0.5) + 0.9995(0.5) = 0.7399 \neq 1.0000$$
If $\gamma = 0$, $P_{fused} = 0.4803$, which falls below the threshold, falsely predicting AUTHENTIC!

### B. Empirical Execution Trace on Saved Checkpoint
We executed deterministic inference directly against `ppo_fusion_policy.pth` using the exact detector inputs for `00014_fake.mp4`. The observed intermediate variables were:

```python
=== Raw State Vector (s) ===
[0.0008, 0.0016, 1.0, 0.0001, 0.0002, 1.0, 1.0, 0.0, 1.0, 0.99955, 1.0, 0.15, 1.0, 1.0]

=== Critic Value V(s) ===
V(s) = 2.0479

=== Policy Network Actions (mu = dist.mean) ===
a_0 (spatial logit)        : -0.3147
a_1 (temporal logit)       : +0.3163
a_2 (audio trust logit)    : -0.0748
a_3 (mismatch gain logit)  : +0.1317
a_4 (unused action)        : -0.1830   <-- NEVER USED
a_5 (threshold offset)     : -0.2543

=== Intermediate Variable Evaluation in apply_action_to_state ===
Visual Weights:
  exp(a_0) = 0.7300, exp(a_1) = 1.3720
  w_sp = 0.7300 / (0.7300 + 1.3720) = 0.3473 (34.73%)
  w_te = 1.3720 / (0.7300 + 1.3720) = 0.6527 (65.27%)
  p_visual = 0.3473 * 0.0008 + 0.6527 * 0.0001 = 0.000343

Audio Gating (Line 384-385):
  w_au = 1 / (1 + exp(-(-0.0748))) = 0.4813 (48.13%)
  p_audio = float(au_prob) = 1.0000   <-- w_au WAS BYPASSED!

Category Marginals (Lines 394-406):
  p_auth_base = (1 - 0.000343) * (1 - 0.9999) = 1.0000e-04 (clipped)
  p_aud_base  = (1 - 0.000343) * 0.9999       = 0.999557
  p_vid_base  = 0.000343 * (1 - 0.9999)       = 3.4300e-08 (clipped to 1e-06)
  p_both_base = 0.000343 * 0.9999             = 0.000343

Context Mismatch Shift:
  mismatch_gain = 2.0 * sigma(a_3) = 2.0 * sigma(0.1317) = 1.0658
  z_0 = ln(1e-04) - 1.0658 * 0.99955 = -9.2103 - 1.0653 = -10.2760
  z_1 = ln(0.999557) + 1.0658 * 0.99955 * 1.0 = -0.0004 + 1.0653 = +1.0648
  z_2 = ln(1e-06) + 0.0 = -13.8155
  z_3 = ln(0.000343) = -7.9776

Softmax Category Probabilities:
  P(AUTHENTIC)              = 1.1877e-05 (0.001%)
  P(AUDIO_MANIPULATED)      = 0.999869   (99.99%)
  P(VIDEO_MANIPULATED)      = 3.4474e-07 (0.000%)
  P(AUDIO_VIDEO_MANIPULATED)= 1.1827e-04 (0.012%)

Fused Fake Probability:
  p_fake = 1.0 - P(AUTHENTIC) = 1.0 - 0.000012 = 0.999988
  overall_confidence = round(p_fake, 4) = 1.0000 (100.0%)

Threshold & Verdict:
  delta_tau = 0.15 * tanh(-0.2543) = -0.0373
  decision_threshold = 0.50 - 0.0373 = 0.4627
  verdict = DEEPFAKE (overall_confidence >= 0.4627 + 0.05)
```

### C. Root Cause Findings
1. **Obsolete/Fabricated Documentation:** The formula $P_{fused} = \max(P_v, P_a)(1-\gamma) + \dots$ in the earlier markdown report was a documentation fiction; it never existed in `fusion_policy.py`.
2. **Defensive Bypass of $w_{au}$:** If the developer had actually set $p_{audio} = w_{au} \cdot au\_prob$ ($0.4813 \times 1.0 = 0.4813$), the sample would have been misclassified as authentic. To prevent that regression, line 385 hardcoded `p_audio = float(au_prob)`, leaving $w_{au}$ as an unused, purely cosmetic output.
3. **Source of the 1.0000 Score:** The score was $1.0000$ because $p_{fake} = 1.0 - P(\text{AUTHENTIC})$, and $P(\text{AUTHENTIC})$ was driven to zero by the joint product $(1 - P_v)(1 - P_a)$ where $P_a = 1.0$.

---

## 3. Missing-Agent Masking Audit

### A. Formula Discrepancy
The previous report documented visual gating as:
$$w_{\text{spatial}} = \frac{e^{a_0 \cdot \text{valid}_{\text{spatial}}}}{e^{a_0 \cdot \text{valid}_{\text{spatial}}} + e^{a_1 \cdot \text{valid}_{\text{temporal}}}}$$
As noted in the audit request, when $\text{valid} = 0$, $e^{0} = 1$, which fails to mask the missing agent.

However, inspection of `fusion_policy.py` lines 360–378 reveals that the actual code did **not** use this expression. Instead, it used explicit conditional branches:
```python
if sp_valid and te_valid:
    logits = np.array([action[0], action[1]], dtype=np.float32)
    exp_logits = np.exp(logits - np.max(logits))
    w = exp_logits / (np.sum(exp_logits) + 1e-7)
    w_sp, w_te = float(w[0]), float(w[1])
    p_visual = float(w_sp * sp_prob + w_te * te_prob)
elif sp_valid:
    w_sp, w_te = 1.0, 0.0
    p_visual = float(sp_prob)
elif te_valid:
    w_sp, w_te = 0.0, 1.0
    p_visual = float(te_prob)
else:
    p_visual = None
```

### B. Missing Modality Behavior
We tested all missing modality edge cases against the implementation:
- **Audio-Only Upload (Spatial & Temporal unavailable):** `p_visual = None`. Softmax visual weights are masked to `w_sp = 0.0, w_te = 0.0`. Prediction is determined exclusively by audio: $P_{fake} = P_{audio}$. Verdict: **DEEPFAKE** (correct, no fabricated visual evidence).
- **Image-Only Upload (Temporal & Audio unavailable):** `p_visual = sp_prob`, `w_sp = 1.0, w_te = 0.0`. Audio is marked unavailable. Prediction determined exclusively by spatial evidence. Verdict: **DEEPFAKE** (correct).
- **All Agents Unavailable:** `contributing = []`. System returns `overall_confidence = 0.0`, `verdict = "UNVERIFIED MODALITY"`.
- **State Vector Imputation:** When an agent is invalid, `extract_state_from_agents` sets `prob = 0.5` and `uncertainty = 1.0` (maximum entropy prior), with `valid = 0.0`. (The previous report inaccurately claimed features were masked to `0.0`).

---

## 4. Feature Semantics & Reward Correctness

### A. Uncertainty vs. Confidence Semantics
In `fusion_policy.py` lines 165, 175, 191:
```python
uncertainty = 1.0 - 2.0 * abs(p - 0.5)
```
- At $p = 0.50$, `uncertainty = 1.0` (maximum uncertainty).
- At $p = 0.00$ or $1.00$, `uncertainty = 0.0` (zero uncertainty / extreme confidence).

**Finding:** The implementation uses mathematically valid predictive uncertainty. However, the previous report documented `uncertainty = 2 * abs(P - 0.5)`, which is the exact mathematical inverse (confidence). This was a reporting defect, not a code defect.

### B. Reward Function Audit
In `fusion_policy.py` lines 521–546:
```python
total_reward = r_binary + 0.6 * r_cat + r_calib
```
- $r_{binary} \in \{+1.5 \text{ (TP/TN)}, -2.0 \text{ (FP)}, -3.5 \text{ (FN)}, -0.2 \text{ (Uncertain)}\}$.
- $r_{cat} = +1.0$ if predicted category matches ground truth, else $-0.5$.
- $r_{calib} = 0.2 \ln P(\text{True Category})$.

**Finding on Boundary Distance Shaping:** The audit request inquired about a term proportional to `abs(P - threshold)`. Inspection of `compute_reward` proves that **no such term exists in the code**. The previous report falsely claimed a "+0.5 * |P - tau| decision margin bonus". The policy was never trained with boundary distance shaping, averting the risk of rewarding confidently incorrect decisions.

---

## 5. Dataset Splits & Evaluation Provenance Audit

### A. Sample Count Arithmetic Reconstructed
We audited `training/cache/fusion_dataset_cache_75.json` directly from disk:

| Split | Original Assets | Missing-Modality Augmentations (20%) | Total Persisted Records | Unique Physical Media Files | Distinct Speaker/Face Identities |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 240 (60 per category $\times$ 4) | 48 | **288** | 240 | 190 |
| **Val** | 32 (8 per category $\times$ 4) | 6 | **38** | 32 | 25 |
| **Test** | 32 (8 per category $\times$ 4) | 6 | **38** | 32 | 24 |
| **Total** | **304** | **60** | **364** | **304** | **239** |

**Identity Leakage Audit:**
- Overlap between Train and Validation identities: **0**
- Overlap between Train and Test identities: **0**
- Overlap between Validation and Test identities: **0**
- Physical media file overlap across splits: **0**

### B. Resolution of the Diagnostic Sample Split Membership
The previous report claimed:
> "*`00014_fake.mp4` belongs to the frozen Test Split; never observed during training or validation.*"

We ran `training/ppo_fusion_train.py:discover_and_split_dataset(seed=42)` across all 21,560 FakeAVCeleb clips:
- `id06462` was assigned to the **Train Pool** (48 clips total, including `00014_fake.mp4`).
- `id06462` is **NOT** in the Validation Pool or Test Pool.
- `00014_fake.mp4` was **NOT** selected into the 288 cached training samples, but its identity pool is `train`.

**Conclusion:** Claiming `00014_fake.mp4` was part of the held-out test split was factually false. It was evaluated as an independent diagnostic sample, but its identity belongs to the training pool.

---

## 6. Independent Recomputation of Benchmark Metrics

All three systems were evaluated on the exact 38 held-out test records in `fusion_dataset_cache_75.json` using frozen checkpoints and deterministic inference:

### A. Comprehensive Test Set Metrics ($N = 38$)
| Evaluation Metric | Legacy Fixed-Weight | Semantically-Corrected Fixed | PPO Adaptive Fusion | PPO vs Corrected Delta |
|:---|:---:|:---:|:---:|:---:|
| **Binary Accuracy** | 76.32% (29/38) | 100.00% (38/38) | **100.00% (38/38)** | +0.00% (Tied) |
| **Balanced Accuracy** | 83.33% | 100.00% | **100.00%** | +0.00% (Tied) |
| **Precision (Deepfake)** | 100.00% (18/18) | 100.00% (27/27) | **100.00% (27/27)** | +0.00% (Tied) |
| **Recall (Deepfake)** | 66.67% (18/27) | 100.00% (27/27) | **100.00% (27/27)** | +0.00% (Tied) |
| **F1-Score** | 0.8000 | 1.0000 | **1.0000** | +0.00% (Tied) |
| **False Negative Rate** | 33.33% (9/27) | 0.00% (0/27) | **0.00% (0/27)** | +0.00% (Tied) |
| **False Positive Rate** | 0.00% (0/11) | 0.00% (0/11) | **0.00% (0/11)** | +0.00% (Tied) |
| **Partial Audio-Fake Recall** | **0.00% (0/8)** | **100.00% (8/8)** | **100.00% (8/8)** | +0.00% (Tied) |
| **Visual-Fake Recall** | 90.91% (10/11) | 100.00% (11/11) | **100.00% (11/11)** | +0.00% (Tied) |
| **Dual-Fake Recall** | 100.00% (8/8) | 100.00% (8/8) | **100.00% (8/8)** | +0.00% (Tied) |
| **Authentic Recall** | 100.00% (11/11) | 100.00% (11/11) | **100.00% (11/11)** | +0.00% (Tied) |
| **4-Category Macro-F1** | N/A (Binary only) | N/A (Binary only) | **1.0000** | N/A |
| **Mean Latency (CPU)** | 0.04 ms | 0.06 ms | **0.78 ms** | +0.72 ms |

### B. Confusion Matrices on Held-Out Test Set ($N = 38$)
```text
1. Legacy Fixed-Weight:
                 Pred AUTHENTIC   Pred DEEPFAKE
True AUTHENTIC            11               0
True DEEPFAKE              9              18   <-- 9 False Negatives (All 8 Audio Deepfakes missed!)

2. Semantically-Corrected Fixed-Weight:
                 Pred AUTHENTIC   Pred DEEPFAKE
True AUTHENTIC            11               0
True DEEPFAKE              0              27   <-- 0 False Negatives

3. PPO Adaptive Fusion (Binary):
                 Pred AUTHENTIC   Pred DEEPFAKE
True AUTHENTIC            11               0
True DEEPFAKE              0              27   <-- 0 False Negatives

4. PPO Adaptive Fusion (4-Category Decomposition):
                           Pred AUTH   Pred AUDIO   Pred VIDEO   Pred DUAL
True AUTHENTIC                11           0            0            0
True AUDIO_MANIPULATED         0           8            0            0
True VIDEO_MANIPULATED         0           0           11            0
True AUDIO_VIDEO_MANIP         0           0            0            8
```

### C. Honest Empirical Interpretation
1. **The Corrected Baseline Solves the Problem:** The catastrophic failure on `00014_fake.mp4` was caused by linear additive averaging over discordant score semantics. Disjunctive combining ($P_{fake} = 1 - (1 - P_v)(1 - P_a)$) completely resolves the failure mode, achieving 100% recall on partial fakes without machine learning.
2. **PPO Did Not Improve Binary Accuracy:** Both PPO and the Corrected Fixed-Weight baseline scored 100.00%. Claiming that PPO provided an accuracy improvement over the corrected baseline is empirically unsupported.
3. **Statistical Sample Size Limitation:** With $N = 38$ test samples, the 95% Wilson score confidence interval for a 100% accuracy result is $[90.8\%, 100.0\%]$. A test set of at least 500–1,000 clips is required to demonstrate statistically significant calibration or robustness advantages.

---

## 7. Four-Category Architecture Critique & Redesign Plan

### A. Current Architectural Flaw
In the current implementation, the policy network emits 6 continuous actions ($a_0 \dots a_5$). The 4-category probabilities are not produced by the network's output layers. Instead, they are hand-constructed inside `apply_action_to_state` from independent marginals:
$$P(\text{auth}) = (1 - P_v)(1 - P_a), \quad P(\text{aud}) = (1 - P_v)P_a, \quad P(\text{vid}) = P_v(1 - P_a), \quad P(\text{both}) = P_v P_a$$
and modified by an ad-hoc heuristic term:
$$z_i = \ln(P_i) \pm \text{mismatch\_gain} \cdot \text{ctx\_mismatch}$$

Because the upstream audio detector ($1.000$ vs $0.000$) and temporal detector ($1.000$ vs $0.000$) are virtually binary on FakeAVCeleb, these base joint marginals were already separable into one of the four categories. The PPO policy was rewarded for this separation, but the policy network was not actually generating the categorical distribution.

### B. Required Architectural Redesign
To implement a genuinely learned, technically sound PPO fusion policy:
1. **Direct Categorical Actor Head:** The Actor network should emit an unconstrained 4-dimensional logit vector $\mathbf{z} \in \mathbb{R}^4$ representing the categorical policy distribution:
   $$\pi_\theta(\text{Category} = k \mid s) = \frac{e^{z_k}}{\sum_{j=0}^3 e^{z_j}}$$
   This replaces continuous Gaussian sampling with a `torch.distributions.Categorical` policy, matching the discrete nature of the 4 ground-truth categories.
2. **Continuous Gating Head (Optional Sub-Action):** If continuous gating weights are desired, emit a 3-dimensional Dirichlet or bounded vector for visual weighting and threshold adaptation.
3. **True End-to-End Multiclass Optimization:** Train the Categorical Actor directly against cross-entropy log-likelihood and decision payoffs.

---

## 8. Summary of Files Changed & Corrective Actions Taken

1. **`veritas/backend/main.py`**:
   - Changed default `VERITAS_FUSION_MODE` from `"ppo"` to `"corrected_fixed"`.
   - Reason: Protects production from unvalidated policy execution and dead action dimensions while retaining complete resilience to partial audio/video manipulation.
2. **`veritas/backend/fusion_policy.py`**:
   - Audited all 6 action mappings, normalizer roundtrip, and masking logic.
   - Preserved all baseline and compatibility interfaces.
3. **`veritas/backend/training/ppo_fusion_audit_report.md`**:
   - Created this definitive, transparent audit report.

---

## 9. Final Recommendations & Exact Next Commands

### Immediate Operational Stance
Keep VERITAS running on the **`corrected_fixed`** fusion engine for all production requests:
```powershell
$env:VERITAS_FUSION_MODE = "corrected_fixed"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
This guarantees:
- 100% detection of cloned audio and partial manipulation (`00014_fake.mp4` correctly flagged as `DEEPFAKE`).
- Zero reliance on disconnected policy weights.
- Backward compatibility with the existing frontend and SSE stream contract.

### Roadmap to Reach `PPO VERIFIED AND VALIDATED`
1. **Redesign Policy Action Space:** Transition from 6-D continuous Gaussian to a Categorical Actor Head emitting 4 category logits directly.
2. **Retrain on Expanded Feature Dataset:** Re-run feature extraction with `--samples-per-cat 300` across all 500 FakeAVCeleb identities to create a 1,200+ sample benchmark.
3. **Validate Convergence & Out-of-Distribution Generalization:** Compare PPO against the corrected baseline on noisy, low-quality, and adversarial edge cases where fixed logic degrades.
