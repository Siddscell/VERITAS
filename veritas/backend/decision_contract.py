"""
decision_contract.py
====================
Single, documented output and evaluation contract for VERITAS fusion decisions.

Field semantics
---------------
category_probabilities
    Normalized four-category distribution over
    {AUTHENTIC, AUDIO_MANIPULATED, VIDEO_MANIPULATED, AUDIO_VIDEO_MANIPULATED}.
    When produced by apply_action_to_state these are *model-implied* probabilities
    (softmax over fusion logits / modality marginals). They MUST NOT be described
    as externally calibrated unless a validation calibration procedure has been
    measured and frozen.

predicted_category
    Argmax category hypothesis under the policy rules. Inside the binary
    abstention band this remains a *leading hypothesis*, not a definitive
    authenticity claim.

fake_probability
    P(fake) = P(AUDIO_MANIPULATED) + P(VIDEO_MANIPULATED) + P(AUDIO_VIDEO_MANIPULATED)
    when a valid normalized four-category distribution is available.
    Equivalent to 1 - P(AUTHENTIC).

verdict / verdict_class
    Binary operating decision with abstention:
      DEEPFAKE | AUTHENTIC | UNCERTAIN | UNVERIFIED
    UNCERTAIN is an abstention, never an authentic prediction.

decision_confidence
    Strength of the *binary* operating decision, distinct from fake_probability.
    Zero inside the abstention band; increases with margin outside the band.

decision_thresholds
    Frozen operating points: {tau, tau_low, tau_high, deadband}.
    For PPO: tau = clip(0.50 + 0.15*tanh(a[4]), 0.35, 0.65), deadband = 0.05.
    For corrected_fixed: tau = 0.50, tau_low = 0.45, tau_high = 0.55, deadband = 0.05.

Evaluation surfaces (must stay separate)
----------------------------------------
1. Binary definitive-only: UNCERTAIN excluded from the confusion matrix.
2. Coverage / abstention / selective error among definitive predictions.
3. Three-outcome assessment treating UNCERTAIN as its own class.
4. Four-category: predicted_category vs ground-truth category.
5. Optional disclosed policy: map UNCERTAIN -> AUTHENTIC for legacy binary
   accuracy ONLY when explicitly labeled as such (never silently).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

CATEGORIES = [
    "AUTHENTIC",
    "AUDIO_MANIPULATED",
    "VIDEO_MANIPULATED",
    "AUDIO_VIDEO_MANIPULATED",
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}

# Architectural deadband shared by PPO and corrected_fixed operating rules.
# Frozen design constant (not selected on the held-out test set).
DEFAULT_DEADBAND = 0.05
CORRECTED_FIXED_TAU = 0.50
CORRECTED_FIXED_TAU_LOW = 0.45
CORRECTED_FIXED_TAU_HIGH = 0.55

VERDICT_DEEPFAKE = "DEEPFAKE"
VERDICT_AUTHENTIC = "AUTHENTIC"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICT_UNVERIFIED = "UNVERIFIED"

# Canonical string emitted by fusion_policy for abstention
VERDICT_UNCERTAIN_FULL = "UNCERTAIN / INSUFFICIENT EVIDENCE"
VERDICT_UNVERIFIED_FULL = "UNVERIFIED MODALITY"


def normalize_verdict_class(verdict: Optional[str]) -> str:
    """Map any fusion verdict string to {AUTHENTIC, DEEPFAKE, UNCERTAIN, UNVERIFIED}."""
    if not verdict:
        return VERDICT_UNVERIFIED
    v = str(verdict).strip().upper()
    if v in ("DEEPFAKE", "FAKE"):
        return VERDICT_DEEPFAKE
    if v in ("AUTHENTIC", "REAL"):
        return VERDICT_AUTHENTIC
    if "UNVERIFIED" in v:
        return VERDICT_UNVERIFIED
    if "UNCERTAIN" in v or "INSUFFICIENT" in v:
        return VERDICT_UNCERTAIN
    return VERDICT_UNCERTAIN


def is_abstention(verdict: Optional[str]) -> bool:
    cls = normalize_verdict_class(verdict)
    return cls in (VERDICT_UNCERTAIN, VERDICT_UNVERIFIED)


def fake_probability_from_categories(
    category_probabilities: Mapping[str, float],
) -> float:
    """P(fake) as the sum of the three manipulated category masses."""
    total_manip = (
        float(category_probabilities.get("AUDIO_MANIPULATED", 0.0))
        + float(category_probabilities.get("VIDEO_MANIPULATED", 0.0))
        + float(category_probabilities.get("AUDIO_VIDEO_MANIPULATED", 0.0))
    )
    return float(np.clip(total_manip, 0.0, 1.0))


def categories_sum_to_one(
    category_probabilities: Mapping[str, float],
    tol: float = 1e-3,
) -> bool:
    s = sum(float(category_probabilities.get(c, 0.0)) for c in CATEGORIES)
    return abs(s - 1.0) <= tol


def build_decision_thresholds(
    tau: float,
    deadband: float = DEFAULT_DEADBAND,
) -> Dict[str, float]:
    tau = float(np.clip(tau, 0.0, 1.0))
    deadband = float(max(0.0, deadband))
    return {
        "tau": round(tau, 6),
        "tau_low": round(tau - deadband, 6),
        "tau_high": round(tau + deadband, 6),
        "deadband": round(deadband, 6),
    }


def decision_confidence_from_scores(
    fake_probability: float,
    tau: float,
    deadband: float = DEFAULT_DEADBAND,
) -> float:
    """
    Confidence in the binary operating decision (not a second fake score).

    Inside [tau - deadband, tau + deadband] the policy abstains → 0.0.
    Outside, confidence grows with distance from the nearer band edge,
    normalized into [0, 1].
    """
    p = float(np.clip(fake_probability, 0.0, 1.0))
    tau = float(tau)
    deadband = float(max(0.0, deadband))
    low = tau - deadband
    high = tau + deadband
    if low <= p <= high:
        return 0.0
    if p > high:
        denom = max(1.0 - high, 1e-8)
        return float(np.clip((p - high) / denom, 0.0, 1.0))
    denom = max(low, 1e-8)
    return float(np.clip((low - p) / denom, 0.0, 1.0))


def enrich_decision_fields(
    details: Dict[str, Any],
    *,
    tau: Optional[float] = None,
    deadband: float = DEFAULT_DEADBAND,
    probabilities_are_calibrated: bool = False,
) -> Dict[str, Any]:
    """
    Attach contract fields to a fusion details dict in-place and return it.

    Preserves existing keys; adds aliases and explicit semantics.
    """
    out = details
    verdict = out.get("verdict")
    verdict_class = normalize_verdict_class(verdict)
    out["verdict_class"] = verdict_class
    out["is_abstention"] = verdict_class in (VERDICT_UNCERTAIN, VERDICT_UNVERIFIED)

    cat_probs = out.get("category_probabilities")
    if isinstance(cat_probs, dict) and categories_sum_to_one(cat_probs, tol=1e-2):
        p_fake = fake_probability_from_categories(cat_probs)
        out["fake_probability"] = round(float(p_fake), 6)
        # Keep p_fake aligned when present
        if "p_fake" not in out or out.get("p_fake") is None:
            out["p_fake"] = float(p_fake)
        else:
            # Soft consistency: prefer sum-of-manip when distribution is valid
            out["p_fake"] = float(p_fake)
            out["fake_probability"] = round(float(p_fake), 6)
    else:
        p_fake = float(out.get("p_fake", out.get("overall_confidence", 0.0)) or 0.0)
        out["fake_probability"] = round(float(np.clip(p_fake, 0.0, 1.0)), 6)

    if tau is None:
        thr = out.get("decision_threshold")
        if thr is not None:
            tau = float(thr)
        elif out.get("method") == "corrected_fixed":
            tau = CORRECTED_FIXED_TAU
        else:
            tau = 0.50

    thresholds = build_decision_thresholds(float(tau), deadband)
    # corrected_fixed historically uses hard 0.45 / 0.55 cutoffs (= tau±deadband at 0.50)
    if out.get("method") == "corrected_fixed":
        thresholds = {
            "tau": CORRECTED_FIXED_TAU,
            "tau_low": CORRECTED_FIXED_TAU_LOW,
            "tau_high": CORRECTED_FIXED_TAU_HIGH,
            "deadband": DEFAULT_DEADBAND,
        }
        tau = CORRECTED_FIXED_TAU

    out["decision_thresholds"] = thresholds
    if "decision_threshold" not in out or out["decision_threshold"] is None:
        out["decision_threshold"] = thresholds["tau"]

    out["decision_confidence"] = round(
        decision_confidence_from_scores(
            float(out["fake_probability"]),
            float(tau),
            deadband=float(thresholds["deadband"]),
        ),
        6,
    )
    out["category_probabilities_calibrated"] = bool(probabilities_are_calibrated)
    out["leading_category_is_hypothesis"] = bool(out["is_abstention"])
    return out


# ---------------------------------------------------------------------------
# Evaluation surfaces (from frozen record-level predictions)
# ---------------------------------------------------------------------------

def _as_int_array(x: Sequence[Union[int, bool]]) -> np.ndarray:
    return np.asarray(list(x), dtype=int)


def binary_confusion_definitive(
    y_true_fake: Sequence[int],
    verdicts: Sequence[str],
) -> Dict[str, Any]:
    """
    Binary confusion matrix over definitive predictions only.

    UNCERTAIN / UNVERIFIED are excluded (never counted as AUTHENTIC).
    pred_fake = 1 iff verdict_class == DEEPFAKE.
    """
    y_t = _as_int_array(y_true_fake)
    classes = [normalize_verdict_class(v) for v in verdicts]
    mask = np.array([c in (VERDICT_AUTHENTIC, VERDICT_DEEPFAKE) for c in classes], dtype=bool)
    if not np.any(mask):
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "n_definitive": 0,
            "n_total": int(len(y_t)),
            "excluded_abstentions": int(len(y_t)),
        }
    y_td = y_t[mask]
    y_pd = np.array(
        [1 if c == VERDICT_DEEPFAKE else 0 for c, m in zip(classes, mask) if m],
        dtype=int,
    )
    tp = int(np.sum((y_td == 1) & (y_pd == 1)))
    tn = int(np.sum((y_td == 0) & (y_pd == 0)))
    fp = int(np.sum((y_td == 0) & (y_pd == 1)))
    fn = int(np.sum((y_td == 1) & (y_pd == 0)))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_definitive": int(mask.sum()),
        "n_total": int(len(y_t)),
        "excluded_abstentions": int((~mask).sum()),
    }


def selective_binary_metrics(
    y_true_fake: Sequence[int],
    verdicts: Sequence[str],
) -> Dict[str, Any]:
    """Coverage, abstention rate, and error rate among definitive predictions."""
    y_t = _as_int_array(y_true_fake)
    n = len(y_t)
    classes = [normalize_verdict_class(v) for v in verdicts]
    definitive = [c in (VERDICT_AUTHENTIC, VERDICT_DEEPFAKE) for c in classes]
    n_def = int(sum(definitive))
    n_abs = n - n_def
    cm = binary_confusion_definitive(y_true_fake, verdicts)
    errors = cm["fp"] + cm["fn"]
    correct = cm["tp"] + cm["tn"]
    coverage = n_def / max(n, 1)
    abstention_rate = n_abs / max(n, 1)
    selective_error = errors / max(n_def, 1)
    selective_accuracy = correct / max(n_def, 1)
    return {
        "n_total": n,
        "n_definitive": n_def,
        "n_abstained": n_abs,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_error_rate": selective_error,
        "selective_accuracy": selective_accuracy,
        "confusion_definitive": cm,
        "note": (
            "UNCERTAIN/UNVERIFIED are excluded from the definitive confusion matrix; "
            "they are never silently counted as AUTHENTIC."
        ),
    }


def three_outcome_assessment(
    y_true_fake: Sequence[int],
    verdicts: Sequence[str],
) -> Dict[str, Any]:
    """
    Three-outcome table:
      outcomes = {AUTHENTIC_PRED, DEEPFAKE_PRED, ABSTAIN}
    against binary ground truth {authentic, fake}.
    """
    y_t = _as_int_array(y_true_fake)
    classes = [normalize_verdict_class(v) for v in verdicts]
    counts = {
        "gt_authentic_pred_authentic": 0,
        "gt_authentic_pred_deepfake": 0,
        "gt_authentic_abstain": 0,
        "gt_fake_pred_authentic": 0,
        "gt_fake_pred_deepfake": 0,
        "gt_fake_abstain": 0,
    }
    for yt, c in zip(y_t.tolist(), classes):
        if yt == 0:
            if c == VERDICT_AUTHENTIC:
                counts["gt_authentic_pred_authentic"] += 1
            elif c == VERDICT_DEEPFAKE:
                counts["gt_authentic_pred_deepfake"] += 1
            else:
                counts["gt_authentic_abstain"] += 1
        else:
            if c == VERDICT_AUTHENTIC:
                counts["gt_fake_pred_authentic"] += 1
            elif c == VERDICT_DEEPFAKE:
                counts["gt_fake_pred_deepfake"] += 1
            else:
                counts["gt_fake_abstain"] += 1
    n = max(len(y_t), 1)
    return {
        "counts": counts,
        "rates": {k: v / n for k, v in counts.items()},
        "n": int(len(y_t)),
    }


def legacy_binary_mapping_disclosed(
    y_true_fake: Sequence[int],
    verdicts: Sequence[str],
    *,
    uncertain_maps_to_fake: bool = False,
) -> Dict[str, Any]:
    """
    Conventional binary accuracy with an *explicit* abstention mapping policy.

    Default historical (and previously silent) policy: UNCERTAIN → not-fake (0).
    This helper always discloses the mapping; callers must not present it as
    selective accuracy.
    """
    y_t = _as_int_array(y_true_fake)
    y_p = []
    for v in verdicts:
        c = normalize_verdict_class(v)
        if c == VERDICT_DEEPFAKE:
            y_p.append(1)
        elif c == VERDICT_AUTHENTIC:
            y_p.append(0)
        else:
            y_p.append(1 if uncertain_maps_to_fake else 0)
    y_p_arr = np.asarray(y_p, dtype=int)
    tp = int(np.sum((y_t == 1) & (y_p_arr == 1)))
    tn = int(np.sum((y_t == 0) & (y_p_arr == 0)))
    fp = int(np.sum((y_t == 0) & (y_p_arr == 1)))
    fn = int(np.sum((y_t == 1) & (y_p_arr == 0)))
    n = len(y_t)
    return {
        "policy": (
            "UNCERTAIN→DEEPFAKE" if uncertain_maps_to_fake else "UNCERTAIN→AUTHENTIC/not-fake"
        ),
        "disclosed": True,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": n,
        "accuracy": (tp + tn) / max(n, 1),
        "warning": (
            "This metric folds abstentions into a binary label. "
            "Report selective metrics separately; do not hide abstentions."
        ),
    }


def four_category_metrics(
    y_true_cat: Sequence[int],
    y_pred_cat: Sequence[int],
    n_classes: int = 4,
) -> Dict[str, Any]:
    """Four-category CM, macro-F1, and per-category precision/recall/F1."""
    y_t = _as_int_array(y_true_cat)
    y_p = _as_int_array(y_pred_cat)
    assert len(y_t) == len(y_p)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_t.tolist(), y_p.tolist()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    per_cat: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for i, name in enumerate(CATEGORIES[:n_classes]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_cat[name] = {
            "support": support,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp_diagonal": tp,
        }
        f1s.append(f1)
    n = len(y_t)
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(np.trace(cm) / max(n, 1)),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_category": per_cat,
        "n": n,
    }


def metrics_from_frozen_records(
    records: Sequence[Mapping[str, Any]],
    *,
    verdict_key: str = "verdict",
    pred_cat_key: str = "predicted_category",
    gt_fake_key: str = "gt_is_fake",
    gt_cat_key: str = "gt_cat_idx",
) -> Dict[str, Any]:
    """
    Derive binary selective metrics and four-category metrics from the *same*
    frozen record-level prediction table.
    """
    verdicts = [str(r[verdict_key]) for r in records]
    y_fake = [int(r[gt_fake_key]) for r in records]
    y_true_cat = [int(r[gt_cat_key]) for r in records]
    y_pred_cat = []
    for r in records:
        cat = r[pred_cat_key]
        if isinstance(cat, int):
            y_pred_cat.append(int(cat))
        else:
            y_pred_cat.append(int(CATEGORY_TO_IDX.get(str(cat), 0)))

    selective = selective_binary_metrics(y_fake, verdicts)
    three = three_outcome_assessment(y_fake, verdicts)
    legacy = legacy_binary_mapping_disclosed(y_fake, verdicts, uncertain_maps_to_fake=False)
    four = four_category_metrics(y_true_cat, y_pred_cat)
    return {
        "n": len(records),
        "binary_selective": selective,
        "three_outcome": three,
        "legacy_binary_disclosed_uncertain_as_authentic": legacy,
        "four_category": four,
        "surfaces_share_same_records": True,
    }


def mcnemar_binary(
    y_true: Sequence[int],
    y_a: Sequence[int],
    y_b: Sequence[int],
) -> Dict[str, Any]:
    """
    Yates-corrected McNemar on paired binary predictions.

    b = A correct & B wrong; c = B correct & A wrong.
    This is a *binary* test. It is not a four-category comparison.
    """
    y_t = _as_int_array(y_true)
    ya = _as_int_array(y_a)
    yb = _as_int_array(y_b)
    a_ok = ya == y_t
    b_ok = yb == y_t
    b = int(np.sum(a_ok & (~b_ok)))
    c = int(np.sum(b_ok & (~a_ok)))
    if (b + c) > 0:
        chi2 = float((abs(b - c) - 1.0) ** 2 / (b + c))
        # chi2 survival without scipy dependency for unit tests
        try:
            from scipy import stats

            p_val = float(stats.chi2.sf(chi2, df=1))
        except Exception:
            p_val = float("nan")
    else:
        chi2 = 0.0
        p_val = 1.0
    return {
        "comparison_type": "binary_paired",
        "b_a_correct_b_wrong": b,
        "c_b_correct_a_wrong": c,
        "chi2_yates": chi2,
        "p_value": p_val,
        "significant_at_0_05": bool(p_val < 0.05) if p_val == p_val else False,
        "note": (
            "Non-significance is NOT proof of equivalence. "
            "This test compares binary labels only; it is not a four-category test."
        ),
    }
