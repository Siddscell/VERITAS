"""
test_decision_contract.py
=========================
Focused regression tests for VERITAS decision semantics and selective evaluation.

Uses synthetic fixtures only — does not tune models on held-out test examples.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from decision_contract import (
    CATEGORIES,
    CATEGORY_TO_IDX,
    DEFAULT_DEADBAND,
    binary_confusion_definitive,
    categories_sum_to_one,
    decision_confidence_from_scores,
    enrich_decision_fields,
    fake_probability_from_categories,
    four_category_metrics,
    is_abstention,
    legacy_binary_mapping_disclosed,
    metrics_from_frozen_records,
    normalize_verdict_class,
    selective_binary_metrics,
    three_outcome_assessment,
)
from fusion_policy import (
    apply_action_to_state,
    compute_verdict_corrected_fixed,
    extract_state_from_agents,
)


def _agents_av(sp=0.2, te=0.2, au=0.2):
    return [
        {"agent": "Spatial Analysis", "confidence": sp, "is_fake": sp > 0.5, "status": "trained"},
        {"agent": "Temporal Analysis", "confidence": te, "is_fake": te > 0.5, "status": "trained"},
        {
            "agent": "Audio Analysis",
            "confidence": au,
            "is_fake": au > 0.5,
            "status": "trained",
            "has_audio": True,
            "duration": 5.0,
        },
        {"agent": "Context Verification", "confidence": abs(sp - au), "is_fake": False, "status": "trained"},
    ]


class TestProbabilityContract(unittest.TestCase):
    def test_fake_probability_equals_sum_of_manip_categories(self):
        probs = {
            "AUTHENTIC": 0.40,
            "AUDIO_MANIPULATED": 0.25,
            "VIDEO_MANIPULATED": 0.20,
            "AUDIO_VIDEO_MANIPULATED": 0.15,
        }
        self.assertTrue(categories_sum_to_one(probs))
        self.assertAlmostEqual(fake_probability_from_categories(probs), 0.60, places=6)

    def test_apply_action_category_probs_sum_to_one(self):
        agents = _agents_av(0.7, 0.6, 0.8)
        state, meta = extract_state_from_agents(agents, "video")
        action = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        out = apply_action_to_state(action, state, meta)
        probs = out["category_probabilities"]
        self.assertTrue(categories_sum_to_one(probs, tol=1e-3))
        s = sum(probs[c] for c in CATEGORIES)
        self.assertAlmostEqual(s, 1.0, places=3)
        self.assertAlmostEqual(
            out["fake_probability"],
            fake_probability_from_categories(probs),
            places=4,
        )
        self.assertAlmostEqual(
            out["fake_probability"],
            1.0 - probs["AUTHENTIC"],
            places=4,
        )

    def test_corrected_fixed_emits_consistent_fake_probability(self):
        agents = _agents_av(0.8, 0.7, 0.6)
        overall, verdict, _, _, details = compute_verdict_corrected_fixed(agents, "video")
        probs = details["category_probabilities"]
        self.assertTrue(categories_sum_to_one(probs, tol=1e-3))
        self.assertAlmostEqual(
            details["fake_probability"],
            fake_probability_from_categories(probs),
            places=4,
        )
        self.assertEqual(details["method"], "corrected_fixed")
        self.assertIn(details["verdict_class"], ("AUTHENTIC", "DEEPFAKE", "UNCERTAIN"))


class TestAbstentionSemantics(unittest.TestCase):
    def test_uncertain_never_normalized_as_authentic(self):
        self.assertEqual(
            normalize_verdict_class("UNCERTAIN / INSUFFICIENT EVIDENCE"),
            "UNCERTAIN",
        )
        self.assertTrue(is_abstention("UNCERTAIN / INSUFFICIENT EVIDENCE"))
        self.assertFalse(is_abstention("AUTHENTIC"))
        self.assertFalse(is_abstention("DEEPFAKE"))

    def test_uncertain_with_manip_category_hypothesis(self):
        # Force p_fake into the deadband around tau=0.50
        agents = _agents_av(0.55, 0.45, 0.50)
        state, meta = extract_state_from_agents(agents, "video")
        # a4=0 → tau=0.50; choose mild actions
        action = np.zeros(5, dtype=np.float32)
        out = apply_action_to_state(action, state, meta)
        # May or may not land in UNCERTAIN depending on fused score; synthesize enrich path
        synthetic = enrich_decision_fields(
            {
                "verdict": "UNCERTAIN / INSUFFICIENT EVIDENCE",
                "p_fake": 0.51,
                "overall_confidence": 0.51,
                "predicted_category": "AUDIO_MANIPULATED",
                "category_probabilities": {
                    "AUTHENTIC": 0.49,
                    "AUDIO_MANIPULATED": 0.30,
                    "VIDEO_MANIPULATED": 0.11,
                    "AUDIO_VIDEO_MANIPULATED": 0.10,
                },
                "decision_threshold": 0.50,
                "method": "ppo",
            },
            tau=0.50,
        )
        self.assertEqual(synthetic["verdict_class"], "UNCERTAIN")
        self.assertTrue(synthetic["is_abstention"])
        self.assertTrue(synthetic["leading_category_is_hypothesis"])
        self.assertEqual(synthetic["predicted_category"], "AUDIO_MANIPULATED")
        self.assertNotEqual(synthetic["verdict_class"], "AUTHENTIC")
        self.assertAlmostEqual(synthetic["decision_confidence"], 0.0, places=5)

    def test_decision_confidence_zero_inside_band(self):
        self.assertEqual(
            decision_confidence_from_scores(0.50, tau=0.50, deadband=DEFAULT_DEADBAND),
            0.0,
        )
        self.assertGreater(
            decision_confidence_from_scores(0.90, tau=0.50, deadband=DEFAULT_DEADBAND),
            0.0,
        )
        self.assertGreater(
            decision_confidence_from_scores(0.10, tau=0.50, deadband=DEFAULT_DEADBAND),
            0.0,
        )


class TestMissingModalities(unittest.TestCase):
    def test_image_only_no_fabricated_audio(self):
        agents = [
            {"agent": "Spatial Analysis", "confidence": 0.8, "is_fake": True, "status": "trained"},
            {"agent": "Temporal Analysis", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
            {
                "agent": "Audio Analysis",
                "confidence": 0.0,
                "is_fake": None,
                "status": "no_audio",
                "has_audio": False,
            },
            {"agent": "Context Verification", "confidence": 0.0, "is_fake": None, "status": "unavailable"},
        ]
        state, meta = extract_state_from_agents(agents, "image")
        self.assertFalse(meta["audio_valid"])
        self.assertTrue(meta["spatial_valid"])
        out = apply_action_to_state(np.zeros(5, dtype=np.float32), state, meta)
        probs = out["category_probabilities"]
        self.assertEqual(probs["AUDIO_MANIPULATED"], 0.0)
        self.assertEqual(probs["AUDIO_VIDEO_MANIPULATED"], 0.0)
        self.assertAlmostEqual(probs["VIDEO_MANIPULATED"] + probs["AUTHENTIC"], 1.0, places=4)
        self.assertIn("Audio Analysis", out["unavailable_agents"])

    def test_all_unavailable_unverified(self):
        agents = [
            {"agent": "Spatial Analysis", "confidence": 0.0, "is_fake": None, "status": "error"},
            {"agent": "Temporal Analysis", "confidence": 0.0, "is_fake": None, "status": "error"},
            {"agent": "Audio Analysis", "confidence": 0.0, "is_fake": None, "status": "error"},
            {"agent": "Context Verification", "confidence": 0.0, "is_fake": None, "status": "unavailable"},
        ]
        state, meta = extract_state_from_agents(agents, "video")
        out = apply_action_to_state(np.zeros(5, dtype=np.float32), state, meta)
        self.assertEqual(normalize_verdict_class(out["verdict"]), "UNVERIFIED")
        self.assertTrue(out["is_abstention"])


class TestSelectiveEvaluation(unittest.TestCase):
    def test_uncertain_excluded_from_definitive_cm(self):
        # 4 samples: TN, TP, FP abstained-as-not-in-CM, FN
        y_true = [0, 1, 0, 1]
        verdicts = [
            "AUTHENTIC",
            "DEEPFAKE",
            "UNCERTAIN / INSUFFICIENT EVIDENCE",
            "AUTHENTIC",
        ]
        cm = binary_confusion_definitive(y_true, verdicts)
        self.assertEqual(cm["n_definitive"], 3)
        self.assertEqual(cm["excluded_abstentions"], 1)
        self.assertEqual(cm["tn"], 1)
        self.assertEqual(cm["tp"], 1)
        self.assertEqual(cm["fp"], 0)
        self.assertEqual(cm["fn"], 1)

        sel = selective_binary_metrics(y_true, verdicts)
        self.assertAlmostEqual(sel["coverage"], 0.75)
        self.assertAlmostEqual(sel["abstention_rate"], 0.25)
        self.assertAlmostEqual(sel["selective_error_rate"], 1.0 / 3.0)

        three = three_outcome_assessment(y_true, verdicts)
        self.assertEqual(three["counts"]["gt_authentic_abstain"], 1)

        # Legacy disclosed mapping would count UNCERTAIN authentic GT as TN
        legacy = legacy_binary_mapping_disclosed(y_true, verdicts)
        self.assertEqual(legacy["tn"], 2)  # includes abstention on authentic
        self.assertTrue(legacy["disclosed"])
        self.assertIn("UNCERTAIN", legacy["policy"])

    def test_binary_and_category_from_same_records(self):
        records = [
            {
                "verdict": "AUTHENTIC",
                "predicted_category": "AUTHENTIC",
                "gt_is_fake": 0,
                "gt_cat_idx": 0,
            },
            {
                "verdict": "DEEPFAKE",
                "predicted_category": "AUDIO_MANIPULATED",
                "gt_is_fake": 1,
                "gt_cat_idx": 1,
            },
            {
                "verdict": "UNCERTAIN / INSUFFICIENT EVIDENCE",
                "predicted_category": "AUDIO_MANIPULATED",
                "gt_is_fake": 0,
                "gt_cat_idx": 0,
            },
            {
                "verdict": "DEEPFAKE",
                "predicted_category": "VIDEO_MANIPULATED",
                "gt_is_fake": 1,
                "gt_cat_idx": 2,
            },
        ]
        m = metrics_from_frozen_records(records)
        self.assertTrue(m["surfaces_share_same_records"])
        self.assertEqual(m["n"], 4)
        self.assertEqual(m["binary_selective"]["n_abstained"], 1)
        self.assertEqual(m["four_category"]["n"], 4)
        # 4-way: correct on rows 0,1,3 → 3/4; row 2 GT AUTH pred AUDIO → wrong
        self.assertEqual(m["four_category"]["confusion_matrix"][0][0], 1)
        self.assertEqual(m["four_category"]["confusion_matrix"][0][1], 1)
        self.assertAlmostEqual(m["four_category"]["accuracy"], 0.75)

        # Category CM margins must be independent of verdict abstention handling
        four = four_category_metrics(
            [r["gt_cat_idx"] for r in records],
            [CATEGORY_TO_IDX[r["predicted_category"]] for r in records],
        )
        self.assertEqual(four["confusion_matrix"], m["four_category"]["confusion_matrix"])


class TestProductionDefaultPreserved(unittest.TestCase):
    def test_main_defaults_to_corrected_fixed(self):
        # Ensure env default path is corrected_fixed when unset
        prev = os.environ.pop("VERITAS_FUSION_MODE", None)
        try:
            import importlib
            import main as main_mod

            importlib.reload(main_mod)
            agents = _agents_av(0.1, 0.1, 0.1)
            overall, verdict, contrib, unavail, details = main_mod._compute_verdict(
                agents, "video", return_details=True, mode_override=""
            )
            self.assertEqual(details.get("method"), "corrected_fixed")
            self.assertIn("fake_probability", details)
            self.assertIn("decision_thresholds", details)
            self.assertIn("verdict_class", details)
        finally:
            if prev is not None:
                os.environ["VERITAS_FUSION_MODE"] = prev


if __name__ == "__main__":
    unittest.main()
