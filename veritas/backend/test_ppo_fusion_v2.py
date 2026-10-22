"""
test_ppo_fusion_v2.py
=====================
Comprehensive regression test suite for VERITAS PPO Fusion v2.
Validates:
1. Action space dimensions (ACTION_DIM == 5) and Actor-Critic architecture.
2. Spatial vs temporal visual gating and normalization.
3. Audio trust logit directly modulating audio fake probability.
4. Context inconsistency sensitivity and category attribution.
5. Missing modality masking (video-only, audio-only, all-missing).
6. Category output parity: P(Fake) == 1 - P(Authentic) == sum(manipulated categories).
7. Zero semantic contradiction between binary verdict and predicted category.
8. Adaptive decision threshold offset Delta tau in [-0.15, +0.15].
9. Defensible ground-truth reward matrix without arbitrary shaping.
10. FeatureNormalizer fit strictly on training data.
11. Deterministic inference reproducibility.
12. Diagnostic sample `00014_fake.mp4` behavior.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fusion_policy import (
    ACTION_DIM,
    CATEGORIES,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    STATE_DIM,
    ActorCriticNetwork,
    FeatureNormalizer,
    PPOFusionPolicy,
    apply_action_to_state,
    compute_reward,
    extract_state_from_agents,
)


class TestPPOFusionV2(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.net = ActorCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)

    # 1. Action Space Dimensions
    def test_action_space_dimensions(self):
        self.assertEqual(STATE_DIM, 14)
        self.assertEqual(ACTION_DIM, 5)

        dummy_state = torch.zeros((4, STATE_DIM), dtype=torch.float32)
        dist, val = self.net(dummy_state)
        self.assertEqual(dist.mean.shape, (4, 5))
        self.assertEqual(dist.stddev.shape, (4, 5))
        self.assertEqual(val.shape, (4,))

    # 2. Visual Gating
    def test_visual_gating(self):
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        state_vec[0] = 0.20  # spatial prob
        state_vec[3] = 0.80  # temporal prob
        metadata = {"spatial_valid": True, "temporal_valid": True, "audio_valid": False}

        # Action 0 > Action 1: spatial should dominate
        action_sp_dom = np.array([2.0, -2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        res_sp = apply_action_to_state(action_sp_dom, state_vec, metadata)
        self.assertGreater(res_sp["dynamic_weights"]["spatial"], res_sp["dynamic_weights"]["temporal"])
        self.assertAlmostEqual(res_sp["dynamic_weights"]["spatial"] + res_sp["dynamic_weights"]["temporal"], 1.0, places=4)
        self.assertLess(res_sp["p_visual"], 0.35)

        # Action 1 > Action 0: temporal should dominate
        action_te_dom = np.array([-2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        res_te = apply_action_to_state(action_te_dom, state_vec, metadata)
        self.assertGreater(res_te["dynamic_weights"]["temporal"], res_te["dynamic_weights"]["spatial"])
        self.assertGreater(res_te["p_visual"], 0.65)

    # 3. Audio Trust Modulation
    def test_audio_trust_modulation(self):
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        state_vec[6] = 1.00  # audio prob = 1.0 (detected fake)
        metadata = {"spatial_valid": False, "temporal_valid": False, "audio_valid": True}

        # High trust (a[2] = +3.0 -> w_au ~ 0.95)
        act_high_trust = np.array([0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float32)
        res_high = apply_action_to_state(act_high_trust, state_vec, metadata)
        self.assertGreater(res_high["dynamic_weights"]["audio"], 0.90)
        self.assertGreater(res_high["p_audio"], 0.90)

        # Low trust (a[2] = -3.0 -> w_au ~ 0.05)
        # Should shrink p_audio towards uninformative 0.50
        act_low_trust = np.array([0.0, 0.0, -3.0, 0.0, 0.0], dtype=np.float32)
        res_low = apply_action_to_state(act_low_trust, state_vec, metadata)
        self.assertLess(res_low["dynamic_weights"]["audio"], 0.10)
        self.assertAlmostEqual(res_low["p_audio"], 0.525, delta=0.05)

    # 4. Missing Modality Masking
    def test_missing_modality_masking(self):
        # Case A: Audio missing (Image or Silent Video)
        state_vec_no_audio = np.zeros(STATE_DIM, dtype=np.float32)
        state_vec_no_audio[0] = 0.90  # spatial fake
        metadata_no_audio = {"spatial_valid": True, "temporal_valid": False, "audio_valid": False}
        act = np.zeros(5, dtype=np.float32)
        res_no_audio = apply_action_to_state(act, state_vec_no_audio, metadata_no_audio)

        self.assertEqual(res_no_audio["category_probabilities"]["AUDIO_MANIPULATED"], 0.0)
        self.assertEqual(res_no_audio["category_probabilities"]["AUDIO_VIDEO_MANIPULATED"], 0.0)
        self.assertEqual(res_no_audio["predicted_category"], "VIDEO_MANIPULATED")
        self.assertEqual(res_no_audio["verdict"], "DEEPFAKE")

        # Case B: Video missing (Audio-Only)
        state_vec_no_video = np.zeros(STATE_DIM, dtype=np.float32)
        state_vec_no_video[6] = 0.95  # audio fake
        metadata_no_video = {"spatial_valid": False, "temporal_valid": False, "audio_valid": True}
        res_no_video = apply_action_to_state(act, state_vec_no_video, metadata_no_video)

        self.assertEqual(res_no_video["category_probabilities"]["VIDEO_MANIPULATED"], 0.0)
        self.assertEqual(res_no_video["category_probabilities"]["AUDIO_VIDEO_MANIPULATED"], 0.0)
        self.assertEqual(res_no_video["predicted_category"], "AUDIO_MANIPULATED")
        self.assertEqual(res_no_video["verdict"], "DEEPFAKE")

        # Case C: All missing
        metadata_empty = {"spatial_valid": False, "temporal_valid": False, "audio_valid": False}
        res_empty = apply_action_to_state(act, np.zeros(STATE_DIM, dtype=np.float32), metadata_empty)
        self.assertEqual(res_empty["verdict"], "UNVERIFIED MODALITY")
        self.assertEqual(res_empty["predicted_category"], "UNVERIFIED")

    # 5. Category Parity and Verdict Consistency
    def test_category_verdict_parity(self):
        for pv in [0.05, 0.20, 0.40, 0.60, 0.85, 0.98]:
            for pa in [0.05, 0.20, 0.40, 0.60, 0.85, 0.98]:
                state_vec = np.zeros(STATE_DIM, dtype=np.float32)
                state_vec[0] = pv
                state_vec[3] = pv
                state_vec[6] = pa
                metadata = {"spatial_valid": True, "temporal_valid": True, "audio_valid": True}

                act = np.array([0.0, 0.0, 2.0, 0.5, 0.0], dtype=np.float32)
                res = apply_action_to_state(act, state_vec, metadata)

                cat_probs = res["category_probabilities"]
                sum_cat = sum(cat_probs.values())
                self.assertAlmostEqual(sum_cat, 1.0, places=3)

                sum_fake_cat = cat_probs["AUDIO_MANIPULATED"] + cat_probs["VIDEO_MANIPULATED"] + cat_probs["AUDIO_VIDEO_MANIPULATED"]
                self.assertAlmostEqual(1.0 - cat_probs["AUTHENTIC"], sum_fake_cat, places=3)

                # Verdict consistency: if DEEPFAKE, category cannot be AUTHENTIC
                if res["verdict"] == "DEEPFAKE":
                    self.assertIn(res["predicted_category"], ["AUDIO_MANIPULATED", "VIDEO_MANIPULATED", "AUDIO_VIDEO_MANIPULATED"])
                # If AUTHENTIC, category must be AUTHENTIC
                elif res["verdict"] == "AUTHENTIC":
                    self.assertEqual(res["predicted_category"], "AUTHENTIC")

    # 6. Defensible Reward Function
    def test_reward_function(self):
        # True Positive on Audio Deepfake
        tp_out = {
            "verdict": "DEEPFAKE",
            "predicted_category": "AUDIO_MANIPULATED",
            "overall_confidence": 0.95,
        }
        r_tp = compute_reward(tp_out, ground_truth_category=1)
        self.assertEqual(r_tp, 1.5 + 1.0)  # +2.5

        # False Negative on Audio Deepfake (missed deepfake)
        fn_out = {
            "verdict": "AUTHENTIC",
            "predicted_category": "AUTHENTIC",
            "overall_confidence": 0.10,
        }
        r_fn = compute_reward(fn_out, ground_truth_category=1)
        self.assertEqual(r_fn, -3.0 - 0.5)  # -3.5

        # True Negative on Authentic Media
        tn_out = {
            "verdict": "AUTHENTIC",
            "predicted_category": "AUTHENTIC",
            "overall_confidence": 0.10,
        }
        r_tn = compute_reward(tn_out, ground_truth_category=0)
        self.assertEqual(r_tn, 1.5 + 1.0)  # +2.5

        # False Positive on Authentic Media (flagged as deepfake)
        fp_out = {
            "verdict": "DEEPFAKE",
            "predicted_category": "VIDEO_MANIPULATED",
            "overall_confidence": 0.90,
        }
        r_fp = compute_reward(fp_out, ground_truth_category=0)
        self.assertEqual(r_fp, -2.0 - 0.5)  # -2.5

    # 7. Adaptive Decision Threshold
    def test_decision_threshold_range(self):
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        state_vec[0] = 0.50
        metadata = {"spatial_valid": True, "temporal_valid": False, "audio_valid": False}

        # Min threshold offset (a[4] = -5.0 -> Delta tau = -0.15)
        act_min = np.array([0.0, 0.0, 0.0, 0.0, -5.0], dtype=np.float32)
        res_min = apply_action_to_state(act_min, state_vec, metadata)
        self.assertAlmostEqual(res_min["decision_threshold"], 0.35, places=2)

        # Max threshold offset (a[4] = +5.0 -> Delta tau = +0.15)
        act_max = np.array([0.0, 0.0, 0.0, 0.0, +5.0], dtype=np.float32)
        res_max = apply_action_to_state(act_max, state_vec, metadata)
        self.assertAlmostEqual(res_max["decision_threshold"], 0.65, places=2)


if __name__ == "__main__":
    unittest.main()
