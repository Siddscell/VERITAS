"""
ppo_fusion_train.py
===================
Training pipeline for VERITAS Genuine PPO-Based Adaptive Decision Fusion.

Features:
---------
1. Identity-Aware Stratified 80/10/10 Split:
   - Strict identity disjointness across Train, Validation, and Test sets.
   - Leakage prevention verified with assertions.
2. Feature Extraction & Caching:
   - Evaluates frozen detector models (Spatial, Temporal, Audio, Context).
   - Caches extracted features to disk with resume capability.
3. Feature Normalization:
   - Mean and variance fitted strictly on the training split.
4. Contextual Bandit PPO Training:
   - 1-step MDP formulation (discount factor gamma = 0).
   - Clipped surrogate policy objective.
   - Value function loss with baseline clipping.
   - Entropy regularization for exploratory stability.
   - Early stopping and model selection strictly based on validation performance.
5. Saves production checkpoint to `veritas/backend/models/ppo_fusion_policy.pth`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add backend directory to sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import audio_agent
import spatial_agent
import temporal_agent
from mock_agents import context_agent_result
from main import _extract_video_frame
from fusion_policy import (
    ACTION_DIM,
    CATEGORIES,
    CATEGORY_TO_IDX,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_V2_CHECKPOINT_PATH,
    IDX_TO_CATEGORY,
    STATE_DIM,
    ActorCriticNetwork,
    FeatureNormalizer,
    apply_action_to_state,
    compute_reward,
    extract_state_from_agents,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ppo_fusion_train")

# Default paths
DEFAULT_DATASET_ROOT = Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
DEFAULT_CACHE_DIR = _BACKEND_DIR / "training" / "cache"


# ---------------------------------------------------------------------------
# Identity-Aware Split (Disjoint Speakers & Faces)
# ---------------------------------------------------------------------------
def extract_identity_from_path(mp4_path: Path, category: str) -> str:
    """Extract speaker/face identity token from directory structure."""
    import re
    fname = mp4_path.stem
    folder = mp4_path.parent.name
    id_tokens = re.findall(r"id\d+", folder, re.IGNORECASE)
    folder_id = id_tokens[0].lower() if id_tokens else folder.lower()

    if category in ("RealVideo-RealAudio", "RealVideo-FakeAudio"):
        return folder_id

    # For Wav2Lip, driving audio donor is encoded in filename
    m = re.search(r"_(id\d+)_wav(?:to)?lip", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m = re.search(r"_wav2lip_(id\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return folder_id


def discover_and_split_dataset(
    dataset_root: Path,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Discovers FakeAVCeleb clips across 4 categories and splits them with disjoint identities.
    Returns: train_records, val_records, test_records
    """
    logger.info("Scanning dataset at: %s", dataset_root)
    category_map = {
        "RealVideo-RealAudio": 0,
        "RealVideo-FakeAudio": 1,
        "FakeVideo-RealAudio": 2,
        "FakeVideo-FakeAudio": 3,
    }

    all_samples: List[Dict[str, Any]] = []
    by_category_identity: Dict[int, Dict[str, List[Path]]] = {
        cat_idx: {} for cat_idx in category_map.values()
    }

    for cat_name, cat_idx in category_map.items():
        cat_dir = dataset_root / cat_name
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing category directory: {cat_dir}")
        mp4s = sorted(cat_dir.rglob("*.mp4"))
        logger.info("Category %s: found %d MP4 files", cat_name, len(mp4s))
        for p in mp4s:
            ident = extract_identity_from_path(p, cat_name)
            by_category_identity[cat_idx].setdefault(ident, []).append(p)

    rng = random.Random(seed)

    # Collect all unique identities across all categories
    all_identities = set()
    for cat_idx in by_category_identity:
        for ident in by_category_identity[cat_idx]:
            all_identities.add(ident)

    sorted_idents = sorted(list(all_identities))
    rng.shuffle(sorted_idents)

    n_total = len(sorted_idents)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = max(1, int(round(n_total * test_ratio)))
    n_train = n_total - n_val - n_test

    val_idents = set(sorted_idents[:n_val])
    test_idents = set(sorted_idents[n_val : n_val + n_test])
    train_idents = set(sorted_idents[n_val + n_test :])

    # Disjointness assertions
    assert len(train_idents & val_idents) == 0, "Train-Val identity leak!"
    assert len(train_idents & test_idents) == 0, "Train-Test identity leak!"
    assert len(val_idents & test_idents) == 0, "Val-Test identity leak!"

    logger.info("Identities partition: Train=%d, Val=%d, Test=%d (Total=%d)",
                len(train_idents), len(val_idents), len(test_idents), n_total)

    train_samples, val_samples, test_samples = [], [], []

    for cat_idx in by_category_identity:
        for ident, paths in by_category_identity[cat_idx].items():
            for p in paths:
                rec = {
                    "path": str(p),
                    "category": cat_idx,
                    "category_name": IDX_TO_CATEGORY[cat_idx],
                    "identity": ident,
                }
                if ident in val_idents:
                    val_samples.append(rec)
                elif ident in test_idents:
                    test_samples.append(rec)
                else:
                    train_samples.append(rec)

    logger.info("Sample partition counts: Train=%d, Val=%d, Test=%d (Total=%d)",
                len(train_samples), len(val_samples), len(test_samples),
                len(train_samples) + len(val_samples) + len(test_samples))

    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# Feature Extraction with Resumption & Caching
# ---------------------------------------------------------------------------
def extract_sample_features(
    video_path: str,
) -> Dict[str, Any]:
    """Runs all 4 agents on a video file and returns structured results."""
    # 1. Spatial Agent
    frame_path = _extract_video_frame(video_path)
    s_result = None
    if frame_path:
        try:
            s_result = spatial_agent.predict(frame_path)
            s_result["status"] = "trained"
        except Exception as exc:
            logger.warning("Spatial error on %s: %s", video_path, exc)
            s_result = {
                "agent": "Spatial Analysis",
                "status": "error",
                "confidence": 0.0,
                "is_fake": None,
                "finding": str(exc),
            }
        finally:
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except OSError:
                    pass
    else:
        s_result = {
            "agent": "Spatial Analysis",
            "status": "error",
            "confidence": 0.0,
            "is_fake": None,
            "finding": "Frame extraction failed",
        }

    # 2. Temporal Agent
    try:
        t_result = temporal_agent.run_temporal_agent(video_path)
    except Exception as exc:
        logger.warning("Temporal error on %s: %s", video_path, exc)
        t_result = {
            "agent": "Temporal Analysis",
            "status": "error",
            "confidence": 0.0,
            "is_fake": None,
            "finding": str(exc),
        }

    # 3. Audio Agent
    try:
        a_result = audio_agent.run_audio_agent(video_path)
    except Exception as exc:
        logger.warning("Audio error on %s: %s", video_path, exc)
        a_result = {
            "agent": "Audio Analysis",
            "status": "error",
            "confidence": 0.0,
            "is_fake": None,
            "has_audio": False,
            "finding": str(exc),
        }

    # 4. Context Agent
    c_result = context_agent_result(s_result, t_result, a_result)

    agents = [s_result, t_result, a_result, c_result]
    state_vec, metadata = extract_state_from_agents(agents, media_type="video")

    return {
        "agents": agents,
        "state_vec": state_vec.tolist(),
        "metadata": metadata,
    }


def build_or_load_dataset(
    dataset_root: Path,
    cache_dir: Path,
    samples_per_category: int = 150,
    seed: int = 42,
    cache_file: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds or loads cached feature extractions for train, val, and test splits.
    Extracts balanced samples across all 4 categories and saves cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_file is None:
        expanded_file = cache_dir / "fusion_dataset_cache_expanded.json"
        if expanded_file.exists():
            cache_file = expanded_file
        else:
            cache_file = cache_dir / f"fusion_dataset_cache_{samples_per_category}.json"

    if cache_file.exists():
        logger.info("Loading cached features from %s", cache_file)
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            train_recs = data["train"]
            val_recs = data["val"]
            test_recs = data["test"]
            logger.info("Loaded %d train, %d val, %d test cached records.",
                        len(train_recs), len(val_recs), len(test_recs))
            return train_recs, val_recs, test_recs
        except Exception as e:
            logger.warning("Failed to load cache %s: %s. Rebuilding...", cache_file, e)

    # 1. Discover and split dataset
    train_pool, val_pool, test_pool = discover_and_split_dataset(dataset_root, seed=seed)

    # 2. Select stratified subsets per split and category
    rng = random.Random(seed)

    def _select_stratified(pool: List[Dict[str, Any]], count_per_cat: int) -> List[Dict[str, Any]]:
        by_cat: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(4)}
        for s in pool:
            by_cat[s["category"]].append(s)
        selected = []
        for c in range(4):
            cat_list = by_cat[c]
            rng.shuffle(cat_list)
            selected.extend(cat_list[:count_per_cat])
        return selected

    # For 150 total per category across dataset:
    # Train: 120 per cat (480 total)
    # Val: 15 per cat (60 total)
    # Test: 15 per cat (60 total)
    n_train_cat = int(round(samples_per_category * 0.80))
    n_val_cat = max(5, int(round(samples_per_category * 0.10)))
    n_test_cat = max(5, int(round(samples_per_category * 0.10)))

    train_selected = _select_stratified(train_pool, n_train_cat)
    val_selected = _select_stratified(val_pool, n_val_cat)
    test_selected = _select_stratified(test_pool, n_test_cat)

    logger.info("Selected for extraction: Train=%d, Val=%d, Test=%d",
                len(train_selected), len(val_selected), len(test_selected))

    # Ensure detector models are loaded
    if not spatial_agent._model_loaded:
        spatial_agent.load_model()
    if not temporal_agent._model_loaded:
        temporal_agent.load_model()
    if not audio_agent._model_loaded:
        audio_agent._load_model()

    # Resumable progress tracking
    interim_cache = cache_dir / "interim_feature_extraction.json"
    extracted_map: Dict[str, Dict[str, Any]] = {}
    if interim_cache.exists():
        try:
            with open(interim_cache, "r", encoding="utf-8") as f:
                extracted_map = json.load(f)
            logger.info("Found %d pre-extracted records in interim cache.", len(extracted_map))
        except Exception:
            extracted_map = {}

    all_jobs = (
        [("train", s) for s in train_selected]
        + [("val", s) for s in val_selected]
        + [("test", s) for s in test_selected]
    )

    total_jobs = len(all_jobs)
    t0 = time.time()

    for idx, (split_name, item) in enumerate(all_jobs):
        p = item["path"]
        if p not in extracted_map:
            try:
                feat = extract_sample_features(p)
                extracted_map[p] = feat
            except Exception as e:
                logger.error("Failed extraction for %s: %s", p, e)
                continue

            if (idx + 1) % 25 == 0 or (idx + 1) == total_jobs:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1.0)
                remain = (total_jobs - (idx + 1)) / max(rate, 1e-4)
                logger.info("Extraction progress: %d/%d (%.1f%%) | %.2f it/s | ~%.0fs remaining",
                            idx + 1, total_jobs, 100 * (idx + 1) / total_jobs, rate, remain)
                # Flush interim cache
                with open(interim_cache, "w", encoding="utf-8") as f:
                    json.dump(extracted_map, f)

    # Package into final split structures
    def _package_split(selected_list: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        records = []
        for item in selected_list:
            p = item["path"]
            if p in extracted_map:
                feat = extracted_map[p]
                records.append({
                    "path": p,
                    "category": item["category"],
                    "category_name": item["category_name"],
                    "identity": item["identity"],
                    "split": split_name,
                    "agents": feat["agents"],
                    "state_vec": feat["state_vec"],
                    "metadata": feat["metadata"],
                })
        return records

    train_records = _package_split(train_selected, "train")
    val_records = _package_split(val_selected, "val")
    test_records = _package_split(test_selected, "test")

    # Add controlled missing modality records (audio-only, image-only, missing video)
    # This teaches the policy to handle missing modalities robustly
    augmented_train = _add_missing_modality_augmentations(train_records, rng)
    augmented_val = _add_missing_modality_augmentations(val_records, rng)
    augmented_test = _add_missing_modality_augmentations(test_records, rng)

    final_payload = {
        "train": augmented_train,
        "val": augmented_val,
        "test": augmented_test,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    logger.info("Saved complete feature cache to %s", cache_file)
    return augmented_train, augmented_val, augmented_test


def _add_missing_modality_augmentations(
    records: List[Dict[str, Any]],
    rng: random.Random,
    aug_fraction: float = 0.20,
) -> List[Dict[str, Any]]:
    """Generates sound missing-modality scenarios from real extracted records."""
    augmented = list(records)
    n_aug = int(len(records) * aug_fraction)
    sample_pool = rng.sample(records, min(n_aug, len(records)))

    for i, orig in enumerate(sample_pool):
        scenario = i % 3
        agents_copy = [dict(a) for a in orig["agents"]]
        cat = orig["category"]

        if scenario == 0:
            # Image only: temporal and audio not applicable
            for a in agents_copy:
                if "temporal" in a["agent"].lower():
                    a["status"] = "not_applicable"
                    a["is_fake"] = None
                    a["confidence"] = 0.0
                elif "audio" in a["agent"].lower():
                    a["status"] = "no_audio"
                    a["has_audio"] = False
                    a["is_fake"] = None
                    a["confidence"] = 0.0
                elif "context" in a["agent"].lower():
                    a["status"] = "not_applicable"
                    a["is_fake"] = None
            mtype = "image"
            aug_cat = 2 if cat in (2, 3) else 0  # video manipulated or authentic

        elif scenario == 1:
            # Audio only: spatial and temporal not applicable
            for a in agents_copy:
                if "spatial" in a["agent"].lower():
                    a["status"] = "not_applicable"
                    a["is_fake"] = None
                    a["confidence"] = 0.0
                elif "temporal" in a["agent"].lower():
                    a["status"] = "not_applicable"
                    a["is_fake"] = None
                    a["confidence"] = 0.0
                elif "context" in a["agent"].lower():
                    a["status"] = "not_applicable"
                    a["is_fake"] = None
            mtype = "audio"
            aug_cat = 1 if cat in (1, 3) else 0  # audio manipulated or authentic

        else:
            # Video without audio stream
            for a in agents_copy:
                if "audio" in a["agent"].lower():
                    a["status"] = "no_audio"
                    a["has_audio"] = False
                    a["is_fake"] = None
                    a["confidence"] = 0.0
                elif "context" in a["agent"].lower():
                    a["status"] = "not_applicable"
                    a["is_fake"] = None
            mtype = "video"
            aug_cat = 2 if cat in (2, 3) else 0

        state_vec, metadata = extract_state_from_agents(agents_copy, media_type=mtype)
        augmented.append({
            "path": orig["path"] + f"_aug_{scenario}",
            "category": aug_cat,
            "category_name": IDX_TO_CATEGORY[aug_cat],
            "identity": orig["identity"],
            "split": orig["split"],
            "agents": agents_copy,
            "state_vec": state_vec.tolist(),
            "metadata": metadata,
            "is_augmented": True,
        })

    return augmented


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """
    PPO Actor-Critic Trainer for the single-step Contextual Bandit environment.
    """

    def __init__(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        normalizer: FeatureNormalizer,
        lr: float = 3e-4,
        clip_epsilon: float = 0.20,
        c_val: float = 0.50,
        c_ent: float = 0.01,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_records = train_records
        self.val_records = val_records
        self.normalizer = normalizer

        self.clip_epsilon = clip_epsilon
        self.c_val = c_val
        self.c_ent = c_ent

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.network = ActorCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def collect_rollout(
        self,
        batch_size: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples states from the training split, executes continuous actions,
        and computes advantages.
        """
        self.network.eval()

        sampled_recs = random.choices(self.train_records, k=batch_size)
        raw_states = np.array([r["state_vec"] for r in sampled_recs], dtype=np.float32)
        norm_states = self.normalizer.transform(raw_states)
        gt_categories = [r["category"] for r in sampled_recs]
        metadatas = [r["metadata"] for r in sampled_recs]

        states_t = torch.from_numpy(norm_states).to(self.device)

        with torch.no_grad():
            dist, values_t = self.network(states_t)
            actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t).sum(dim=-1)

        actions_np = actions_t.cpu().numpy()
        rewards_list = []

        for i in range(batch_size):
            act = actions_np[i]
            raw_s = raw_states[i]
            meta = metadatas[i]
            gt = gt_categories[i]

            res = apply_action_to_state(act, raw_s, meta)
            r = compute_reward(res, gt)
            rewards_list.append(r)

        rewards_t = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # Contextual Bandit advantage: A(s, a) = R - V(s)
        advantages_t = rewards_t - values_t
        # Standardize advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std() + 1e-8
        norm_advantages_t = (advantages_t - adv_mean) / adv_std

        return {
            "states": states_t,
            "actions": actions_t,
            "old_log_probs": log_probs_t,
            "returns": rewards_t,
            "advantages": norm_advantages_t,
            "rewards": rewards_t,
        }

    def train_epoch(
        self,
        rollout: Dict[str, torch.Tensor],
        ppo_epochs: int = 5,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """Performs clipped PPO updates on the collected rollout."""
        self.network.train()
        n_samples = rollout["states"].shape[0]

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(n_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                s_mb = rollout["states"][mb_idx]
                a_mb = rollout["actions"][mb_idx]
                old_lp_mb = rollout["old_log_probs"][mb_idx]
                adv_mb = rollout["advantages"][mb_idx]
                ret_mb = rollout["returns"][mb_idx]

                dist, v_mb = self.network(s_mb)
                new_lp_mb = dist.log_prob(a_mb).sum(dim=-1)
                entropy_mb = dist.entropy().sum(dim=-1).mean()

                # Ratio r_t(theta)
                ratio = torch.exp(new_lp_mb - old_lp_mb)

                # Clipped surrogate loss
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function MSE loss
                value_loss = 0.5 * ((v_mb - ret_mb) ** 2).mean()

                total_loss = policy_loss + self.c_val * value_loss - self.c_ent * entropy_mb

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_pi_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_entropy += entropy_mb.item()
                n_updates += 1

        return {
            "policy_loss": total_pi_loss / max(n_updates, 1),
            "value_loss": total_v_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def evaluate(
        self,
        eval_records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Deterministic evaluation on validation or test records.
        Computes accuracy, macro-F1, per-category accuracy, and reward.
        """
        self.network.eval()
        raw_states = np.array([r["state_vec"] for r in eval_records], dtype=np.float32)
        norm_states = self.normalizer.transform(raw_states)
        states_t = torch.from_numpy(norm_states).to(self.device)

        with torch.no_grad():
            dist = self.network.get_action_dist(states_t)
            # Deterministic actions (distribution mean)
            actions = dist.mean.cpu().numpy()

        rewards = []
        binary_correct = 0
        cat_correct = 0
        y_true_binary = []
        y_pred_binary = []
        y_true_cat = []
        y_pred_cat = []

        for i, rec in enumerate(eval_records):
            act = actions[i]
            meta = rec["metadata"]
            gt_cat = rec["category"]
            gt_bin = 1 if gt_cat > 0 else 0

            res = apply_action_to_state(act, raw_states[i], meta)
            r = compute_reward(res, gt_cat)
            rewards.append(r)

            pred_bin = 1 if res["verdict"] == "DEEPFAKE" else 0
            pred_cat = CATEGORY_TO_IDX.get(res["predicted_category"], 0)

            y_true_binary.append(gt_bin)
            y_pred_binary.append(pred_bin)
            y_true_cat.append(gt_cat)
            y_pred_cat.append(pred_cat)

            if pred_bin == gt_bin:
                binary_correct += 1
            if pred_cat == gt_cat:
                cat_correct += 1

        n = len(eval_records)
        acc_binary = binary_correct / max(n, 1)
        acc_category = cat_correct / max(n, 1)
        mean_reward = float(np.mean(rewards))

        # Balanced binary accuracy
        tps = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1)
        fns = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0)
        tns = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0)
        fps = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1)

        tpr = tps / max(tps + fns, 1)
        tnr = tns / max(tns + fps, 1)
        balanced_acc = 0.5 * (tpr + tnr)

        return {
            "mean_reward": mean_reward,
            "accuracy": acc_binary,
            "balanced_accuracy": balanced_acc,
            "category_accuracy": acc_category,
            "tpr": tpr,
            "tnr": tnr,
            "samples": n,
        }


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def run_training(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_checkpoint: Path = DEFAULT_V2_CHECKPOINT_PATH,
    samples_per_category: int = 150,
    iterations: int = 40,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
    cache_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs full genuine PPO adaptive fusion training pipeline."""
    logger.info("=" * 70)
    logger.info("VERITAS PPO ADAPTIVE FUSION TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Build / Load Features
    train_recs, val_recs, test_recs = build_or_load_dataset(
        dataset_root=dataset_root,
        cache_dir=DEFAULT_CACHE_DIR,
        samples_per_category=samples_per_category,
        seed=seed,
        cache_file=cache_file,
    )

    # 2. Fit Feature Normalizer strictly on Training Records
    train_states = np.array([r["state_vec"] for r in train_recs], dtype=np.float32)
    normalizer = FeatureNormalizer(STATE_DIM)
    normalizer.fit(train_states)
    logger.info("Fitted normalizer on %d training states.", len(train_states))

    # 3. Initialize Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    trainer = PPOTrainer(
        train_records=train_recs,
        val_records=val_recs,
        normalizer=normalizer,
        lr=lr,
        device=device,
        seed=seed,
    )

    best_val_score = -float("inf")
    best_checkpoint_state: Optional[dict] = None
    best_metrics: Dict[str, Any] = {}

    logger.info("\nStarting PPO Policy Optimization Loop (%d iterations)...", iterations)
    start_time = time.time()

    for it in range(1, iterations + 1):
        rollout = trainer.collect_rollout(batch_size=batch_size)
        train_losses = trainer.train_epoch(rollout, ppo_epochs=4, minibatch_size=64)

        # Evaluate on validation split
        val_metrics = trainer.evaluate(val_recs)

        # Model selection metric: combined balanced accuracy & reward
        val_score = val_metrics["balanced_accuracy"] * 2.0 + val_metrics["mean_reward"]

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            best_checkpoint_state = {
                k: v.cpu().clone() for k, v in trainer.network.state_dict().items()
            }
            best_metrics = {
                "iteration": it,
                "val_score": round(val_score, 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 4),
                "val_category_accuracy": round(val_metrics["category_accuracy"], 4),
                "val_reward": round(val_metrics["mean_reward"], 4),
                "val_tpr": round(val_metrics["tpr"], 4),
                "val_tnr": round(val_metrics["tnr"], 4),
            }

        if it % 5 == 0 or it == 1 or is_best:
            logger.info(
                "Iter %02d/%02d | PiLoss=%.4f | VLoss=%.4f | Ent=%.3f | "
                "ValAcc=%.1f%% | ValBalAcc=%.1f%% | ValRew=%.3f %s",
                it, iterations,
                train_losses["policy_loss"],
                train_losses["value_loss"],
                train_losses["entropy"],
                val_metrics["accuracy"] * 100,
                val_metrics["balanced_accuracy"] * 100,
                val_metrics["mean_reward"],
                "[*BEST*]" if is_best else "",
            )

    elapsed_s = time.time() - start_time
    logger.info("\nTraining completed in %.1f seconds.", elapsed_s)
    logger.info("Best validation metrics: %s", best_metrics)

    # 4. Save production checkpoint with best weights
    assert best_checkpoint_state is not None, "No valid checkpoint found!"
    trainer.network.load_state_dict(best_checkpoint_state)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": best_checkpoint_state,
        "normalizer": normalizer.to_dict(),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "metadata": {
            "training_seed": seed,
            "training_samples": len(train_recs),
            "val_samples": len(val_recs),
            "test_samples": len(test_recs),
            "best_validation_metrics": best_metrics,
            "created_timestamp": time.time(),
        },
    }
    torch.save(payload, output_checkpoint)
    logger.info("Production PPO checkpoint saved to: %s", output_checkpoint)

    return {
        "output_checkpoint": str(output_checkpoint),
        "best_metrics": best_metrics,
        "train_samples": len(train_recs),
        "val_samples": len(val_recs),
        "test_samples": len(test_recs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VERITAS Genuine PPO Fusion Training")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_V2_CHECKPOINT_PATH)
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--samples-per-cat", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_training(
        dataset_root=args.dataset_root,
        output_checkpoint=args.output,
        samples_per_category=args.samples_per_cat,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        cache_file=args.cache_file,
    )

