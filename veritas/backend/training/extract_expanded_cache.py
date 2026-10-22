"""
extract_expanded_cache.py
=========================
Resumable, deterministic feature extraction pipeline for VERITAS PPO Fusion.
Extracts balanced, group-disjoint subsets from FakeAVCeleb v1.2:
- Test: 200 raw clips (50 per category across 50 held-out test IDs) + 40 missing-modality aug = 240 records.
- Val:  200 raw clips (50 per category across 50 held-out val IDs) + 40 missing-modality aug = 240 records.
- Train: 400 raw clips (100 per category across 400 train IDs) + 80 missing-modality aug = 480 records.
Total = 960 records (800 raw clips).

Reuses existing cached predictions in `training/cache/interim_feature_extraction.json`.
"""

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import audio_agent
import spatial_agent
import temporal_agent
from mock_agents import context_agent_result
from main import _extract_video_frame
from fusion_policy import extract_state_from_agents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_expanded_cache")

DATASET_ROOT = Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
CACHE_DIR = _BACKEND_DIR / "training" / "cache"
MANIFEST_PATH = _BACKEND_DIR / "training" / "ppo_split_manifest.json"
INTERIM_CACHE = CACHE_DIR / "interim_feature_extraction.json"
OUTPUT_CACHE = CACHE_DIR / "fusion_dataset_cache_expanded.json"

CATEGORIES = [
    "RealVideo-RealAudio",      # 0: AUTHENTIC
    "RealVideo-FakeAudio",      # 1: AUDIO_MANIPULATED
    "FakeVideo-RealAudio",      # 2: VIDEO_MANIPULATED
    "FakeVideo-FakeAudio",      # 3: AUDIO_VIDEO_MANIPULATED
]


def extract_identity_from_path(mp4_path: Path, category: str) -> str:
    """Extract speaker/face identity token from path or filename."""
    folder = mp4_path.parent.name
    id_tokens = re.findall(r"id\d+", folder, re.IGNORECASE)
    folder_id = id_tokens[0].lower() if id_tokens else folder.lower()

    if category in ("RealVideo-RealAudio", "RealVideo-FakeAudio"):
        return folder_id

    fname = mp4_path.stem
    m = re.search(r"_(id\d+)_wav(?:to)?lip", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m = re.search(r"_wav2lip_(id\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return folder_id


def extract_single_clip(video_path: str) -> Dict[str, Any]:
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


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and identities
    logger.info("Loading split configuration from %s", MANIFEST_PATH)
    all_identities: Set[str] = set()
    category_files: Dict[int, Dict[str, List[Path]]] = {c: {} for c in range(4)}

    for cat_idx, cat_name in enumerate(CATEGORIES):
        cat_dir = DATASET_ROOT / cat_name
        mp4s = sorted(cat_dir.rglob("*.mp4"))
        for p in mp4s:
            # Exclude decompression collision files
            if " (1).mp4" in p.name:
                continue
            ident = extract_identity_from_path(p, cat_name)
            all_identities.add(ident)
            category_files[cat_idx].setdefault(ident, []).append(p)

    sorted_idents = sorted(list(all_identities))
    rng = random.Random(42)
    rng.shuffle(sorted_idents)

    val_idents = set(sorted_idents[:50])
    test_idents = set(sorted_idents[50:100])
    train_idents = set(sorted_idents[100:])

    logger.info("Identities partitioned: Train=%d, Val=%d, Test=%d",
                len(train_idents), len(val_idents), len(test_idents))

    # Assert strict disjointness
    assert len(train_idents & val_idents) == 0, "Train-Val leak!"
    assert len(train_idents & test_idents) == 0, "Train-Test leak!"
    assert len(val_idents & test_idents) == 0, "Val-Test leak!"

    # 2. Select balanced clips per pool
    def select_clips(ident_set: Set[str], count_per_cat: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for cat_idx in range(4):
            cat_candidates: List[Path] = []
            for ident in sorted(list(ident_set)):
                clips = category_files[cat_idx].get(ident, [])
                if clips:
                    # Pick deterministically 1 clip per identity
                    shuffled_clips = sorted(clips, key=lambda x: str(x))
                    cat_candidates.append(shuffled_clips[0])

            # If more candidates than needed, pick deterministically
            r = random.Random(42 + cat_idx)
            r.shuffle(cat_candidates)
            chosen = cat_candidates[:count_per_cat]

            for p in chosen:
                selected.append({
                    "path": str(p),
                    "filename": p.name,
                    "category": cat_idx,
                    "category_name": CATEGORIES[cat_idx],
                    "identity": extract_identity_from_path(p, CATEGORIES[cat_idx]),
                })
        return selected

    test_selected = select_clips(test_idents, 50)  # 50 per cat = 200
    val_selected = select_clips(val_idents, 50)    # 50 per cat = 200
    train_selected = select_clips(train_idents, 100) # 100 per cat = 400

    logger.info("Target raw clips: Train=%d, Val=%d, Test=%d (Total=%d)",
                len(train_selected), len(val_selected), len(test_selected),
                len(train_selected) + len(val_selected) + len(test_selected))

    # 3. Load or initialize interim extraction cache
    interim_cache: Dict[str, Dict[str, Any]] = {}
    if INTERIM_CACHE.exists():
        try:
            with open(INTERIM_CACHE, "r", encoding="utf-8") as f:
                interim_cache = json.load(f)
            logger.info("Loaded %d pre-extracted records from interim cache.", len(interim_cache))
        except Exception as e:
            logger.warning("Could not read interim cache: %s", e)

    # 4. Check detector models
    spatial_agent.load_model()
    temporal_agent.load_model()
    audio_agent._load_model()

    # 5. Extract all needed clips
    all_selected = train_selected + val_selected + test_selected
    needed = [s for s in all_selected if s["path"] not in interim_cache]
    logger.info("Clips to extract now: %d (already cached: %d)",
                len(needed), len(all_selected) - len(needed))

    t_start = time.time()
    for idx, sample in enumerate(needed, 1):
        p_str = sample["path"]
        try:
            res = extract_single_clip(p_str)
            interim_cache[p_str] = res
        except Exception as exc:
            logger.error("Failed extracting %s: %s", p_str, exc)

        if idx % 10 == 0 or idx == len(needed):
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1e-4)
            rem = (len(needed) - idx) / max(rate, 1e-4)
            logger.info("[%d/%d] Extracted %.1f clips/sec, ~%.1fs remaining...",
                        idx, len(needed), rate, rem)
            # Save progress
            with open(INTERIM_CACHE, "w", encoding="utf-8") as f:
                json.dump(interim_cache, f)

    # Final save of interim cache
    with open(INTERIM_CACHE, "w", encoding="utf-8") as f:
        json.dump(interim_cache, f)
    logger.info("All raw clips extracted and saved to interim cache.")

    # 6. Build split records and add missing-modality augmentations
    def build_split_records(
        raw_selection: List[Dict[str, Any]],
        aug_count_per_cat: int,
        split_name: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for s in raw_selection:
            p_str = s["path"]
            ext = interim_cache.get(p_str)
            if not ext:
                continue
            rec = {
                "sample_id": f"{split_name}_{s['identity']}_{s['category']}_{Path(p_str).stem}",
                "path": p_str,
                "identity": s["identity"],
                "category": s["category"],
                "category_name": s["category_name"],
                "split": split_name,
                "state_vec": ext["state_vec"],
                "metadata": ext["metadata"],
                "agents": ext["agents"],
                "modality_type": "audiovisual",
            }
            records.append(rec)

        # Missing-modality augmentations (audio missing or video missing)
        r_aug = random.Random(42)
        by_cat: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(4)}
        for r in records:
            by_cat[r["category"]].append(r)

        for c in range(4):
            candidates = by_cat[c]
            r_aug.shuffle(candidates)
            for i in range(min(aug_count_per_cat, len(candidates))):
                base_rec = candidates[i]
                orig_state = list(base_rec["state_vec"])
                orig_meta = dict(base_rec["metadata"])

                if i % 2 == 0:
                    # Strip Audio (simulate image or silent video)
                    aug_state = list(orig_state)
                    aug_state[6] = 0.5   # au_prob
                    aug_state[7] = 1.0   # au_uncertainty
                    aug_state[8] = 0.0   # au_valid
                    aug_state[9] = 0.0   # ctx_mismatch
                    aug_state[10] = 0.0  # ctx_valid
                    aug_state[11] = 0.0  # au_duration
                    aug_state[12] = 0.0  # has_audio
                    aug_meta = dict(orig_meta)
                    aug_meta["audio_valid"] = False
                    aug_meta["context_valid"] = False
                    aug_meta["has_audio"] = False
                    aug_meta["raw_audio_prob"] = None

                    records.append({
                        "sample_id": f"{base_rec['sample_id']}_aug_no_audio",
                        "path": base_rec["path"],
                        "identity": base_rec["identity"],
                        "category": base_rec["category"],
                        "category_name": base_rec["category_name"],
                        "split": split_name,
                        "state_vec": aug_state,
                        "metadata": aug_meta,
                        "agents": base_rec["agents"],
                        "modality_type": "video_only",
                    })
                else:
                    # Strip Video (simulate audio-only recording)
                    aug_state = list(orig_state)
                    aug_state[0] = 0.5   # sp_prob
                    aug_state[1] = 1.0   # sp_uncertainty
                    aug_state[2] = 0.0   # sp_valid
                    aug_state[3] = 0.5   # te_prob
                    aug_state[4] = 1.0   # te_uncertainty
                    aug_state[5] = 0.0   # te_valid
                    aug_state[9] = 0.0   # ctx_mismatch
                    aug_state[10] = 0.0  # ctx_valid
                    aug_state[13] = 0.0  # has_video
                    aug_meta = dict(orig_meta)
                    aug_meta["spatial_valid"] = False
                    aug_meta["temporal_valid"] = False
                    aug_meta["context_valid"] = False
                    aug_meta["has_video"] = False
                    aug_meta["raw_spatial_prob"] = None
                    aug_meta["raw_temporal_prob"] = None

                    records.append({
                        "sample_id": f"{base_rec['sample_id']}_aug_no_video",
                        "path": base_rec["path"],
                        "identity": base_rec["identity"],
                        "category": base_rec["category"],
                        "category_name": base_rec["category_name"],
                        "split": split_name,
                        "state_vec": aug_state,
                        "metadata": aug_meta,
                        "agents": base_rec["agents"],
                        "modality_type": "audio_only",
                    })

        return records

    test_records = build_split_records(test_selected, aug_count_per_cat=10, split_name="test")
    val_records = build_split_records(val_selected, aug_count_per_cat=10, split_name="val")
    train_records = build_split_records(train_selected, aug_count_per_cat=20, split_name="train")

    logger.info("Final record counts: Train=%d, Val=%d, Test=%d (Total=%d)",
                len(train_records), len(val_records), len(test_records),
                len(train_records) + len(val_records) + len(test_records))

    dataset_payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "FakeAVCeleb v1.2",
            "feature_version": "v2_expanded",
            "state_dim": 14,
            "counts": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
                "total": len(train_records) + len(val_records) + len(test_records),
            },
        },
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    with open(OUTPUT_CACHE, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f)

    logger.info("Successfully generated and saved expanded dataset to: %s", OUTPUT_CACHE)


if __name__ == "__main__":
    main()
