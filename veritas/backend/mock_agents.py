"""
mock_agents.py

Context Verification agent.

Unlike the spatial/temporal/audio agents, "context" is NOT a trained neural
network — FakeAVCeleb has no context-classification target, and a model trained
on its folder structure would merely leak the label. Instead, the context agent
performs *cross-modal consistency analysis*: it compares what the visual agents
(spatial + temporal) concluded against what the audio agent concluded.

This is the exact signal FakeAVCeleb's category taxonomy encodes:
    RealVideo-RealAudio  → both modalities agree REAL
    RealVideo-FakeAudio  → visual REAL, audio FAKE   (mismatch → manipulated)
    FakeVideo-RealAudio  → visual FAKE, audio REAL   (mismatch → manipulated)
    FakeVideo-FakeAudio  → both modalities agree FAKE

Decision logic
--------------
- If fewer than two modalities produced a definitive verdict (is_fake not None),
  context cannot reason about consistency → returns is_fake=None (contributes
  zero to fusion, exactly like the old stub). This keeps image-only and
  audio-only uploads unaffected.
- If both a visual verdict and an audio verdict exist:
    * agreement (both fake or both real) → context confidence tracks the shared
      direction, mildly reinforcing the consensus.
    * disagreement → context flags manipulation (is_fake=True) with confidence
      scaled by how strongly the two modalities diverge.

The returned dict keeps the same shape the frontend and fusion already expect
(agent/color/confidence/is_fake/finding), so no downstream changes are needed
beyond passing the upstream results in.
"""

from typing import Optional

_AGENT_NAME = "Context Verification"
_COLOR = "purple"


def _visual_fake_prob(spatial: Optional[dict], temporal: Optional[dict]) -> Optional[float]:
    """
    Combine the visual agents into a single P(FAKE), weighting spatial:temporal
    2:1 (mirrors the 0.40:0.30 fusion weights). Returns None if neither made a
    definitive prediction.
    """
    parts: list[tuple[float, float]] = []  # (weight, fake_prob)
    if spatial and spatial.get("is_fake") is not None:
        parts.append((0.40, float(spatial.get("confidence", 0.0))))
    if temporal and temporal.get("is_fake") is not None:
        parts.append((0.30, float(temporal.get("confidence", 0.0))))
    if not parts:
        return None
    wsum = sum(w for w, _ in parts)
    return sum(w * p for w, p in parts) / wsum


def _unavailable(reason: str) -> dict:
    """Old-stub-compatible response: no authenticity contribution."""
    return {
        "agent":      _AGENT_NAME,
        "status":     "not_applicable",
        "color":      _COLOR,
        "confidence": 0,
        "is_fake":    None,
        "finding":    reason,
    }


def context_agent_result(
    spatial_result: Optional[dict] = None,
    temporal_result: Optional[dict] = None,
    audio_result: Optional[dict] = None,
) -> dict:
    """
    Cross-modal consistency verdict.

    Parameters
    ----------
    spatial_result, temporal_result, audio_result : dict | None
        The upstream agent results produced earlier in the pipeline. Any may be
        None or carry is_fake=None when a modality does not apply.

    Returns
    -------
    dict with the standard agent shape. is_fake is None (zero fusion weight)
    unless BOTH a visual and an audio verdict are available.
    """
    # Called with no context (e.g. legacy callers) → behave like the old stub.
    if spatial_result is None and temporal_result is None and audio_result is None:
        return _unavailable(
            "Context agent requires upstream agent results. Authenticity cannot be verified."
        )

    visual_p = _visual_fake_prob(spatial_result, temporal_result)
    audio_has = bool(audio_result) and audio_result.get("is_fake") is not None
    audio_p = float(audio_result.get("confidence", 0.0)) if audio_has else None

    # Need both a visual and an audio verdict to reason about consistency.
    if visual_p is None or audio_p is None:
        if visual_p is None and audio_p is None:
            reason = "No modality verdicts available for cross-modal consistency analysis."
        elif audio_p is None:
            reason = (
                "Only a visual verdict is available (no audio track). "
                "Cross-modal consistency cannot be assessed."
            )
        else:
            reason = (
                "Only an audio verdict is available (no visual analysis). "
                "Cross-modal consistency cannot be assessed."
            )
        return _unavailable(reason)

    visual_fake = visual_p >= 0.5
    audio_fake  = audio_p >= 0.5
    divergence  = abs(visual_p - audio_p)   # 0 (identical) … 1 (opposite)

    if visual_fake != audio_fake:
        # Modalities disagree → hallmark of a partially-manipulated sample
        # (e.g. genuine face dubbed with synthetic audio, or vice-versa).
        confidence = round(min(0.5 + divergence / 2.0, 0.99), 4)
        v_word = "manipulated" if visual_fake else "authentic"
        a_word = "synthetic"   if audio_fake  else "authentic"
        finding = (
            f"Cross-modal inconsistency detected: the visual stream appears {v_word} "
            f"(P={visual_p:.0%}) while the audio stream appears {a_word} (P={audio_p:.0%}). "
            f"This modality mismatch is a strong indicator of partial manipulation."
        )
        return {
            "agent":      _AGENT_NAME,
            "status":     "trained",
            "color":      _COLOR,
            "confidence": confidence,
            "is_fake":    True,
            "finding":    finding,
        }

    # Modalities agree — reinforce the shared direction.
    agreed_p = (visual_p + audio_p) / 2.0
    confidence = round(agreed_p, 4)
    if visual_fake:
        finding = (
            f"Cross-modal agreement: both visual (P={visual_p:.0%}) and audio "
            f"(P={audio_p:.0%}) streams indicate manipulation. Consistent evidence "
            f"of a fully synthetic sample."
        )
    else:
        finding = (
            f"Cross-modal agreement: both visual (P={visual_p:.0%}) and audio "
            f"(P={audio_p:.0%}) streams appear authentic. No inter-modal "
            f"inconsistencies detected."
        )
    return {
        "agent":      _AGENT_NAME,
        "status":     "trained",
        "color":      _COLOR,
        "confidence": confidence,
        "is_fake":    bool(visual_fake),
        "finding":    finding,
    }
