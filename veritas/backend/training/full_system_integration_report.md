# VERITAS — Full Multi-Agent Integration and System Verification Report

**Date**: October 19, 2026  
**System Status**: **READY FOR END-TO-END DEMO**  
**Environment**: Windows 11, Python 3.14 (Backend runtime) / PyTorch 2.6.0+cu124 (Training venv), Node.js v24.20.0, Vite 7.3.6

---

## 1. Executive Summary

The trained **VERITAS Audio Agent** (EfficientNet-B0 mel-spectrogram model, 4,050,894 parameters, calibrated threshold 0.50) has been fully integrated into the VERITAS multi-agent deepfake detection and reasoning system.

An end-to-end audit and integration validation was performed across the complete request flow:
- **Frontend** (React, TypeScript, Vite, Tailwind, Framer Motion)
- **API Proxy** (Node.js/Express on port 3001)
- **Backend Service** (FastAPI, PyTorch, OpenCV on port 8000)
- **Multi-Agent Pipeline** (Spatial Agent, Temporal Agent, Audio Agent, Context Agent, Fixed-Weight Fusion)

All critical integration defects—including silent conversion of agent errors/missing modalities into authentic predictions—have been diagnosed, corrected, and verified with dedicated regression suites. The application executes synchronously and via streaming without regressions.

---

## 2. End-to-End Architecture & Request Flow

```
+---------------------------------------------------------------------------------------+
|                                    REACT FRONTEND                                     |
|                       (Console.tsx: Upload, SSE Client, UI Cards)                     |
+---------------------------------------------------------------------------------------+
                                  |                     ^
            POST /api/analyse/stream                    | SSE Events:
         (multipart/form-data; file)                    | 'agent_update' -> render card
                                  |                     | 'final_verdict' -> render banner
                                  v                     | 'done' -> complete
+---------------------------------------------------------------------------------------+
|                                  NODE.JS EXPRESS PROXY                                |
|                                   (server.js :3001)                                   |
|   - Multer in-memory storage (50MB limit)                                             |
|   - Form-data proxying to FastAPI                                                     |
|   - Translates backend NDJSON lines to Server-Sent Events (SSE)                       |
+---------------------------------------------------------------------------------------+
                                  |                     ^
              POST /analyse/stream                      | NDJSON Lines:
         (multipart/form-data; file)                    | {"type": "agent_result", ...}
                                  |                     | {"type": "final_verdict", ...}
                                  v                     | {"type": "done"}
+---------------------------------------------------------------------------------------+
|                                    FASTAPI BACKEND                                    |
|                                    (main.py :8000)                                    |
|   1. Save upload to temporary file                                                    |
|   2. Media Type Detection: _detect_media_type(filename) -> image | video | audio      |
|   3. Modality Routing:                                                                |
|      * Image: Spatial (Direct Image)                                                  |
|      * Video: Spatial (Middle Frame) + Temporal (16 frames) + Audio (Container Probe) |
|      * Audio: Audio (Native WAV/MP3)                                                  |
|   4. Context Agent: Cross-modal consistency evaluation (Visual vs Audio)              |
|   5. Fusion Engine: _compute_verdict() with active weight normalization               |
|   6. Streamer: Yields NDJSON lines with artificial 400ms pace for UI animation        |
|   7. Temp File Lifecycle: Unlink temporary file in finally block                      |
+---------------------------------------------------------------------------------------+
```

### Endpoints & Contract Mapping

| Route | Protocol | Input | Output / Events |
|---|---|---|---|
| Proxy: `GET /api/health` | HTTP | None | `{"status": "ok", "timestamp": "..."}` |
| Proxy: `POST /api/analyse` | HTTP | `file` (multipart) | Full verdict JSON |
| Proxy: `POST /api/analyse/stream` | SSE | `file` (multipart) | `event: agent_update`<br>`event: final_verdict`<br>`event: done` |
| Backend: `GET /healthz` | HTTP | None | `{"status": "ok"}` |
| Backend: `POST /analyse` | HTTP | `file`, `media_type` (form) | Single aggregated verdict JSON |
| Backend: `POST /analyse/stream` | NDJSON | `file`, `media_type` (form) | Newline-delimited JSON stream |

---

## 3. Agent Contracts & Modality Routing

| Agent | Backbone & Input | Production Checkpoint | Threshold | Applicable Modalities | Inapplicable/Failure Semantics |
|---|---|---|---|---|---|
| **Spatial Analysis** (Weight: 0.40) | EfficientNet-B0 (224×224 RGB image) | `veritas/backend/models/spatial_agent.pth` | 0.50 | Image, Video (middle frame) | `status: "not_applicable"` (for audio) or `status: "error"`, `confidence: 0.0`, `is_fake: None` |
| **Temporal Analysis** (Weight: 0.30) | R3D-18 (16 frames × 112×112) | `veritas/backend/models/temporal_agent_checkpoint.pth` | 0.05 | Video only | `status: "not_applicable"` (image/audio) or `status: "error"`, `confidence: 0.0`, `is_fake: None` |
| **Audio Analysis** (Weight: 0.20) | MelSpectrogram + EfficientNet-B0 (128 mels, 224×224) | `veritas/backend/models/audio_agent_checkpoint.pth` | 0.50 | Audio, Video (with audio track) | `status: "no_audio"` or `status: "untrained"` or `status: "error"`, `confidence: 0.0`, `is_fake: None` |
| **Context Verification** (Weight: 0.10) | Cross-modal consistency analyzer | Algorithmic (Visual vs Audio divergence) | N/A | Multimodal Video only (requires both visual & audio verdicts) | `status: "not_applicable"`, `confidence: 0.0`, `is_fake: None` |

---

## 4. Fixed-Weight Fusion Audit

The fixed-weight baseline is implemented in `_compute_verdict(agents: list[dict])`:

$$\text{active\_weights} = \sum_{a \in \text{valid}} w_a$$

$$\text{overall\_confidence} = \frac{\sum_{a \in \text{valid}} c_a \cdot w_a}{\text{active\_weights}}$$

### Decision Rules
- $\text{overall\_confidence} > 0.55 \implies \textbf{DEEPFAKE}$
- $\text{overall\_confidence} < 0.45 \implies \textbf{AUTHENTIC}$
- $0.45 \le \text{overall\_confidence} \le 0.55 \implies \textbf{UNCERTAIN / INSUFFICIENT EVIDENCE}$
- $\text{active\_weights} = 0.0 \implies \textbf{UNVERIFIED MODALITY}$ (Score: 0.0)

### Invariant Guarantees
1. **Dynamic Denominator Normalization**: If a video lacks audio, Audio (0.20) and Context (0.10) are excluded. The denominator renormalizes to $0.40 + 0.30 = 0.70$. A missing modality never drags the prediction towards 0 (REAL).
2. **Strict Exclusion of Inapplicable / Failed Agents**: An agent with `is_fake is None` or `status` in `("not_applicable", "unavailable", "no_audio", "untrained", "error", "failed")` is strictly placed into `unavailable_agents` and excluded from `active_weights`.

---

## 5. Bugs Identified & Fixed

During end-to-end integration testing, 5 critical bugs were discovered and resolved:

### Bug 1: Temporal Agent Silently Converted Failures to Authentic (REAL)
- **Location**: `veritas/backend/temporal_agent.py` (lines 320–326, 335–341, 378–384)
- **Problem**: When the checkpoint was missing, frames unreadable, or an exception thrown, the agent returned:
  `{"agent": "Temporal Analysis", "confidence": 0.5, "is_fake": False}`
  Because `is_fake` was boolean `False`, fusion interpreted this as a valid prediction of authenticity, diluting real fake detections.
- **Fix**: Updated all failure/unavailable branches to return `status: "unavailable"` or `status: "error"`, `confidence: 0.0`, and `is_fake: None`.

### Bug 2: Video Frame Extraction Failure Returned is_fake=False
- **Location**: `veritas/backend/main.py` (`_build_agents` and `_stream_agents`)
- **Problem**: If `_extract_video_frame` failed, Spatial Analysis returned `confidence: 0.5` and `is_fake: False`.
- **Fix**: Corrected to return `status: "error"`, `confidence: 0.0`, `is_fake: None`.

### Bug 3: Spatial Agent Unhandled Exceptions on Corrupt Inputs
- **Location**: `veritas/backend/main.py`
- **Problem**: Direct calls to `spatial_agent.predict()` were unhandled. Corrupt files or tensor errors threw a 500 server crash.
- **Fix**: Wrapped spatial inference in `try...except`, setting `status: "error"`, `confidence: 0.0`, and `is_fake: None`. Added `"status": "trained"` on success.

### Bug 4: Incomplete `is_unavailable` Filter in Fusion
- **Location**: `veritas/backend/main.py` (`_compute_verdict`)
- **Problem**: The check only tested `a.get("status") in ("not_applicable", "unavailable")`, omitting `no_audio`, `untrained`, `error`, and `failed`.
- **Fix**: Expanded status exclusion tuple to include all non-contributing statuses.

### Bug 5: Frontend UI Card Display for Unavailable / Failed Agents
- **Location**: `veritas/frontend/src/pages/Console.tsx`
- **Problem**: `AgentCard` `isUnavailable` check only evaluated `['untrained', 'mocked', 'no_audio']`, causing `not_applicable` or `error` to display as 0.0% score rather than "N/A" / "Unavailable".
- **Fix**: Updated `isUnavailable` to cover all non-contributing statuses and `is_fake === null || is_fake === undefined`.

---

## 6. Verification Test Matrix & Results

| Check / Test Group | Tool / Script | Status | Observations |
|---|---|---|---|
| **Audio Agent Production Verification** | `test_audio_agent_production_integration.py` | **PASS** | 46/46 checks passed. Model weights, threshold (0.50), preprocessing parity, and CPU/GPU parity verified. |
| **Cross-Modal Context Agent** | `test_context_agent.py` | **PASS** | Evaluated real-real, fake-fake, mismatch detection, and zero-weight exclusion when single-modality. |
| **Backend Routing & Fusion Suite** | `test_integration_routing_fusion.py` | **PASS** | 44/44 checks passed. Modality routing, dynamic normalization, conflict weighting, and failed-agent exclusions verified. |
| **Dynamic Fusion Safeguards** | `test_fusion.py` | **PASS** | Verified formula calculations across all combinations. |
| **Video Frame Routing** | `test_video_routing.py` | **PASS** | Verified middle frame extraction and temporal frame decoding without PIL errors. |
| **Live Smoke Test** | `smoke_test.py` | **PASS** | Live `/healthz` and `/analyse` multipart upload succeeded. |
| **Proxy Integration & Contract** | `test_proxy_integration.js` | **PASS** | 4/4 checks passed. Health check, SSE framing, 50MB file limit, and NDJSON streaming verified. |
| **Frontend TypeScript Build** | `tsc && vite build` | **PASS** | Clean build with zero TypeScript errors. Bundle size: 537 kB. |
| **Live Server Pipeline Test** | Backend (:8000) + Proxy (:3001) | **PASS** | Live requests and SSE streaming processed to completion. |

---

## 7. Preservation of the PPO Research Path

> [!NOTE]
> Per project directives, Proximal Policy Optimization (PPO) has **NOT** been implemented in this phase. The current verified system serves as the frozen, deterministic baseline against which future RL agents will be benchmarked.

### Research Path Documentation:

1. **Current Fixed-Weight Limitations**:
   - Fixed weights (0.40, 0.30, 0.20, 0.10) assume uniform reliability across all noise conditions, compressions, and modalities.
   - Fixed weights cannot adapt when one modality exhibits high perceptual uncertainty (e.g. heavy background music, motion blur, compression artifacts).

2. **PPO State & Feature Vector**:
   Future PPO policy network can receive an observation vector:
   $$s = [p_{\text{spatial}}, p_{\text{temporal}}, p_{\text{audio}}, c_{\text{spatial}}, c_{\text{temporal}}, c_{\text{audio}}, \text{divergence}(V, A), \text{audio\_snr}, \text{frame\_sharpness}, \text{duration}]$$

3. **PPO Action Space**:
   - Continuous: Softmax weight vector $\mathbf{w} \in \Delta^3$ that dynamically weights agents conditioned on the input features.
   - Discrete: Sequential decision to invoke expensive agents (e.g., call temporal or audio only if spatial is uncertain).

4. **Reward Formulation & Leakage Prevention**:
   - Reward: Negative Brier score / cross-entropy loss against ground-truth labels.
   - Penalty: Computational cost penalty for calling unnecessary agents.
   - Leakage Prevention: PPO training must be restricted to FakeAVCeleb training identities. The validation and test sets must remain held-out with zero overlap to avoid identity leakage.

---

## 8. Exact Commands for Starting the Application

To run the complete VERITAS application locally:

### Terminal 1: Backend Service (FastAPI)
```powershell
cd d:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\backend
C:\Python314\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Terminal 2: Proxy Server (Express)
```powershell
cd d:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\proxy
node server.js
```

### Terminal 3: Frontend Client (Vite Dev Server)
```powershell
cd d:\Multi-Agent-DeepFake-Detection-And-Reasoning-main\Multi-Agent-DeepFake-Detection-And-Reasoning-main\veritas\frontend
npm run dev
```
Open **http://localhost:5173/console** in your browser.

---

## 9. Conclusion & Overall Status

**FINAL STATUS**: **READY FOR END-TO-END DEMO**

Every major pipeline component, contract invariant, dynamic fusion rule, and agent integration test has been executed and passed.
