# VERITAS — Deepfake Detection System

A multi-agent AI system that detects deepfakes in images, videos, and audio using four specialised neural network agents.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **pip** (comes with Python)
- **npm** (comes with Node.js)
- **FFmpeg** (required for audio analysis of video files)
  - Windows: `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Setup & Running

Open **three separate terminals** and run each service:

### Terminal 1 — Backend (FastAPI + PyTorch)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

→ Runs on **http://localhost:8000**

> First run will take longer as PyTorch downloads are heavy. Subsequent runs are instant.

### Terminal 2 — Proxy (Express.js)

```bash
cd proxy
npm install
node server.js
```

→ Runs on **http://localhost:3001**
→ Expected output: `[VERITAS proxy] listening on http://localhost:3001`

### Terminal 3 — Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

→ Open **http://localhost:5173** in your browser

---

## Architecture

```
Browser (5173)
    │  /api/*
    ▼
Proxy — Express.js (3001)
    │  Translates multipart → NDJSON → SSE
    ▼
Backend — FastAPI (8000)
    │  Multi-agent inference pipeline
    ├── Spatial Agent   (EfficientNet-B0 + GradCAM)
    ├── Temporal Agent  (R3D-18 video model)
    ├── Audio Agent     (EfficientNet-B0 + Mel Spectrogram)
    └── Context Agent   (metadata analysis)
```

## Agents

| Agent | Model | Input | Detects |
|-------|-------|-------|---------|
| **Spatial** | EfficientNet-B0 + GradCAM | Image/Video frame | Pixel-level artifacts, blending boundaries |
| **Temporal** | R3D-18 (16-frame sampling) | Video | Inter-frame inconsistencies |
| **Audio** | EfficientNet-B0 + Mel Spectrogram | Audio/Video | Speech synthesis artifacts |
| **Context** | Rule-based + metadata | Any | Contextual inconsistencies |

## Model Checkpoints

Place `.pth` checkpoint files in `backend/models/`:

```
backend/models/
├── spatial_agent_checkpoint.pth
├── temporal_agent_checkpoint.pth
└── audio_agent_checkpoint.pth
```

### Model Checkpoints Status

| Agent | Checkpoint | Status |
|-------|-----------|--------|
| Spatial | spatial_agent_checkpoint.pth | ✅ Available |
| Temporal | temporal_agent_checkpoint.pth | ⚠️ Mock mode (training in progress) |
| Audio | audio_agent_checkpoint.pth | ⚠️ Mock mode (training in progress) |
| Context | N/A | Mock (rule-based) |

If checkpoints are missing, agents fall back to mock results so the system still runs end-to-end.

## Tech Stack

- **Frontend**: React 19 + Vite 7 + Tailwind CSS 4 + Framer Motion
- **Proxy**: Express.js 4 (Node.js, CommonJS, no build step)
- **Backend**: FastAPI + PyTorch 2.6

## Notes

- The 50MB file size limit is enforced at both the proxy and backend layers.
- The proxy implements a 30-second timeout for backend inference requests.
- GradCAM heatmaps are rendered in the UI when a spatial checkpoint is available.
- FFmpeg must be installed on the system for audio extraction from video files.
