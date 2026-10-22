# VERITAS — Multi-Agent Deepfake Detection System

A multi-agent AI system for detecting deepfakes in images, videos, and audio using specialized neural network agents.

## Agents
- **Spatial Agent** — EfficientNet-B0 for pixel-level artifact detection
- **Temporal Agent** — R3D-18 for inter-frame inconsistency detection
- **Audio Agent** — EfficientNet-B0 + Mel Spectrogram for speech synthesis detection
- **Context Agent** — Contextual metadata analysis

## Architecture
Three-tier: React frontend → Express proxy → FastAPI + PyTorch backend

## Quick Start
See [veritas/README.md](veritas/README.md) for full setup instructions.
