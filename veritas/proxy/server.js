'use strict';

/**
 * VERITAS API Proxy Server
 * Listens on port 3001.
 * Forwards /api/analyse → Python FastAPI on port 8000.
 * Translates Python NDJSON stream → SSE for the React frontend.
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');

const app = express();
const PORT = 3001;
const PYTHON_BASE = 'http://localhost:8000';

// ── Multer: store upload in memory (max 50 MB) ─────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

// ── CORS ───────────────────────────────────────────────────────────────────
app.use(cors({
  origin: ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000'],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
}));

// ── Health check ───────────────────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ── Helper: emit one SSE event ─────────────────────────────────────────────
function sseEvent(res, event, payload) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(payload !== undefined ? payload : {})}\n\n`);
  if (typeof res.flush === 'function') res.flush();
}

// ── POST /api/analyse  — legacy blocking ──────────────────────────────────
app.post('/api/analyse', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  // Check size (multer already does this, but belt-and-suspenders)
  if (req.file.size > 50 * 1024 * 1024) {
    return res.status(413).json({ error: 'File exceeds 50MB limit.' });
  }

  try {
    const FormDataClass = globalThis.FormData;
    const BlobClass = globalThis.Blob;
    const form = new FormDataClass();
    form.append('file', new BlobClass([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);

    const pythonRes = await fetch(`${PYTHON_BASE}/analyse`, {
      method: 'POST',
      body: form,
    });

    const data = await pythonRes.json();
    return res.status(pythonRes.status).json(data);
  } catch (err) {
    const isConnRefused = err.cause?.code === 'ECONNREFUSED' || err.message?.includes('fetch failed');
    return res.status(502).json({
      error: isConnRefused
        ? 'Backend inference service unavailable (port 8000). Ensure the FastAPI backend is running.'
        : 'Python inference service unavailable',
      detail: err.message,
    });
  }
});

// ── POST /api/analyse/stream  — SSE streaming ─────────────────────────────
app.post('/api/analyse/stream', upload.single('file'), async (req, res) => {
  // Set SSE headers immediately
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  if (!req.file) {
    sseEvent(res, 'error', { message: 'No file uploaded.' });
    return res.end();
  }

  if (req.file.size > 50 * 1024 * 1024) {
    sseEvent(res, 'error', { message: 'File exceeds 50MB limit.' });
    return res.end();
  }

  // Build FormData to forward to Python
  let FormDataClass, BlobClass;
  try {
    // Node 18+ has these globals natively
    FormDataClass = globalThis.FormData;
    BlobClass = globalThis.Blob;
  } catch (_) {
    sseEvent(res, 'error', { message: 'Node 18+ required for FormData support.' });
    return res.end();
  }

  const form = new FormDataClass();
  form.append(
    'file',
    new BlobClass([req.file.buffer], { type: req.file.mimetype }),
    req.file.originalname,
  );

  // 60-second AbortController timeout (prevents premature timeout on CPU video inference)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60_000);

  let pythonRes;
  try {
    pythonRes = await fetch(`${PYTHON_BASE}/analyse/stream`, {
      method: 'POST',
      body: form,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      sseEvent(res, 'error', { message: 'Backend timeout — inference exceeded 60s. Try a shorter video or smaller file.' });
    } else {
      const isConnRefused = err.cause?.code === 'ECONNREFUSED' || err.message?.includes('fetch failed');
      const msg = isConnRefused
        ? 'Backend inference service unavailable (port 8000). Ensure the FastAPI backend is running.'
        : `Proxy connection error: ${err.message}`;
      sseEvent(res, 'error', { message: msg });
    }
    return res.end();
  } finally {
    clearTimeout(timeoutId);
  }

  if (!pythonRes.ok || !pythonRes.body) {
    sseEvent(res, 'error', {
      message: `Python inference service returned an error (${pythonRes ? pythonRes.status : 500}).`,
      status: pythonRes ? pythonRes.status : 500,
    });
    return res.end();
  }

  // ── Stream NDJSON from Python → SSE events ──────────────────────────────
  const reader = pythonRes.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        let record;
        try {
          record = JSON.parse(trimmed);
        } catch {
          continue; // malformed — skip
        }

        const type = record.type;

        if (type === 'agent_result') {
          const { type: _t, ...payload } = record;
          sseEvent(res, 'agent_update', payload);

        } else if (type === 'final_verdict') {
          const { type: _t, ...payload } = record;
          sseEvent(res, 'final_verdict', payload);

        } else if (type === 'done') {
          sseEvent(res, 'done');
          return res.end();
        }
      }
    }
  } catch (err) {
    try {
      sseEvent(res, 'error', {
        error: 'Stream error',
        detail: err.message,
      });
    } catch { /* client disconnected */ }
  }

  // Python closed without sending "done" — end cleanly
  sseEvent(res, 'done');
  res.end();
});

// ── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[VERITAS proxy] listening on http://localhost:${PORT}`);
});
