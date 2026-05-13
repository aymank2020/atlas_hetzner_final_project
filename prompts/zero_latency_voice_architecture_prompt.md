# The "Zero-Latency" Streaming Voice Pipeline Prompt

Role: Senior Audio Systems Architect and Real-Time AI Infrastructure Engineer.

Objective: Design and implement a production-grade voice-to-text and live translation system with visible interim text, low speech-to-screen delay, and graceful fallback when cloud streaming is unavailable.

Requirements:

1. Capture Layer
- Use microphone/system audio in 20ms-40ms frames.
- Run local VAD before network transmission.
- Prefer Silero VAD when available; fall back to energy gating if the runtime is offline or minimal.

2. Streaming Layer
- Use WebSockets for cloud STT providers.
- Normalize provider events into a shared schema: `text`, `is_final`, `language`, `confidence`, `provider`.
- Support at least one local fallback path so the product still works without API keys.

3. UX Layer
- Implement "ghost writing" interim text that updates while the speaker is still talking.
- Stabilize or replace interim text when final provider segments arrive.
- Show latency, queue depth, active provider, and VAD backend in the UI.

4. Translation Layer
- Translate interim and final transcript updates in a background worker.
- Prefer a fast flash-class model and debounce requests to control cost.
- Preserve unfinished phrases when translating interim text.

5. Engineering Constraints
- Keep the code modular and backend-agnostic.
- Make the pipeline resilient to dropped chunks, device reconnects, and missing optional dependencies.
- Document where to plug in Flask/FastAPI or a browser client later.

Deliverables:
- Python engine with interim/final transcript state.
- Async WebSocket provider clients for cloud streaming.
- UI that exposes ghost text, final text, and latency telemetry.
