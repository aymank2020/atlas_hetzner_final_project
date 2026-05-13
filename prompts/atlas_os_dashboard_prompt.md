# Atlas OS Dashboard Prompt

Role: Senior frontend engineer and UI/UX designer focused on AIOps control planes, internal tooling, and policy-governed automation systems.

Mission: Build a premium single-page HTML5 dashboard called `Atlas OS: Centralized Rule Authority` that makes the policy pipeline visible from Discord ingestion to Python API execution.

Product Context:
- Atlas Capture ingests rule updates from Discord manager channels.
- Python normalizes those updates into structured rule chunks and stores them in `current_policy.json` and `staged_rules.jsonl`.
- The runtime does selective retrieval so only the relevant rule chunks are injected into prompts or validators.
- The Lead Architect needs one command center that shows authority, staging, cost savings, and retrieval behavior at a glance.

Output Contract:
- Return exactly one self-contained HTML file.
- Use Tailwind CSS via CDN.
- Use Chart.js for visualization.
- Use Lucide icons for iconography.
- Use vanilla JavaScript for all interactions.
- Keep the page responsive with a sidebar plus adaptive content grid.

Visual Direction:
- Theme: deep dark cyber-grid.
- Palette: `slate-950`, `slate-900`, `emerald-400`, `cyan-400`, `amber-400`.
- Typography: use a purposeful display font plus a monospace utility font.
- Atmosphere: glowing edges, grid overlays, glass panels, restrained motion, no generic admin-template look.

Required Modules:
1. Rule Authority Status
- Show the current master policy version.
- Show authority mode, last promotion time, and sync coverage with core Discord channels.
- Include a visible status badge such as `Synchronized`, `Partial Coverage`, or `Awaiting Harvest`.

2. Token Economy Monitor
- Use a Chart.js comparison of `Prose-heavy Rules` versus `Atomic Rules`.
- Make the savings legible as operational efficiency, not just pretty bars.
- Show total estimated tokens and percentage saved by structured rule chunks.

3. Staging Area
- Render recent Discord rule updates waiting for promotion.
- Include columns for rule chunk, source, confidence, value preview, and actions.
- Add `Approve` and `Reject` buttons with simulated client-side state updates.
- If no staged items exist, show an honest fallback note and seed the table with recent trusted examples for demo purposes.

4. Rule Retrieval Map
- Visualize the flow:
  Discord Managers -> Central Rule Authority -> JSON Vault -> Selective Retrieval -> Python API Executor
- Include a concrete example task like `Video Annotation`.
- Show the trigger that caused retrieval, such as `validator_error`.
- Render the retrieved rule chunks as chips or cards so the operator can see what the runtime actually received.

Interaction Requirements:
- Add a `Context Refresh` control with a clear animation and activity log update.
- Approve/Reject in the staging table should update the row state immediately in the browser.
- Keep all interactivity in plain JavaScript with no framework dependency.

Backend Hook Notes:
- Add developer comments explaining how to later replace inline JSON with Flask or FastAPI endpoints.
- Mention likely endpoints such as:
  - `GET /api/policy/current`
  - `GET /api/policy/staged`
  - `POST /api/policy/promote/{rule_id}`
  - `POST /api/context/refresh`
- Explain that the UI should be able to hydrate from `master_rules.json`, `current_policy.json`, or equivalent Python-generated payloads.

Quality Bar:
- No lorem ipsum.
- No empty hero sections.
- No stock dashboard boilerplate.
- Every widget must communicate part of the Discord-to-runtime bridge.
- The page should feel like a real operator console for a sovereign AI system, not a generic demo theme.
