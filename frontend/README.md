# DENGU HUNTER — Matte Carbon / Electric Violet Frontend

Vanilla HTML, CSS, and JavaScript frontend. No Node.js, React, or npm required.

## Start

1. Start your FastAPI backend in its own terminal:

   `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`

2. Open a **second** terminal in the extracted frontend directory and run:

   `python -m http.server 5173 --bind 127.0.0.1`

3. Visit `http://127.0.0.1:5173` in your browser.
4. Go to **Connection & Settings**, paste your backend `.env` `DENGU_API_TOKEN` into the password input, and click **Connect & Load Live Data**.

The frontend is served on port **5173** to match the CORS allowlist in the provided FastAPI backend. Opening `index.html` directly as `file://` is not recommended.

## Included screens

Dashboard, New Report + YOLO AI, Case Management (search/filter), Case Detail (Overview / AI Detection / Human Review / Worker Task / After Photo / Final Review), Worker Tasks, Human Review, After & Finalize, Analytics, Alerts, Connection & Settings. API actions use the backend's real `/api/v1` routes.

## Integrity & security

- The token is entered at runtime, kept **only in memory**, and never shipped in this ZIP, localStorage, or URL. Reloading the page requires entering it again.
- This local prototype exposes a manually entered privileged token to the browser session. **Do not deploy publicly** without server-side authentication / session management and role-based authorization.
- YOLO object detection is not dengue, larvae, or Aedes confirmation.
- Priority is a heuristic inspection score, not a disease forecast.
- Same-site attestation must be truthful; matching images alone do not prove cleanup.
- Final review is not independent field verification.
- The existing backend API does not serve before/after photos, so this frontend shows case metadata and locally selected upload previews, not remote evidence thumbnails.
- Analytics uses actual case counts and descriptive distributions. The chart is explicitly illustrative, not a fabricated historical trend.

## Backend dependencies

The frontend requires the separate DENGU HUNTER FastAPI backend and its model weights. It does not include those files.
