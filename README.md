# DENGU HUNTER — Local FastAPI Backend (Phase 1)

Reuses the **exact engine source** from the uploaded `DENGU_HUNTER_GITHUB_FINAL_MASTER (2).ipynb` active cell 2/6. Includes uploaded F2 weights and staged case/evidence files. F3 is optional and not included. **No retraining.**

## Windows PowerShell — VS Code terminal

```powershell
cd DENGU_HUNTER_BACKEND
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_local.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API schema. **All `/api/v1/` endpoints require** `Authorization: Bearer <DENGU_API_TOKEN from .env>`; `/docs` is documentation only. To test: use `/api/v1/health` via the docs **Authorize** button (enter `Bearer ...` if supported) or PowerShell:

```powershell
$token = (Get-Content .env | Select-String '^DENGU_API_TOKEN=').ToString().Split('=',2)[1]
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health -Headers @{ Authorization = "Bearer $token" }
```

## API endpoints

`GET /api/v1/health`, `GET /api/v1/cases`, `GET /api/v1/cases/{case_id}`, `GET /api/v1/dashboard`, `POST /api/v1/cases` (image multipart), `POST /api/v1/cases/{case_id}/review`, `/assign`, `/start`, `/after`, `/finalize`.

## Important limitations

- Existing two cases are **staged demo evidence**, not verified field cleanups; GPS `(0,0)` means unavailable.
- Drone capability is simulation only; false alert rate remains not evaluated without independent labelled images.
- This is a **local development backend**, not a hardened public deployment. Single shared bearer token is **not** role-based authentication; do not expose it in frontend JavaScript or publish `.env`.
- React frontend and production-grade login/database are **not yet included**. API must be secured through a proper authentication layer before public use.
- The model can run on CPU; first inference may be slow. Python dependencies may take several minutes to install.
