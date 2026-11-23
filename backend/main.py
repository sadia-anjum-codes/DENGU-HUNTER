
"""Local FastAPI backend reusing the exact uploaded DENGU HUNTER engine."""

import os
import io
import secrets

import hashlib
import hmac
import threading
import time

from pydantic import BaseModel
from fastapi.responses import JSONResponse

from pathlib import Path
from functools import lru_cache

from fastapi import (
    FastAPI,
    APIRouter,
    Depends,
    Header,
    HTTPException,
)
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .engine import (
    DenguHunterEngine,
    RealYoloDetector,
    build_fastapi_router,
)

ROOT = Path(__file__).resolve().parent.parent

# Always prioritize values from the project's .env file.
load_dotenv(ROOT / ".env", override=True)


# --------------------------------------------------
# LOAD EXISTING TRAINED YOLO MODELS
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_engine():

    weights = Path(
        os.getenv(
            "F2_MODEL_PATH",
            str(ROOT / "models" / "f2_best.pt")
        )
    )

    if not weights.is_file():
        raise RuntimeError(f"F2 weights not found: {weights}")

    f3_path = os.getenv("F3_MODEL_PATH", "").strip()

    f3 = (
        RealYoloDetector(f3_path)
        if f3_path and Path(f3_path).is_file()
        else None
    )

    return DenguHunterEngine(
        ROOT / "data",
        RealYoloDetector(weights),
        f3
    )


# --------------------------------------------------
# API TOKEN
# --------------------------------------------------

api_token = os.getenv("DENGU_API_TOKEN", "").strip()

if len(api_token) < 32:
    raise RuntimeError(
        "Set DENGU_API_TOKEN (32+ chars) in .env; "
        "run python setup_local.py"
    )


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="DENGU HUNTER API",
    version="1.1.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"]
)


# --------------------------------------------------
# EXISTING API ROUTES
# --------------------------------------------------

engine = get_engine()

app.include_router(
    build_fastapi_router(engine, api_token)
)


# --------------------------------------------------
# AUTHENTICATED EVIDENCE IMAGE API
# --------------------------------------------------

evidence_router = APIRouter(
    prefix="/api/v1",
    tags=["evidence"]
)


def verify_token(
    authorization: str | None = Header(default=None)
):

    expected = "Bearer " + api_token

    if not secrets.compare_digest(
        str(authorization or ""),
        expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Bearer token required"
        )


@evidence_router.get(
    "/cases/{case_id}/images/{image_type}",
    dependencies=[Depends(verify_token)]
)
def get_case_image(case_id: str, image_type: str):

    image_positions = {
        "before": 0,
        "before_overlay": 1,
        "after": 2,
        "after_overlay": 3
    }

    if image_type not in image_positions:
        raise HTTPException(
            status_code=404,
            detail="Unknown evidence image type"
        )

    try:

        evidence = engine.inspect_case(case_id)

        image = evidence[image_positions[image_type]]

        if image is None:
            raise HTTPException(
                status_code=404,
                detail="This evidence image has not been submitted"
            )

        buffer = io.BytesIO()

        image.save(buffer, format="PNG")

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff"
            }
        )

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        ) from exc


app.include_router(evidence_router)


# --------------------------------------------------
# AUTHENTICATED INTERACTIVE GPS MAP API
# --------------------------------------------------

map_router = APIRouter(
    prefix="/api/v1",
    tags=["geospatial"]
)


@map_router.get(
    "/map",
    dependencies=[Depends(verify_token)]
)
def get_live_map():

    try:

        map_content = engine.map_html()

        return HTMLResponse(
            content=map_content,
            headers={
                "Cache-Control": "no-store"
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Unable to generate the GPS map"
        ) from exc


app.include_router(map_router)


# ==================================================
# ROLE-BASED USER LOGIN
# Citizen / Admin / Field Worker
# ==================================================

AUTH_USERS = {}
ACTIVE_SESSIONS = {}
SESSION_LOCK = threading.Lock()
SESSION_LIFETIME = 4 * 60 * 60


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        300000
    )


def load_login_users():

    roles = {
        "ADMIN": "admin",
        "CITIZEN": "citizen",
        "WORKER": "worker"
    }

    for key, role in roles.items():

        username = os.getenv(
            f"DENGU_{key}_USERNAME", ""
        ).strip()

        password = os.getenv(
            f"DENGU_{key}_PASSWORD", ""
        )

        if not username and not password:
            continue

        if not username or len(password) < 12:
            raise RuntimeError(
                f"Configure valid DENGU_{key} credentials."
            )

        normalized = username.casefold()

        if normalized in AUTH_USERS:
            raise RuntimeError(
                "Login usernames must be unique."
            )

        salt = secrets.token_bytes(16)

        AUTH_USERS[normalized] = {
            "username": username,
            "role": role,
            "salt": salt,
            "password_hash": password_hash(
                password, salt
            )
        }


load_login_users()


class LoginRequest(BaseModel):
    username: str
    password: str


login_router = APIRouter(
    prefix="/api/v1/auth",
    tags=["authentication"]
)


def current_login_user(
    authorization: str | None = Header(
        default=None,
        alias="Authorization"
    )
):

    auth = str(authorization or "")
    scheme, separator, token = auth.partition(" ")

    if (
        not separator
        or scheme.lower() != "bearer"
        or not token.strip()
    ):
        raise HTTPException(
            status_code=401,
            detail="Login required"
        )

    token_hash = hashlib.sha256(
        token.strip().encode("utf-8")
    ).hexdigest()

    with SESSION_LOCK:

        session = ACTIVE_SESSIONS.get(token_hash)

        if session and session["expires"] <= time.monotonic():
            ACTIVE_SESSIONS.pop(token_hash, None)
            session = None

    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid"
        )

    return {
        "username": session["username"],
        "role": session["role"],
        "token_hash": token_hash
    }


@login_router.post("/login")
def login(data: LoginRequest):

    if not AUTH_USERS:
        raise HTTPException(
            status_code=503,
            detail="Login accounts are not configured"
        )

    username = data.username.strip().casefold()
    user = AUTH_USERS.get(username)

    # Use a dummy hash for unknown usernames too.
    salt = (
        user["salt"]
        if user
        else b"invalid-user-salt"
    )

    expected = (
        user["password_hash"]
        if user
        else password_hash("invalid-password", salt)
    )

    supplied = password_hash(
        data.password,
        salt
    )

    if not user or not hmac.compare_digest(
        supplied, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = secrets.token_urlsafe(32)

    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    with SESSION_LOCK:

        ACTIVE_SESSIONS[token_hash] = {
            "username": user["username"],
            "role": user["role"],
            "expires": (
                time.monotonic() + SESSION_LIFETIME
            )
        }

    return JSONResponse(
        content={
            "access_token": token,
            "token_type": "bearer",
            "username": user["username"],
            "role": user["role"],
            "expires_in": SESSION_LIFETIME
        },
        headers={"Cache-Control": "no-store"}
    )


@login_router.get("/me")
def get_my_profile(
    user=Depends(current_login_user)
):

    return {
        "username": user["username"],
        "role": user["role"],
        "authenticated": True
    }


@login_router.post("/logout")
def logout(
    user=Depends(current_login_user)
):

    with SESSION_LOCK:
        ACTIVE_SESSIONS.pop(
            user["token_hash"],
            None
        )

    return {"message": "Logged out successfully"}


app.include_router(login_router)
def validate_worker_username(username):
    normalized = str(username or "").strip().casefold()
    account = AUTH_USERS.get(normalized)

    if account is None or account["role"] != "worker":
        raise ValueError("Valid worker login username required.")

    return account["username"]


# ==================================================
# CITIZEN + WORKER ROLE ACCESS
# ==================================================

import hashlib
import time
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from .engine import image_rgb, safe_case_summary
def same_user(a, b):
    return (
        str(a or "").strip().casefold()
        == str(b or "").strip().casefold()
    )

def require_role(user, role):
    if user["role"] != role:
        raise HTTPException(
            status_code=403,
            detail=f"{role} access only"
        )


async def read_role_image(upload):
    payload = await upload.read(12 * 1024 * 1024 + 1)

    if len(payload) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 12 MB")

    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image_rgb(image)
    except (OSError, ValueError):
        raise HTTPException(422, "Invalid image")


# CITIZEN: SUBMIT OWN REPORT

@app.post("/api/v1/auth/citizen/reports")
async def citizen_submit(
    image: UploadFile = File(...),
    evidence_mode: str = Form("DEMO_CASE_STAGED"),
    source: str = Form("Ground photo"),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    rain_context: str = Form("Unknown"),
    previous_reports: int = Form(0),
    notes: str = Form(""),
    closeup: UploadFile | None = File(None),
    user=Depends(current_login_user)
):
    require_role(user, "citizen")

    before = await read_role_image(image)
    close = (
        await read_role_image(closeup)
        if closeup is not None else None
    )

    try:
        case, _ = engine.submit(
            before, source, latitude, longitude,
            rain_context, previous_reports,
            close, notes, evidence_mode=evidence_mode
        )

        with engine.lock:
            case["submitted_by"] = user["username"]
            engine._save(case)

        return safe_case_summary(case)

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc))


# CITIZEN: TRACK ONLY OWN REPORTS

@app.get("/api/v1/auth/citizen/reports")
def citizen_reports(user=Depends(current_login_user)):
    require_role(user, "citizen")

    return [
        safe_case_summary(case)
        for case in engine.cases()
        if same_user(case.get("submitted_by"), user["username"])
    ]


# WORKER: SEE ONLY ASSIGNED TASKS

@app.get("/api/v1/auth/worker/tasks")
def worker_tasks(user=Depends(current_login_user)):
    require_role(user, "worker")

    return [
        safe_case_summary(case)
        for case in engine.cases()
        if same_user(case.get("assigned_worker"), user["username"])
    ]


# ==================================================
# ENFORCE ROLE PERMISSIONS ON EXISTING API
# ==================================================

class RolePermissionGate:

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):

        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "").rstrip("/")
        method = scope.get("method", "GET")

        enabled = os.getenv(
            "DENGU_RBAC_ENABLED", "1"
        ) == "1"

        if (
            not enabled
            or method == "OPTIONS"
            or not path.startswith("/api/v1/")
            or path.startswith("/api/v1/auth/")
        ):
            return await self.app(scope, receive, send)

        def deny(status, message):
            return JSONResponse(
                status_code=status,
                content={"detail": message}
            )

        headers = scope.get("headers", [])
        auth = next(
            (
                value.decode("latin-1")
                for key, value in headers
                if key.lower() == b"authorization"
            ),
            ""
        )

        scheme, separator, token = auth.partition(" ")

        if not separator or scheme.lower() != "bearer":
            return await deny(
                401, "Login required"
            )(scope, receive, send)

        token_hash = hashlib.sha256(
            token.strip().encode()
        ).hexdigest()

        with SESSION_LOCK:
            session = ACTIVE_SESSIONS.get(token_hash)

            if session and session["expires"] <= time.monotonic():
                ACTIVE_SESSIONS.pop(token_hash, None)
                session = None

            if session:
                session = session.copy()

        if not session:
            return await deny(
                401, "Invalid or expired session"
            )(scope, receive, send)

        role = session["role"]
        username = session["username"]
        allowed = role == "admin"

        parts = path.strip("/").split("/")

        case = None
        if len(parts) >= 4 and parts[:3] == [
            "api", "v1", "cases"
        ]:
            case = next(
                (
                    c for c in engine.cases()
                    if c.get("case_id") == parts[3]
                ),
                None
            )

        if role in {"citizen", "worker"}:
            if method == "GET" and path == "/api/v1/health":
                allowed = True

        if role == "citizen" and case:
            own = same_user(case.get("submitted_by"), username)

            if own and method == "GET":
                allowed = (
                    len(parts) == 4
                    or
                    (len(parts) == 6 and parts[4] == "images")
                )

        if role == "worker" and case:
            own = same_user(case.get("assigned_worker"), username)

            if own:
                if method == "GET":
                    allowed = (
                        len(parts) == 4
                        or
                        (len(parts) == 6 and parts[4] == "images")
                    )

                if method == "POST" and len(parts) == 5:
                    allowed = parts[4] in {"start", "after"}

        if not allowed:
            return await deny(
                403, "Permission denied for this role"
            )(scope, receive, send)

        # Existing engine uses the original internal API token.
        # Only authorized users can reach that engine.
        forwarded = dict(scope)

        forwarded["headers"] = [
            (key, value)
            for key, value in headers
            if key.lower() != b"authorization"
        ] + [
            (
                b"authorization",
                ("Bearer " + api_token).encode()
            )
        ]

        return await self.app(
            forwarded, receive, send
        )


app.add_middleware(RolePermissionGate)

