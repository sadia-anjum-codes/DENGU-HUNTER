"""Local FastAPI backend reusing the exact uploaded DENGU HUNTER engine."""
import os
from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .engine import DenguHunterEngine, RealYoloDetector, build_fastapi_router

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

@lru_cache(maxsize=1)
def get_engine():
    weights = Path(os.getenv("F2_MODEL_PATH", str(ROOT / "models" / "f2_best.pt")))
    if not weights.is_file():
        raise RuntimeError(f"F2 weights not found: {weights}")
    f3_path = os.getenv("F3_MODEL_PATH", "").strip()
    f3 = RealYoloDetector(f3_path) if f3_path and Path(f3_path).is_file() else None
    return DenguHunterEngine(ROOT / "data", RealYoloDetector(weights), f3)

api_token = os.getenv("DENGU_API_TOKEN", "").strip()
if len(api_token) < 32:
    raise RuntimeError("Set DENGU_API_TOKEN (32+ chars) in .env; run python setup_local.py")

app = FastAPI(title="DENGU HUNTER API", version="1.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
app.include_router(build_fastapi_router(get_engine(), api_token))
