"""
FastAPI application entry point for the Network Intrusion Detection Dashboard.
Phase 8 — backend server.
"""
import logging
import sys
from pathlib import Path

# Allow importing from project root (preprocessing.py etc.) if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.api import router
from app.services.model_loader import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Network Intrusion Detection Dashboard API",
    description="Phase 8 backend — serves Phase 7.1 final models and artifacts.",
    version="1.0.0",
)

# Allow Vite dev server and production build
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up — loading Phase 7.1 artifacts …")
    get_store()  # Eager load
    logger.info("Dashboard API ready.")


@app.get("/")
def root():
    return {
        "message": "Network Intrusion Detection Dashboard API",
        "docs": "/docs",
        "health": "/api/health",
    }
