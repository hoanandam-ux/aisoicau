"""
main.py – FastAPI Application Entry Point
Stochastic Sequence Analysis Engine (SSAE)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.patterns import PatternDetector
from app.predictor import EnsemblePredictor
from app.vision import VisionPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ssae")

# ── Singleton services (initialised at startup) ───────────────────────────────
_vision: VisionPipeline | None = None
_detector: PatternDetector | None = None
_predictor: EnsemblePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vision, _detector, _predictor
    logger.info("🚀  SSAE starting – warming up services…")
    _vision = VisionPipeline()
    _detector = PatternDetector()
    _predictor = EnsemblePredictor()
    # Pre-warm EasyOCR reader (downloads models on first call)
    _vision._get_reader()
    logger.info("✅  All services ready.")
    yield
    logger.info("🛑  SSAE shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stochastic Sequence Analysis Engine",
    description=(
        "AI-powered image analytics platform for extracting sequences from "
        "game-history screenshots and predicting next outcomes using LSTM + HMM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
_FRONTEND = Path(__file__).parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    index = _FRONTEND / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text())
    return HTMLResponse("<h1>SSAE API is running. POST an image to /analyze</h1>")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ssae"}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    """
    Accepts a game-history screenshot.
    Returns extracted sequence, detected patterns, and next-outcome prediction.
    """
    t0 = time.perf_counter()

    # ── Validate file ────────────────────────────────────────────────────────
    if image.content_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {image.content_type}. Use JPEG/PNG/WebP.",
        )

    raw_bytes = await image.read()
    if len(raw_bytes) > 20 * 1024 * 1024:   # 20 MB cap
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit.")

    # ── Vision pipeline ──────────────────────────────────────────────────────
    try:
        extraction = _vision.extract(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(extraction.labels) < 3:
        raise HTTPException(
            status_code=422,
            detail=(
                "Too few numeric values extracted from image "
                f"(got {len(extraction.labels)}, need ≥ 3). "
                "Check image quality or crop closer to the data table."
            ),
        )

    # ── Last-N window ─────────────────────────────────────────────────────────
    WINDOW = 20
    window_labels = extraction.labels[-WINDOW:]
    window_values = extraction.sequence[-WINDOW:]

    # ── Pattern detection ─────────────────────────────────────────────────────
    pattern = _detector.detect(window_labels)

    # ── AI Prediction ─────────────────────────────────────────────────────────
    prediction = _predictor.predict(window_labels)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Statistics ────────────────────────────────────────────────────────────
    arr = np.array(window_values, dtype=float)
    stats = {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "high_ratio": round(window_labels.count("H") / len(window_labels), 3),
    }

    return {
        "status": "success",
        "processing_time_ms": elapsed_ms,
        # ── Raw extraction
        "extraction": {
            "total_values_found": len(extraction.sequence),
            "ocr_confidence": round(extraction.confidence, 3),
            "sequence_window": window_values,
            "labels_window": window_labels,
            "metadata": extraction.metadata,
        },
        # ── Patterns
        "pattern": {
            "type": pattern.pattern_type,
            "description": pattern.description,
            "streak": pattern.streak_info,
            "alternating_score": round(pattern.alternating_score, 3),
            "symmetry": pattern.symmetry_info,
            "dominant_label": pattern.dominant_label,
        },
        # ── Prediction
        "prediction": {
            "next_outcome": prediction.prediction,
            "probability_high": prediction.probability_high,
            "probability_low": round(1 - prediction.probability_high, 4),
            "confidence": prediction.confidence,
            "model_used": prediction.model_used,
            "model_components": prediction.components,
        },
        # ── Statistics
        "statistics": stats,
    }
