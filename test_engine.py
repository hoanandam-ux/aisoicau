"""
tests/test_engine.py
Run: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Pattern detector tests ────────────────────────────────────────────────────
from app.patterns import PatternDetector


@pytest.fixture
def detector():
    return PatternDetector()


def test_streak_detection(detector):
    labels = ["H", "L", "H", "H", "H", "H"]
    result = detector.detect(labels)
    assert result.pattern_type in ("streak", "mixed")
    assert result.streak_info["current_run"] >= 4
    assert result.streak_info["current_symbol"] == "H"


def test_alternating_detection(detector):
    labels = ["H", "L", "H", "L", "H", "L", "H", "L"]
    result = detector.detect(labels)
    assert result.pattern_type in ("alternating", "mixed")
    assert result.alternating_score >= 0.75


def test_symmetry_detection(detector):
    labels = ["H", "L", "H", "H", "L", "L"]
    result = detector.detect(labels)
    assert result.symmetry_info.get("found") is True


def test_no_pattern(detector):
    labels = ["H", "H", "L", "H", "L", "L", "H", "H", "L", "H"]
    result = detector.detect(labels)
    # Just check it returns a valid result
    assert result.pattern_type in ("streak", "alternating", "symmetry", "mixed", "none")


def test_empty_labels(detector):
    result = detector.detect([])
    assert result.pattern_type == "none"


def test_dominant_label(detector):
    labels = ["H"] * 7 + ["L"] * 3
    result = detector.detect(labels)
    assert result.dominant_label == "H"


# ── Predictor tests ───────────────────────────────────────────────────────────
from app.predictor import EnsemblePredictor, _encode


def test_encode():
    arr = _encode(["H", "L", "H"])
    np.testing.assert_array_equal(arr, [1, 0, 1])


def test_predict_short_sequence():
    pred = EnsemblePredictor()
    result = pred.predict(["H"])
    assert result.prediction in ("High", "Low")
    assert 0.0 <= result.probability_high <= 1.0


def test_predict_normal_sequence():
    pred = EnsemblePredictor()
    labels = ["H", "L", "H", "L", "H", "H", "L", "L", "H", "L"] * 2
    result = pred.predict(labels)
    assert result.prediction in ("High", "Low")
    assert 0.0 <= result.confidence <= 1.0
    assert result.model_used != ""


def test_predict_streak_favors_continuation():
    pred = EnsemblePredictor()
    labels = ["H"] * 15
    result = pred.predict(labels)
    # Not deterministic across models, but should lean toward High
    assert result.probability_high > 0.5


# ── Vision pipeline tests (mock reader) ──────────────────────────────────────
from app.vision import VisionPipeline, _MockReader


def test_vision_mock_extraction():
    pipe = VisionPipeline()
    pipe._reader = _MockReader()    # inject mock

    # Create a tiny valid PNG in memory
    import io
    from PIL import Image
    img = Image.new("RGB", (200, 100), color=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    result = pipe.extract(image_bytes)
    assert isinstance(result.sequence, list)
    assert isinstance(result.labels, list)
    assert all(l in ("H", "L") for l in result.labels)
    assert len(result.sequence) == len(result.labels)


def test_vision_invalid_bytes():
    pipe = VisionPipeline()
    with pytest.raises(ValueError, match="decode"):
        pipe.extract(b"not an image")


# ── FastAPI endpoint tests ────────────────────────────────────────────────────
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Patch the global singletons before importing app
    import app.main as main_module
    from app.patterns import PatternDetector
    from app.predictor import EnsemblePredictor
    from app.vision import VisionPipeline

    main_module._vision = VisionPipeline()
    main_module._vision._reader = _MockReader()
    main_module._detector = PatternDetector()
    main_module._predictor = EnsemblePredictor()

    return TestClient(main_module.app, raise_server_exceptions=True)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_valid_image(client):
    import io
    from PIL import Image

    img = Image.new("RGB", (300, 200), color=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/analyze",
        files={"image": ("test.png", buf, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "pattern" in data
    assert "extraction" in data
    assert data["prediction"]["next_outcome"] in ("High", "Low")


def test_analyze_bad_content_type(client):
    resp = client.post(
        "/analyze",
        files={"image": ("test.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415
