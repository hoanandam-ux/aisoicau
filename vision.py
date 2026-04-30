"""
vision.py – Automated Vision Pipeline
Extracts numerical sequences from game-history screenshots using EasyOCR + OpenCV.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    raw_text: list[str]
    sequence: list[int]          # e.g. [3, 6, 2, 5, …]
    labels: list[str]            # "H" / "L" derived from totals
    confidence: float            # mean OCR confidence [0-1]
    metadata: dict


class VisionPipeline:
    """Stateless image-to-sequence extractor.  Lazy-loads the OCR reader."""

    _reader = None                # shared across requests

    # ── thresholds ──────────────────────────────────────────────────────────
    HIGH_THRESHOLD: int = 9       # total > 9  → High; ≤ 9 → Low  (adjust per game)
    MIN_SEQUENCE_LEN: int = 5

    # ── preprocessing ───────────────────────────────────────────────────────
    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @classmethod
    def _get_reader(cls):
        if cls._reader is None:
            try:
                import easyocr  # noqa: PLC0415
                cls._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                logger.info("EasyOCR reader initialised (CPU).")
            except ImportError:
                logger.warning("EasyOCR not installed – falling back to mock reader.")
                cls._reader = _MockReader()
        return cls._reader

    # ── public API ──────────────────────────────────────────────────────────
    def extract(self, image_bytes: bytes) -> ExtractionResult:
        """Full pipeline: bytes → ExtractionResult."""
        img_bgr = self._decode(image_bytes)
        preprocessed = self._preprocess(img_bgr)
        raw, confidences = self._run_ocr(preprocessed)
        sequence, labels = self._parse_sequence(raw)

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return ExtractionResult(
            raw_text=raw,
            sequence=sequence,
            labels=labels,
            confidence=mean_conf,
            metadata={
                "image_shape": img_bgr.shape,
                "ocr_hits": len(raw),
                "parsed_values": len(sequence),
            },
        )

    # ── internal ─────────────────────────────────────────────────────────────
    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image – ensure it is JPEG/PNG/WebP.")
        return img

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        # 1. Upscale small images so OCR has enough resolution
        h, w = bgr.shape[:2]
        if max(h, w) < 800:
            scale = 800 / max(h, w)
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE for local contrast enhancement
        enhanced = self._CLAHE.apply(gray)

        # 4. Adaptive threshold – handles uneven lighting
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 2,
        )

        # 5. Slight sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(binary, -1, kernel)
        return sharp

    def _run_ocr(self, img: np.ndarray) -> tuple[list[str], list[float]]:
        reader = self._get_reader()
        results = reader.readtext(img, detail=1, paragraph=False)
        # results: [(bbox, text, conf), …]
        texts, confs = [], []
        for _, text, conf in results:
            text = text.strip()
            if text:
                texts.append(text)
                confs.append(float(conf))
        return texts, confs

    def _parse_sequence(self, texts: list[str]) -> tuple[list[int], list[str]]:
        """
        Extract integers from OCR text tokens.
        Derive H/L labels from totals (dice game convention).
        """
        numbers: list[int] = []
        for token in texts:
            # grab all digit groups in the token
            for m in re.finditer(r"\d+", token):
                val = int(m.group())
                if 1 <= val <= 18:          # valid dice-total range
                    numbers.append(val)

        if not numbers:
            logger.warning("No valid integers found in OCR output.")
            return [], []

        labels = ["H" if v > self.HIGH_THRESHOLD else "L" for v in numbers]
        return numbers, labels


# ── fallback ─────────────────────────────────────────────────────────────────
class _MockReader:
    """Used when EasyOCR is unavailable (unit tests / CI without GPU deps)."""

    def readtext(self, img, **_):  # noqa: ARG002
        rng = np.random.default_rng(42)
        values = rng.integers(2, 18, size=30).tolist()
        return [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], str(v), 0.85)
            for v in values
        ]
