"""
predictor.py – Deep Learning Trend Analyzer
Ensemble of:
  • GaussianHMM  – captures latent regime transitions
  • LSTM (Keras / TensorFlow) – learns long-range sequential patterns
  • Naive Bayes baseline – calibrated fallback

Prediction: probability that the NEXT outcome is "High".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Runtime flags ─────────────────────────────────────────────────────────────
_TF_AVAILABLE = False
_HMM_AVAILABLE = False

try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf   # noqa: F401
    from tensorflow import keras
    _TF_AVAILABLE = True
    logger.info("TensorFlow %s available.", tf.__version__)
except ImportError:
    logger.warning("TensorFlow not available – LSTM disabled.")

try:
    from hmmlearn.hmm import GaussianHMM  # noqa: F401
    _HMM_AVAILABLE = True
    logger.info("hmmlearn available.")
except ImportError:
    logger.warning("hmmlearn not available – HMM disabled.")


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    prediction: str           # "High" | "Low"
    probability_high: float   # [0, 1]
    confidence: float         # [0, 1]  – spread from 0.5
    model_used: str
    components: dict          # per-model probabilities


# ── Helpers ───────────────────────────────────────────────────────────────────
def _encode(labels: list[str]) -> np.ndarray:
    """H → 1, L → 0"""
    return np.array([1 if l == "H" else 0 for l in labels], dtype=np.float32)


# ── HMM predictor ─────────────────────────────────────────────────────────────
class HMMPredictor:
    N_COMPONENTS = 2
    N_ITER = 200

    def predict_proba_high(self, sequence: np.ndarray) -> float:
        if not _HMM_AVAILABLE or len(sequence) < 6:
            return 0.5
        from hmmlearn.hmm import GaussianHMM

        X = sequence.reshape(-1, 1)
        model = GaussianHMM(
            n_components=self.N_COMPONENTS,
            covariance_type="full",
            n_iter=self.N_ITER,
            random_state=42,
        )
        try:
            model.fit(X)
            # Predict next state by propagating last hidden state
            _, state_seq = model.decode(X, algorithm="viterbi")
            last_state = state_seq[-1]
            # Transition probabilities from last state
            trans_probs = model.transmat_[last_state]
            # Identify which hidden state corresponds to "High" via means
            means = model.means_.flatten()
            high_state = int(np.argmax(means))
            prob_high = float(trans_probs[high_state])
            return np.clip(prob_high, 0.01, 0.99)
        except Exception as exc:
            logger.warning("HMM fitting failed: %s", exc)
            return 0.5


# ── LSTM predictor ────────────────────────────────────────────────────────────
class LSTMPredictor:
    WINDOW = 20
    EPOCHS = 30
    BATCH = 8

    def predict_proba_high(self, sequence: np.ndarray) -> float:
        if not _TF_AVAILABLE or len(sequence) <= self.WINDOW:
            return 0.5

        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow import keras

        # Build sliding-window dataset
        X, y = [], []
        for i in range(len(sequence) - self.WINDOW):
            X.append(sequence[i: i + self.WINDOW])
            y.append(sequence[i + self.WINDOW])
        X = np.array(X)[..., np.newaxis]   # (samples, window, 1)
        y = np.array(y)

        if len(X) < 4:
            return 0.5

        # ── Model ──
        inp = keras.Input(shape=(self.WINDOW, 1))
        x = keras.layers.LSTM(64, return_sequences=True)(inp)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy")

        try:
            model.fit(
                X, y,
                epochs=self.EPOCHS,
                batch_size=self.BATCH,
                verbose=0,
                validation_split=0.1 if len(X) > 10 else 0.0,
            )
            last_window = sequence[-self.WINDOW:][np.newaxis, :, np.newaxis]
            prob = float(model.predict(last_window, verbose=0)[0, 0])
            return np.clip(prob, 0.01, 0.99)
        except Exception as exc:
            logger.warning("LSTM prediction failed: %s", exc)
            return 0.5


# ── Naive Bayes baseline ──────────────────────────────────────────────────────
class NaiveBayesPredictor:
    """
    Laplace-smoothed transition probability from the last symbol.
    Always available as a calibration anchor.
    """
    WINDOW = 20
    ALPHA = 1.0   # Laplace smoothing

    def predict_proba_high(self, sequence: np.ndarray) -> float:
        if len(sequence) < 2:
            return 0.5
        window = sequence[-self.WINDOW:]
        last = int(window[-1])
        # Count transitions from last symbol
        from_last = [
            int(window[i + 1])
            for i in range(len(window) - 1)
            if int(window[i]) == last
        ]
        if not from_last:
            return 0.5
        n_high = sum(from_last) + self.ALPHA
        n_total = len(from_last) + 2 * self.ALPHA
        return float(n_high / n_total)


# ── Ensemble ──────────────────────────────────────────────────────────────────
class EnsemblePredictor:
    """
    Weighted average of all available models.
    Weights auto-adjust based on sequence length and model availability.
    """

    def __init__(self) -> None:
        self.hmm = HMMPredictor()
        self.lstm = LSTMPredictor()
        self.nb = NaiveBayesPredictor()

    def predict(self, labels: list[str], window: int = 20) -> PredictionResult:
        if len(labels) < 2:
            return PredictionResult(
                prediction="Low",
                probability_high=0.5,
                confidence=0.0,
                model_used="none",
                components={},
            )

        seq = _encode(labels[-window:])

        # Gather per-model probabilities
        components: dict[str, float] = {}
        weights: dict[str, float] = {}

        # Naive Bayes – always available
        components["naive_bayes"] = self.nb.predict_proba_high(seq)
        weights["naive_bayes"] = 1.0

        # HMM
        if _HMM_AVAILABLE and len(seq) >= 6:
            components["hmm"] = self.hmm.predict_proba_high(seq)
            weights["hmm"] = 2.0

        # LSTM – most powerful, highest weight, but needs data
        if _TF_AVAILABLE and len(seq) > self.lstm.WINDOW:
            components["lstm"] = self.lstm.predict_proba_high(seq)
            weights["lstm"] = 3.0

        # Weighted ensemble
        total_w = sum(weights[k] for k in components)
        prob_high = sum(components[k] * weights[k] for k in components) / total_w
        prob_high = float(np.clip(prob_high, 0.01, 0.99))

        # Confidence = distance from 0.5, scaled to [0, 1]
        confidence = min(abs(prob_high - 0.5) * 2 * 1.1, 1.0)

        model_used = (
            "lstm+hmm+nb" if "lstm" in components
            else "hmm+nb" if "hmm" in components
            else "naive_bayes"
        )

        return PredictionResult(
            prediction="High" if prob_high > 0.5 else "Low",
            probability_high=round(prob_high, 4),
            confidence=round(confidence, 4),
            model_used=model_used,
            components={k: round(v, 4) for k, v in components.items()},
        )
