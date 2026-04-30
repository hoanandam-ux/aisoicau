# SSAE – Stochastic Sequence Analysis Engine

> AI-Powered Image Analytics Platform for sequence prediction.  
> Stack: **FastAPI · EasyOCR · OpenCV · LSTM · HMM · Tailwind-styled UI**

---

## Architecture

```
ssae/
├── app/
│   ├── main.py          ← FastAPI app, /analyze endpoint
│   ├── vision.py        ← Image preprocessing + EasyOCR pipeline
│   ├── patterns.py      ← Streak / Alternating / Symmetry detector
│   └── predictor.py     ← LSTM + HMM + Naive Bayes ensemble
├── frontend/
│   └── index.html       ← Drag-and-drop VIP UI (served by FastAPI)
├── tests/
│   └── test_engine.py   ← Full pytest suite
├── .github/workflows/
│   └── ci.yml           ← GitHub Actions CI
├── Dockerfile           ← Multi-stage, non-root, Render-ready
├── render.yaml          ← Render deployment config
└── requirements.txt
```

---

## Local Development

### 1. Clone & install

```bash
git clone https://github.com/YOUR_ORG/ssae.git
cd ssae
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — drag a screenshot into the UI.

### 3. Test

```bash
pytest tests/ -v
```

---

## API Reference

### `POST /analyze`

Accepts a multipart image upload, returns JSON.

**Request**
```
Content-Type: multipart/form-data
Body: image=<file>
```

**Response**
```jsonc
{
  "status": "success",
  "processing_time_ms": 312,
  "extraction": {
    "total_values_found": 30,
    "ocr_confidence": 0.87,
    "sequence_window": [12, 8, 15, 6, ...],
    "labels_window":   ["H","L","H","L",...]
  },
  "pattern": {
    "type": "alternating",         // streak | alternating | symmetry | mixed | none
    "description": "ABAB pattern detected (score 82%)",
    "streak":      { "current_run": 1, "current_symbol": "L", "max_run": 3 },
    "alternating_score": 0.82,
    "symmetry":    { "found": false }
  },
  "prediction": {
    "next_outcome": "High",
    "probability_high": 0.673,
    "probability_low":  0.327,
    "confidence": 0.346,
    "model_used": "lstm+hmm+nb",
    "model_components": {
      "naive_bayes": 0.62,
      "hmm": 0.68,
      "lstm": 0.71
    }
  },
  "statistics": {
    "mean": 9.8, "std": 3.1, "min": 2, "max": 17, "high_ratio": 0.55
  }
}
```

### `GET /health`
```json
{ "status": "ok", "service": "ssae" }
```

---

## Deploy to Render

### Option A – render.yaml (recommended)

1. Push to GitHub.
2. In Render dashboard → **New → Blueprint** → connect your repo.
3. Render auto-reads `render.yaml` and deploys.

### Option B – Manual

1. Render dashboard → **New → Web Service** → Docker.
2. Set **Root Directory** = `/` (repo root).
3. Set env var `PORT=8000`.
4. Health Check Path: `/health`.

> **First deploy cold-start**: EasyOCR downloads the English model (~200 MB).  
> Subsequent deploys are fast because the model is baked into the Docker image during build.

---

## Tuning

| Parameter | File | Description |
|-----------|------|-------------|
| `HIGH_THRESHOLD` | `vision.py` | Dice total that separates H/L (default 9) |
| `STREAK_MIN` | `patterns.py` | Min run to qualify as streak (default 3) |
| `WINDOW` | `predictor.py` | Sessions fed to LSTM/HMM (default 20) |
| `N_COMPONENTS` | `predictor.py` | HMM hidden states (default 2) |
| `EPOCHS` | `predictor.py` | LSTM training epochs per request (default 30) |

---

## Notes

- **TensorFlow is optional.** If removed from `requirements.txt`, the engine falls back to `HMM + Naive Bayes` automatically with zero code changes.
- The LSTM is trained **per-request** on the extracted window. For production at scale, pre-train on a historical dataset and load saved weights at startup.
- EasyOCR GPU mode can be enabled by switching to `gpu=True` in `vision.py` and using a GPU instance on Render.
