# ── Stage 1: dependency layer ─────────────────────────────────────────────────
FROM python:3.11-slim AS deps

# System libs required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

# Upgrade pip, install wheels in a separate layer for cache efficiency
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install/wheels -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Same system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels from deps stage
COPY --from=deps /install/wheels /usr/local

# Non-root user for security
RUN useradd -m ssae
USER ssae
WORKDIR /home/ssae/app

# Copy source
COPY --chown=ssae:ssae . .

# Pre-download EasyOCR English model during build
# (avoids cold-start delay on first request)
RUN python - <<'EOF'
import easyocr, os
os.makedirs(os.path.expanduser("~/.EasyOCR"), exist_ok=True)
easyocr.Reader(["en"], gpu=False, verbose=False)
EOF

# Expose port (Render injects PORT env var at runtime)
EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Render sets $PORT; uvicorn reads it
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --timeout-keep-alive 75"]
