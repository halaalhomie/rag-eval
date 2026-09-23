FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model weights (embedder, reranker) are cached here; mounted as a volume in compose.
    HF_HOME=/models

WORKDIR /srv

# CPU-only torch wheels keep the image far smaller than the default CUDA build.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install dependencies first for layer caching; the package itself is installed after.
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install . \
    && rm -rf app

COPY app ./app
COPY scripts ./scripts
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 ragforge \
    && mkdir -p /models /srv/data && chown -R ragforge /models /srv/data
USER ragforge

EXPOSE 8000
CMD ["sh", "-c", "python scripts/init_db.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
