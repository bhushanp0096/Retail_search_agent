# AI Search Agent — single image, two entrypoints (API and Streamlit UI).
# Which one runs is decided by CMD/command, not by building two images —
# see docker-compose.yml, which runs this same image twice with different
# commands.

FROM python:3.12-slim AS base

# Prevents Python from writing .pyc files / buffering stdout — better for
# container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first, separately from app code, so a code change
# doesn't invalidate the (slow) dependency install layer on rebuild.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and data.
COPY src/ ./src/
COPY data/ ./data/
COPY main.py app.py streamlit_app.py ./

# Editable install of the search_agent package (uses pyproject.toml/src layout).
RUN pip install --no-cache-dir -e .

# Run as a non-root user rather than the container default root.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# --- IMPORTANT: API keys / secrets are NOT baked into this image. ------------
# No .env file is copied here (see .dockerignore) and no ARG/ENV sets
# ANTHROPIC_API_KEY. Secrets are injected at *container run time* only —
# via `docker run --env-file .env ...` or docker-compose's `env_file: .env`.
# This keeps secrets out of the image layers, `docker history`, and any
# registry the image might get pushed to.

EXPOSE 8000 8501

# Default: run the FastAPI backend. Overridden in docker-compose.yml for the
# frontend service (see the `command:` there).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
