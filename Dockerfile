# ============================================================
# DocuMind Dockerfile
# ============================================================
#
# WHAT IS A DOCKERFILE?
# A Dockerfile is a RECIPE for building a Docker image.
# Each instruction creates a "layer" — like transparent sheets
# stacked on top of each other.
#
# WHAT IS A DOCKER IMAGE?
# An image is a snapshot of your app + all its dependencies.
# It's like a template — you can create many containers from it.
#
# WHAT IS A CONTAINER?
# A container is a RUNNING instance of an image.
# It's like a lightweight VM — isolated but shares the OS kernel.
#
# BUILD:  docker build -t documind:latest .
# RUN:    docker run -p 5000:5000 --env-file .env documind:latest
# ============================================================

# ─── STEP 1: Choose a base image ────────────────────────
# python:3.11-slim = Python 3.11 on a minimal Debian Linux (~150MB)
# vs python:3.11 = full Debian with extra tools (~900MB)
# ALWAYS prefer slim/alpine for smaller, faster images!
FROM python:3.11-slim

# ─── STEP 2: Set working directory ──────────────────────
# All subsequent commands run from /app inside the container.
# If /app doesn't exist, Docker creates it automatically.
WORKDIR /app

# ─── STEP 3: Copy & install dependencies (CACHED!) ──────
# THIS IS THE #1 DOCKER OPTIMIZATION TRICK:
# We copy requirements.txt SEPARATELY from the rest of the code.
#
# WHY? Docker caches each layer. If requirements.txt hasn't changed,
# Docker REUSES the cached layer and skips pip install entirely.
# This saves 2-3 minutes on every rebuild!
#
# Order matters: frequently-changing files go LAST.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir = don't store pip's download cache (saves ~50MB)

# ─── STEP 4: Copy application code ──────────────────────
# This layer changes frequently (every code edit), but the
# dependency layer above stays cached. Smart ordering!
COPY . .

# ─── STEP 5: Expose port (documentation) ────────────────
# EXPOSE doesn't actually publish the port — it's metadata.
# You still need -p 5000:5000 when running the container.
# Think of it as a note saying "this app listens on port 5000".
EXPOSE 5000

# ─── STEP 6: Health check ───────────────────────────────
# Docker runs this command periodically to check if the container
# is healthy. If it fails 3x in a row → container marked "unhealthy".
# CI/CD pipelines use this to verify deployments succeeded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# ─── STEP 7: Start the application ──────────────────────
# CMD defines the default command when the container starts.
# --host 0.0.0.0 = listen on ALL interfaces (not just localhost)
# WHY? Inside Docker, "localhost" means the container itself.
# 0.0.0.0 makes the app accessible from OUTSIDE the container.
CMD ["python", "-m", "flask", "--app", "app.main", "run", "--host", "0.0.0.0", "--port", "5000"]
