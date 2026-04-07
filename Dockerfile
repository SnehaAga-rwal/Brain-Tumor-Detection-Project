# Optional: deploy to Azure App Service (Containers), ACI, or ACR + Container Apps
# TensorFlow + OpenCV need system libs; image is large (~several GB).

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Writable SQLite / uploads (App Service mounts /home; adjust DATABASE_URL if needed)
RUN mkdir -p instance app/static/uploads logs

EXPOSE 8000

# One worker: each worker loads TensorFlow + models (high RAM)
CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 300 --workers 1 --threads 2 run:app
