# Container image for Google Cloud Run (free tier)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs needed by Pillow / TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + trained artifacts (models/ and artifacts/ must exist)
COPY . .

# Cloud Run injects $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
