# SACHET production serving image.
# Builds the serving layer only — training happens offline and produces
# a versioned artifact (see training/model_registry.py) that this image
# loads at startup, it does not train inside the container.

FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm/catboost native libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY production/requirements.txt ./production/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r production/requirements.txt

COPY src/ /app/src/
COPY production/ /app/production/
COPY models/ /app/models/

ENV PYTHONPATH=/app/src:/app/production/governance:/app/production/feature_pipeline:/app/production/serving

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

CMD ["python", "-m", "flask", "--app", "production/serving/api.py", "run", "--host=0.0.0.0", "--port=8080"]
