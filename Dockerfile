FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_DISABLE_MPS_FALLBACK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Спочатку залежності, щоб кешувались
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Код і моделі (очікується, що models/*.pth|onnx у контексті збірки)
COPY . .

EXPOSE 8000

# Gunicorn + uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "web.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "180"]
