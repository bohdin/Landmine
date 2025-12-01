FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_DISABLE_MPS_FALLBACK=1

# системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- 2. Ставимо залежності проєкту ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 3. Gunicorn окремо (бо у req.txt його нема) ---
RUN pip install --no-cache-dir gunicorn

# --- 4. Копіюємо код ---
COPY . .

EXPOSE 8000

# --- 5. Довший timeout і preload OFF (бо модель вантажиться довго) ---
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "web.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "180"]
