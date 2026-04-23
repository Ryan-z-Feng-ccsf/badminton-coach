# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11.9-slim AS builder

WORKDIR /backend
# Install dependencies into a separate layer for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip\
&& pip install --no-cache-dir -r requirements.txt
# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11.9-slim 

WORKDIR /backend

# Copy installed packages from builder (keeps final image small)
COPY  --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy code
COPY . .
RUN mkdir -p data/audio/ data/raw_videos/

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
