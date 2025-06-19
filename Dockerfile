# Dockerfile
FROM python:3.10.15-slim

# 1) install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) copy only requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3) copy rest of your service
COPY . .

# 4) run the handler
CMD ["python", "src/handler.py"]