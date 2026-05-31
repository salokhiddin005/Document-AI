FROM python:3.12-slim

# System dependencies for OpenCV, PDF, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libzbar0 \
    gcc \
    g++ \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements_cloud.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_cloud.txt

# Copy project
COPY . .

# Create required directories
RUN mkdir -p data fonts logs data/telegram_tmp data/feedback/crops

CMD ["python", "telegram_bot/bot.py"]
