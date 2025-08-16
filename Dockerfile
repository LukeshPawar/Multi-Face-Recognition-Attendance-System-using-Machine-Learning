
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for dlib/face_recognition and OpenCV (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Tracking.py", "--server.port=8501", "--server.enableCORS=false"]
