
FROM python:3.9-slim

# System dependencies for face_recognition and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app


COPY . .


RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools wheel
RUN pip install dlib==19.24.2 --only-binary :all:
RUN pip install face-recognition-models==0.3.0 --only-binary :all:
RUN pip install -r requirements.txt


EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Tracking.py", "--server.port=8501", "--server.enableCORS=false"]
