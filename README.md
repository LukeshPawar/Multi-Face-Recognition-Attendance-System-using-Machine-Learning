# Multi-Face-Recognition-Attendance-System-using-Machine-Learning
This project is a real-time, multi-face recognition attendance system built using OpenCV and Machine Learning techniques. It captures multiple faces from a live camera feed, recognizes them, and automatically marks their attendance into a CSV file with timestamping.

---

## 📌 Features

- 🔍 **Real-time face detection and recognition**
- 👥 Detects and identifies **multiple faces simultaneously**
- 📁 Automatically logs attendance into a CSV file
- 🧠 Uses **ML-based face encodings** (like LBPH, Eigen, or deep learning embeddings)
- 🖼️ Face registration and dataset generation modules
- 📊 Streamlit-based interface (optional) for demo/testing

---

## 🧰 Technologies Used

- Python
- OpenCV (`opencv-python`)
- Face Recognition (optional: `dlib`, `face_recognition`)
- Machine Learning (LBPH, Eigenfaces, or deep embeddings)
- CMake
- PyYAML (for config handling)
- Streamlit (`streamlit==1.22.0`)

---

## 📦 Installation

Make sure Python 3.7+ is installed.

```bash
pip install opencv-python
pip install cmake
pip install pyyaml
pip install streamlit==1.22.0
```

---

## 🚀 Deploy

- Docker:
```bash
docker build -t face-attendance .
docker run -p 8501:8501 face-attendance
```

- Streamlit Cloud or similar PaaS:
  - Ensure `requirements.txt`, `packages.txt`, `runtime.txt` are present (already included)
  - Set the app entrypoint to `Tracking.py`

If you see build errors related to `dlib` or `face_recognition`, make sure system packages in `packages.txt` are installed, and the Python version matches `runtime.txt`.
