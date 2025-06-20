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
