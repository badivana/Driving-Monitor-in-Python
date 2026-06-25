# <h1 align="center">🚗 Driver Drowsiness Detection System</h1>

<p align="center">
Real-time Driver Monitoring System using <b>Python</b>, <b>OpenCV</b> and <b>MediaPipe FaceMesh</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-blue">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green">
<img src="https://img.shields.io/badge/MediaPipe-FaceMesh-orange">
<img src="https://img.shields.io/badge/License-MIT-yellow">
<img src="https://img.shields.io/badge/Status-Active-success">
</p>

---

## 📸 Project Preview

> Replace `images/banner.png` with your project screenshot.

<p align="center">
<img src="images/banner.png" width="900">
</p>

---

# 🎥 Demo

<p align="center">
<img src="demo/demo.gif" width="800">
</p>

---

# 📑 Table of Contents

* Project Overview
* Features
* Technologies Used
* Project Workflow
* Detection Parameters
* Performance
* Limitations
* Project Structure
* Installation
* Usage
* How It Works
* Libraries
* Roadmap
* Future Improvements
* Contributing
* License

---

# 📌 Project Overview

Driver fatigue is one of the leading causes of road accidents worldwide. This project provides a **real-time Driver Monitoring System (DMS)** that continuously analyzes the driver's facial landmarks using **MediaPipe FaceMesh**.

The system estimates the driver's alertness using:

* 👁 Eye Aspect Ratio (EAR)
* 😮 Mouth Aspect Ratio (MAR)
* 😴 Percentage of Eye Closure (PERCLOS)
* 🧠 Head Pose Estimation

If signs of drowsiness persist beyond predefined thresholds, the system automatically detects the driver's fatigue state and can trigger an alert.

---

# ✨ Features

* ✅ Real-time webcam monitoring
* ✅ Face detection using MediaPipe FaceMesh
* ✅ 468 facial landmark detection
* ✅ Eye Aspect Ratio (EAR) calculation
* ✅ Mouth Aspect Ratio (MAR) calculation
* ✅ Percentage of Eye Closure (PERCLOS)
* ✅ Head Pose estimation
* ✅ Driver state classification
* ✅ Lightweight and fast
* ✅ Modular Python implementation
* ✅ Easy to customize thresholds

---

# 📸 Screenshots

## Face Detection

<p align="center">
<img src="images/face_detection.png" width="700">
</p>

---

## Facial Landmarks

<p align="center">
<img src="images/landmarks.png" width="700">
</p>

---

## Driver Alert

<p align="center">
<img src="images/alert.png" width="700">
</p>

---

# 🛠 Technologies Used

* Python
* OpenCV
* MediaPipe FaceMesh
* NumPy

---

# 🧠 Project Workflow

```text
            Webcam
               │
               ▼
       Face Detection
               │
               ▼
    MediaPipe FaceMesh
      (468 Landmarks)
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
    EAR       MAR   Head Pose
      │        │        │
      └────────┼────────┘
               ▼
      Driver State Analysis
               ▼
      Drowsiness Detection
               ▼
          Driver Alert
```

---

# 📐 Detection Parameters

## 👁 Eye Aspect Ratio (EAR)

Measures eye openness.

Lower EAR values indicate prolonged eye closure and possible drowsiness.

---

## 😮 Mouth Aspect Ratio (MAR)

Measures mouth opening.

Higher MAR values indicate yawning.

---

## 😴 PERCLOS

Percentage of Eye Closure over time.

PERCLOS is one of the most reliable fatigue indicators used in modern Driver Monitoring Systems.

---

## 🧠 Head Pose Estimation

Tracks the driver's head orientation.

Detects excessive downward or sideways head movement indicating fatigue or distraction.

---

# 📊 Performance

* Real-time Processing
* Approximately **25–30 FPS**
* Detects **468 facial landmarks**
* Low latency
* Lightweight implementation

---

# ⚠️ Limitations

* Works with one driver at a time
* Requires sufficient lighting
* Performance decreases if the face is heavily occluded
* Requires a webcam

---

# 📁 Project Structure

```text
Driving-Monitor-in-Python/
│
├── calibration/
├── detection/
├── demo/
│   └── demo.gif
├── images/
│   ├── banner.png
│   ├── face_detection.png
│   ├── landmarks.png
│   └── alert.png
├── main.py
├── state.py
├── utils.py
├── face_landmarker.task
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🚀 Installation

## Clone the repository

```bash
git clone https://github.com/badivana/Driving-Monitor-in-Python.git

cd Driving-Monitor-in-Python
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the application

```bash
python main.py
```

The application will:

* Open your webcam
* Detect facial landmarks
* Calculate Eye Aspect Ratio (EAR)
* Calculate Mouth Aspect Ratio (MAR)
* Estimate Head Pose
* Monitor driver fatigue
* Display alerts when drowsiness is detected

Press **Esc** to exit.

---

# ⚙️ How It Works

The Driver Monitoring System consists of three major components.

## Face Detection

Uses **MediaPipe FaceMesh** to detect **468 facial landmarks** in real time.

---

## Head Pose Estimation

Calculates the driver's head orientation using facial landmarks.

---

## Driver State Classification

Combines:

* Eye Aspect Ratio (EAR)
* Mouth Aspect Ratio (MAR)
* PERCLOS
* Head Pose

to classify the driver as:

🟢 Alert

🔴 Drowsy

---

# 📚 Libraries

* OpenCV
* MediaPipe
* NumPy

---

# 🗺 Roadmap

* [x] Face Detection
* [x] Eye Aspect Ratio (EAR)
* [x] Mouth Aspect Ratio (MAR)
* [x] Head Pose Estimation
* [x] Driver State Classification
* [x] Drowsiness Detection
* [ ] Audio Alarm
* [ ] Mobile Notifications
* [ ] Raspberry Pi Deployment
* [ ] TensorFlow Lite Optimization
* [ ] Multi-person Detection
* [ ] Streamlit Dashboard

---

# 🚀 Future Improvements

* Audio warning system
* Mobile notification support
* TensorFlow Lite optimization
* Raspberry Pi deployment
* Infrared camera support
* Driver identification
* Performance benchmarking
* Cloud logging
* AI-based fatigue prediction

---

# 🤝 Contributing

Contributions are welcome.

If you find a bug or have an improvement, feel free to:

* Fork the repository
* Create a new branch
* Commit your changes
* Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
