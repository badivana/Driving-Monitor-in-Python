# 🚗 Driver Drowsiness Detection System using Python

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

A real-time **Driver Monitoring System (DMS)** built with **Python**, **OpenCV**, and **MediaPipe FaceMesh** to detect driver drowsiness using facial landmarks. The system continuously analyzes eye closure, mouth movement, and head posture to identify fatigue and alert the driver.

---

## 🎥 Demo

> Add your demo GIF inside the `demo/` folder.

![Demo](demo/demo.gif)

---

# 📑 Table of Contents

* Project Overview
* Features
* Technologies Used
* Project Workflow
* Detection Parameters
* Project Structure
* Installation
* Usage
* How It Works
* Future Improvements
* License

---

# 📌 Project Overview

Driver fatigue is one of the leading causes of road accidents worldwide. This project aims to reduce accidents by continuously monitoring the driver's facial features through a webcam.

The system estimates the driver's alertness using:

* 👁 Eye Aspect Ratio (EAR)
* 😮 Mouth Aspect Ratio (MAR)
* 😴 Percentage of Eye Closure (PERCLOS)
* 🧠 Head Pose Estimation

If signs of drowsiness persist beyond predefined thresholds, the system alerts the driver.

---

# ✨ Features

* ✅ Real-time webcam monitoring
* ✅ Face detection using MediaPipe FaceMesh
* ✅ Eye Aspect Ratio (EAR) calculation
* ✅ Mouth Aspect Ratio (MAR) calculation
* ✅ Head pose estimation
* ✅ Driver state classification
* ✅ Drowsiness alert system
* ✅ Lightweight and fast
* ✅ Modular Python implementation

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
      ┌───────┼────────┐
      ▼       ▼        ▼
    EAR      MAR   Head Pose
      │       │        │
      └───────┼────────┘
              ▼
    Driver State Analysis
              ▼
      Drowsiness Alert
```

---

# 📐 Detection Parameters

### 👁 EAR (Eye Aspect Ratio)

Measures eye openness.

Lower EAR values indicate prolonged eye closure, which may suggest drowsiness.

---

### 😮 MAR (Mouth Aspect Ratio)

Measures mouth opening.

High MAR values indicate yawning.

---

### 😴 PERCLOS

Percentage of Eye Closure over time.

One of the most reliable fatigue indicators used in modern Driver Monitoring Systems.

---

### 🧠 Head Pose

Tracks the driver's head orientation.

Detects excessive downward or sideways head movement that may indicate fatigue or distraction.

---

# 📁 Project Structure

```text
Driving-Monitor-in-Python/
│
├── calibration/
├── detection/
├── demo/
│   └── demo.gif
├── main.py
├── state.py
├── utils.py
├── requirements.txt
├── face_landmarker.task
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
* Monitor eye closure
* Detect yawning
* Estimate head pose
* Alert the driver when drowsiness is detected

Press **Esc** to exit.

---

# ⚙️ How It Works

The system consists of three major components.

## Face Detection

Uses **MediaPipe FaceMesh** to detect **468 facial landmarks** in real time.

---

## Head Pose Estimation

Estimates the orientation of the driver's head to identify distraction or fatigue.

---

## Driver State Classifier

Combines:

* Eye Aspect Ratio (EAR)
* Mouth Aspect Ratio (MAR)
* PERCLOS
* Head Pose

to classify the driver as:

* 🟢 Alert
* 🔴 Drowsy

---

# 📚 Libraries

* OpenCV
* MediaPipe
* NumPy

---

# 🚀 Future Improvements

* Audio alarm
* Mobile notifications
* TensorFlow Lite optimization
* Raspberry Pi deployment
* Infrared camera support
* Driver identification
* Performance benchmarking
* Multi-person detection
* Streamlit dashboard

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork the repository, open issues, or submit pull requests.

---

# 📄 License

This project is licensed under the MIT License.
