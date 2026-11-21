# AI Face Vision HUD 👁️

A real-time face detection web dashboard powered by *YOLOv8, **Intel OpenVINO, **OpenCV, and **Flask*.

## ✨ Features
- *Real-Time Face Detection:* Fast inference using a YOLOv8 face model exported to OpenVINO format.
- *Cyberpunk HUD Overlay:* Custom glowing corner brackets and confidence score readouts instead of standard bounding boxes.
- *Live Telemetry Dashboard:* Displays real-time *Faces Detected* count and *Live FPS* speed.
- *Interactive Controls:*
  - *Start / Stop Camera:* Toggle the live webcam stream cleanly from the browser.
  - *Privacy Blur Mode:* Automatically blurs detected faces in real time using Gaussian blur.
  - *Snapshot Capture:* Saves high-resolution frames directly to a local snapshots/ folder.

## 🛠️ Tech Stack
- *Backend:* Python, Flask, OpenCV
- *AI / Inference:* Ultralytics YOLOv8, Intel OpenVINO
- *Frontend:* HTML5, CSS3, Vanilla JavaScript

## 🚀 How to Run Locally

1. *Clone the repository:*
   bash
   git clone https://github.com
   cd face_detection-hud
   

2. *Install dependencies:*
   bash
   pip install flask opencv-python ultralytics openvino numpy
   

3. *Run the application:*
   bash
   python app.py
   
   Your browser will automatically open at http://127.0.0.1:5000.
