# Real-Time Object Detection using YOLOv8

A real-time computer vision application built with **Python, OpenCV, NumPy, and Ultralytics YOLOv8** for detecting and classifying objects through a webcam.

The application processes live video frames, detects objects using the YOLOv8 Nano model, and displays bounding boxes, class labels, and confidence scores in real time.

---

## 📸 Detection Results

> Add your best object-detection screenshot here.
<img width="1920" height="1080" alt="Screenshot (10)" src="https://github.com/user-attachments/assets/7e89671c-2a5f-425e-abee-68e310ec6999" />


![YOLOv8 Object Detection](assets/detection-result.png)

---

## 🎥 Demo

> https://github.com/sumankr62800-del/object-detection/blob/712099797418a447cff2ae9ca61e338f4c631d5e/Recording%202026-08-14%20193122.mp4

The application performs real-time object detection using a webcam and provides an interactive OpenCV window with fullscreen support.

---

## 🎯 Project Objective

The objective of this project is to implement a practical real-time object detection system using a pretrained YOLOv8 model.

The system demonstrates how modern deep-learning-based computer vision models can be integrated with OpenCV to process live camera input and identify multiple objects simultaneously.

---

## ✨ Features

- 🔍 Real-time object detection
- 📷 Webcam-based video processing
- 🧠 YOLOv8 Nano pretrained model
- 🏷️ Object class and confidence score display
- 📦 Custom bounding boxes
- 🎨 Consistent class-based bounding-box colors
- 🖥️ Resizable OpenCV detection window
- ⛶ Fullscreen toggle using `F`
- ❌ Exit using `Q` or `ESC`
- ⚡ Lightweight YOLOv8 Nano model for efficient inference

---

## 🧠 How It Works

The application follows this pipeline:

```text
Webcam
   ↓
Capture Video Frame
   ↓
YOLOv8 Object Detection
   ↓
Identify Objects
   ↓
Calculate Confidence Scores
   ↓
Draw Bounding Boxes & Labels
   ↓
Display Detection Result
