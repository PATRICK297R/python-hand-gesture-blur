# ✋ Hand Gesture Blur Detection

A real-time hand gesture recognition project built with **Python**, **OpenCV**, and **MediaPipe Tasks API**.

This application detects a **Peace (✌️) gesture** using your webcam (or an IP camera) and toggles a **Gaussian Blur** effect on the video stream.

The project uses the **latest MediaPipe Hand Landmarker Tasks API**, making it compatible with modern MediaPipe releases instead of the deprecated `mp.solutions` API.

---

# Features

* ✅ Real-time hand tracking
* ✅ Detect up to 2 hands simultaneously
* ✅ Draw all 21 hand landmarks
* ✅ Draw full hand skeleton connections
* ✅ Peace (✌️) gesture recognition
* ✅ Toggle blur effect using gestures
* ✅ Blur state memory (ON/OFF)
* ✅ Built-in cooldown to prevent accidental toggles
* ✅ FPS counter
* ✅ Webcam support
* ✅ IP Camera support (Android/iPhone)

---

# Demo

When the Peace gesture is detected:

* Blur toggles **ON**
* Perform the gesture again to toggle **OFF**

Information displayed on screen:

* FPS
* Current gesture
* Blur status

---

# Requirements

* Python 3.13+
* Webcam or IP Camera
* OpenCV
* MediaPipe 0.10.35+

---

# Installation

Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/hand-gesture-blur.git

cd hand-gesture-blur
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Download MediaPipe Model

This project requires the official **Hand Landmarker** model from Google.

Download:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place the downloaded file in the project folder:

```
hand-gesture-blur/
│
├── main.py
├── hand_landmarker.task
```

---

# Run

```bash
python main.py
```

---

# Controls

| Key | Action           |
| --- | ---------------- |
| Q   | Quit application |

---

# Supported Cameras

## Webcam

Default:

```python
cap = cv2.VideoCapture(0)
```

---

## IP Camera

Example:

```python
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")
```

Compatible with applications such as:

* DroidCam
* IP Webcam
* Iriun Webcam

---

# Gesture Logic

The application recognizes only the **Peace (✌️)** gesture.

Conditions:

* Index finger extended
* Middle finger extended
* Ring finger folded
* Pinky folded
* Thumb folded

Instead of simply counting fingers, this project calculates distance ratios between landmarks for improved robustness.

---

# Project Structure

```
hand-gesture-blur/
│
├── main.py
├── requirements.txt
├── README.md
├── hand_landmarker.task
└── screenshots/
```

---

# Technologies

* Python
* OpenCV
* MediaPipe Tasks API
* NumPy

---

# Customization

You can easily modify:

* Blur kernel size
* Detection confidence
* Maximum hands
* Camera source
* Gesture recognition logic
* Overlay colors
* Toggle cooldown

---

# Future Improvements

* Additional gestures
* Volume control
* Media player control
* Virtual mouse
* Virtual keyboard
* Gesture shortcuts
* Recording mode
* Face detection
* Object detection
* Multi-gesture support

---

# License

This project is released under the MIT License.

The Hand Landmarker model is provided by Google under its respective license.
