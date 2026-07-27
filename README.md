# 🛡️ VisionGuard AI
### Intelligent Real-Time Object Detection, Person Counting & Crowd Monitoring System using YOLOv8

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

</p>

---
To Start This System Make Sure These Commands running on the terminal

PS D:\VisionGuard AI> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "d:\VisionGuard AI\venv\Scripts\Activate.ps1")
(venv) PS D:\VisionGuard AI>  py frontend\app.py

---

---

## 📌 Overview

**VisionGuard AI** is a real-time computer vision surveillance system built using **YOLOv8**, **Python**, **OpenCV**, and **Flask**.

The application detects multiple objects, counts people in real time, and intelligently monitors crowd density. When the number of detected people exceeds a predefined threshold, the system automatically triggers an alert, captures evidence, and stores event logs.

The project demonstrates practical applications of Artificial Intelligence in:

- Smart Surveillance
- Public Safety
- Crowd Monitoring
- Security Automation
- Computer Vision

---

# 🚀 Key Features

### 🎯 Phase 1 — Object Detection
- Detects all COCO dataset objects (80 classes)
- Real-time webcam detection
- High FPS inference using YOLOv8
- Displays confidence score and bounding boxes

---

### 👤 Phase 2 — Person Detection
- Detects only human beings
- Live person counting
- Accurate people tracking
- Optimized for real-time monitoring

---

### 🚨 Phase 3 — Crowd Alert System

If the detected number of people exceeds a configurable limit:

- 🔴 Alert is triggered
- 🔊 Beep sound is played
- 📸 Screenshot is automatically captured
- 📝 Event is stored inside CSV log
- ⚡ Live alert notification appears on screen

---

# 🖥️ Modern Dashboard

The project includes a responsive Flask-based dashboard featuring:

- Beautiful modern UI
- Live camera streaming
- Real-time statistics
- Phase switching
- Animated interface
- Professional typography
- Dark theme

---

# 🏗️ Project Architecture

```
                Webcam
                   │
                   ▼
          Video Frame Capture
                   │
                   ▼
             YOLOv8 Inference
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
 Object Detection Person Count Crowd Analysis
        │          │          │
        └──────────┼──────────┘
                   ▼
          Threshold Verification
                   │
          ┌────────┴────────┐
          ▼                 ▼
     Normal Mode      Alert Triggered
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
      Save Image         CSV Logging       Audio Alert
```

---

# 📂 Project Structure

```
VisionGuard AI/
│
├── frontend/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── models/
│   ├── yolov8n.pt
│   └── yolov8n.onnx
│
├── outputs/
│   ├── crowd_log.csv
│   └── Alert Images
│
├── src/
│   ├── phase1_object_detection.py
│   ├── phase2_person_detection.py
│   └── phase3_crowd_alert.py
│
├── requirements.txt
└── README.md
```

---

# 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| YOLOv8 | Object Detection |
| OpenCV | Image Processing |
| Flask | Web Dashboard |
| NumPy | Numerical Operations |
| Pillow | Image Handling |
| HTML5 | Frontend |
| CSS3 | Styling |
| JavaScript | Interactive UI |

---

# ⚙️ Installation

## 1 Clone Repository

```bash
git clone https://github.com/yourusername/VisionGuard-AI.git

cd VisionGuard-AI
```

---

## 2 Create Virtual Environment

```bash
python -m venv venv
```

---

## 3 Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4 Install PyTorch (CPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 5 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
python -m frontend.app
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 🎮 Workflow

```
Start Application
        │
        ▼
Choose Camera
        │
        ▼
Select Detection Phase
        │
        ▼
Run Live Detection
        │
        ▼
Display Results
        │
        ▼
If Crowd Limit Crossed
        │
        ▼
Alert + Screenshot + Log
```

---

# 📊 Output

The system automatically generates:

- 📸 Alert Images
- 📝 Crowd Logs (CSV)
- 📈 Live Detection Statistics
- 🚨 Alert Notifications

---

# 🎯 Applications

- Smart City Surveillance
- Railway Stations
- Airports
- Shopping Malls
- Schools & Universities
- Industrial Safety
- Stadium Crowd Monitoring
- Public Events
- Office Security

---

# 💡 Future Enhancements

- Face Recognition
- Multi-camera Support
- Object Tracking (ByteTrack/DeepSORT)
- Email & SMS Alerts
- WhatsApp Notifications
- Cloud Deployment
- Mobile App Integration
- Database Support
- Heatmap Analytics
- GPU Acceleration
- REST API
- Docker Deployment

---

# 📈 Performance

| Feature | Status |
|---------|--------|
| Real-Time Detection | ✅ |
| Person Counting | ✅ |
| Crowd Alert | ✅ |
| Screenshot Capture | ✅ |
| CSV Logging | ✅ |
| Flask Dashboard | ✅ |
| Responsive UI | ✅ |

---

# 🏆 Highlights

- Real-time AI Surveillance System
- Multi-phase Detection Pipeline
- Professional Dashboard
- Automated Crowd Alerts
- Evidence Capture
- Event Logging
- Modular Architecture
- Clean Code Structure
- Easily Extendable

---

# 👨‍💻 Skills Demonstrated

This project showcases expertise in:

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Object Detection
- Python Development
- Flask Development
- OpenCV
- YOLOv8
- Real-Time Video Processing
- Software Architecture
- Frontend Development
- Problem Solving

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👤 Author

**Software Engineer**

**VisionGuard AI — Intelligent Surveillance Powered by Computer Vision**

---

## ⭐ If you found this project useful, consider giving it a Star!

A ⭐ helps the project gain visibility and motivates future development.

