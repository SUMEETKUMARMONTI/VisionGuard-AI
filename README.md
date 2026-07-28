# 🛡️ VisionGuard AI

### Intelligent Real-Time Object Detection, Person Counting & Crowd Monitoring System using YOLOv8

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

</p>

---

# ⚡ Quick Start

Run the following commands to launch **VisionGuard AI** locally.

### 1️⃣ Open Project Folder

```powershell
cd "D:\VisionGuard AI"
```

### 2️⃣ Activate Virtual Environment (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### If using Command Prompt (CMD)

```cmd
venv\Scripts\activate
```

### 3️⃣ Start the Application

```powershell
py frontend\app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

# 📌 Overview

VisionGuard AI is an intelligent real-time surveillance system powered by **YOLOv8**, **OpenCV**, **Flask**, and **Python**.

The application detects multiple objects, counts people in real time, and monitors crowd density. When the number of detected people exceeds a configurable threshold, the system automatically triggers alerts, captures screenshots, and stores event logs.

This project demonstrates practical applications of Artificial Intelligence in:

- Smart Surveillance
- Public Safety
- Crowd Monitoring
- Security Automation
- Computer Vision

---

# 🖥️ UI / UX Screenshot

<p align="center">

<img src="screenshots/UI%20UX%20ScreenShot.png" alt="VisionGuard AI Dashboard" width="100%">

</p>

> **Modern AI-powered dashboard featuring real-time object detection, person counting, crowd monitoring, live statistics, event logging, and an intuitive dark-themed interface.**

---

# 🚀 Key Features

## 🎯 Phase 1 — Object Detection

- Detects all 80 COCO dataset object classes
- Real-time webcam detection
- High FPS YOLOv8 inference
- Confidence score display
- Bounding box visualization

---

## 👤 Phase 2 — Person Detection

- Detects only human beings
- Live person counting
- Optimized real-time performance
- Accurate people monitoring

---

## 🚨 Phase 3 — Crowd Alert System

When the detected number of people exceeds the configured limit:

- 🔴 Crowd alert notification
- 🔊 Audio beep alert
- 📸 Automatic screenshot capture
- 📝 CSV event logging
- ⚡ Live visual warning

---

# 🖥️ Modern Dashboard

The dashboard includes:

- Modern responsive UI
- Live webcam streaming
- Detection phase switching
- Camera selection
- Real-time statistics
- FPS monitoring
- Event logs
- Crowd alert notifications
- Dark professional theme

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
VisionGuard-AI/
│
├── frontend/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── models/
│   └── yolov8n.pt
│
├── outputs/
│   ├── crowd_log.csv
│   └── Alert Images/
│
├── screenshots/
│   └── UI UX ScreenShot.png
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
| OpenCV | Computer Vision |
| Flask | Web Dashboard |
| NumPy | Numerical Computing |
| Pillow | Image Processing |
| HTML5 | Frontend |
| CSS3 | Styling |
| JavaScript | Interactive UI |

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/SUMEETKUMARMONTI/VisionGuard-AI.git
cd VisionGuard-AI
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install PyTorch (CPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
python -m frontend.app
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# 🎮 Application Workflow

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
If Crowd Limit Exceeded
        │
        ▼
Alert + Screenshot + Event Log
```

---

# 📊 Generated Outputs

The system automatically generates:

- 📸 Alert screenshots
- 📝 Crowd event logs (CSV)
- 📈 Live detection statistics
- 🚨 Crowd alert notifications

---

# 🎯 Applications

- Smart City Surveillance
- Railway Stations
- Airports
- Shopping Malls
- Schools & Universities
- Industrial Safety
- Stadium Monitoring
- Public Events
- Office Security

---

# 💡 Future Enhancements

- Face Recognition
- Multi-camera Support
- DeepSORT / ByteTrack Integration
- Email Notifications
- WhatsApp Alerts
- Cloud Deployment
- Mobile Application
- Database Integration
- Heatmap Analytics
- GPU Optimization
- REST API
- Docker Support

---

# 📈 Performance

| Feature | Status |
|---------|--------|
| Real-Time Detection | ✅ |
| Object Detection | ✅ |
| Person Counting | ✅ |
| Crowd Alert | ✅ |
| Screenshot Capture | ✅ |
| CSV Logging | ✅ |
| Flask Dashboard | ✅ |
| Responsive UI | ✅ |

---

# 🏆 Project Highlights

- Intelligent AI Surveillance System
- Three Detection Phases
- Professional Dashboard UI
- Automated Crowd Alert System
- Screenshot Evidence Capture
- Event Logging
- Modular Architecture
- Clean & Maintainable Code
- Easily Extendable

---

# 👨‍💻 Skills Demonstrated

This project demonstrates knowledge of:

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Object Detection
- YOLOv8
- Python Development
- Flask Development
- OpenCV
- Frontend Development
- Real-Time Video Processing
- Software Architecture
- Problem Solving

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Software Engineer**

VisionGuard AI — Intelligent Surveillance Powered by Computer Vision

---

## ⭐ Support the Project

If you found this project useful, please consider giving it a **Star ⭐** on GitHub.

Your support helps the project reach more developers and motivates future improvements.