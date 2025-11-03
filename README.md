<div align="center">

<img src="docs/assets/trsyp_banner.png" alt="IEEE TRSYP 2025 - AutoEmbodAI" width="100%"/>

# TRSYP-RERO-ENICarthage

[![Docs](https://img.shields.io/badge/Documentation-MkDocs-4B8BBE?logo=mkdocs)](https://YOUR-USERNAME.github.io/TRSYP-RERO-ENICarthage/)  
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](requirements.txt)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Lane%20Detection-5C3EE8?logo=opencv)](src/)  
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Risk%20Prediction-00D4AA?logo=ultralytics)](src/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**IEEE Tunisian RAS Student and Young Professional (TRSYP) – 2nd Edition**  
**Theme**: *AutoEmbodAI* – **Smart Autonomous Vehicle Challenge**  
**ADAS Level 3+ Prototype**

</div>

---

## Overview

A **fully autonomous miniature vehicle** that demonstrates **next-generation mobility** with:

- **Autonomous Navigation** using onboard sensors  
- **AI-Powered Risk Prediction** with real-time object & sign detection  
- **Emergency Response System** for immediate threat mitigation  
- **Sensor Health Diagnostics** with onboard screen alerts  

**Fully compliant with IEEE TRSYP 2025 Specification Book**

---



## Emergency Response System

| Trigger | Action |
|-------|--------|
| `STOP` sign detected | Immediate full brake |
| Object within 20 cm | Emergency stop + reverse 10 cm |
| Risk score > 0.8 | Slow down + alert on OLED |
| Sensor failure | Safe stop + diagnostic mode |

- **Response Time**: < 100 ms from detection to motor halt  
- **Feedback**: Real-time alert on **0.96" OLED (I2C)**

---

## Sensor Health Diagnostics

| Sensor | Check | Alert |
|-------|-------|-------|
| **Camera** | Frame rate < 10 FPS | "CAMERA LAG" |
| **Ultrasonic** | Value < 2 cm or > 400 cm | "ULTRASONIC ERROR" |
| **IMU (optional)** | Sudden drift > 15° | "IMU FAILURE" |

- **Display**: All alerts shown on **OLED screen** with timestamp  
- **Logging**: Errors saved to `logs/diagnostics.log`

---

## Hardware Integration

| Component | Role |
|---------|------|
| **Raspberry Pi 4** | Main SBC – runs AI + control loop |
| **Pi Camera v2** | Vision input |
| **HC-SR04 ×2** | Obstacle detection |
| **L298N** | Dual H-bridge motor control |
| **0.96" OLED (SSD1306)** | Real-time diagnostics display |
| **DC Motors + Chassis** | Mobility |

> **Schematics**: `cad/schematics/autonomous_vehicle.kicad_pro`

---

## Software Pipeline (`src/main.py`)

```bash
Camera Feed 
    ↓
[OpenCV] Lane Detection → Steering
    ↓
[YOLOv8] Object & Sign Detection → Risk Score
    ↓
[Decision Logic] → Motor Control + Emergency Action
    ↓
[Diagnostics] → OLED Display + Logging
```

---

## Deliverables

| Item | Location |
|------|----------|
| **Codebase** | `src/` |
| **YOLO Weights** | `models/yolov8n_trained.pt` |
| **Test Video** | `data/test_drive.mp4` |
| **Technical Report** | `docs/technical_report.pdf` |
| **Schematics** | `cad/schematics/` |

---


**All features fully functional in simulation and hardware.**
