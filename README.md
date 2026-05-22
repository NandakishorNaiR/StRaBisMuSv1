
---
title: Strabismus Detection System
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.0.0"
python_version: "3.13"
app_file: app.py
pinned: false
---

Live Demo- https://octane12v1-strabismus.hf.space/

# 👁️ Strabismus Detection System

An AI-powered eye screening tool that detects whether eyes are **normal** or show signs of **strabismus** (crossed/misaligned eyes).

## Model
- Architecture: Custom Binary CNN (Conv2D × 3 → Dense → Sigmoid)
- Input: 224 × 224 × 3
- Output: Single sigmoid probability → NORMAL (< 0.5) / STRABISMUS (≥ 0.5)
- File: `models/strabismus_binary_model.keras`

## Features
- 📁 Upload an eye image for analysis
- 📷 Webcam live capture and analysis
- 🌐 12 languages supported
- 👁️ OpenCV face/eye validation (rejects non-eye images)
- 📊 Confidence probability chart

## Pipeline
```
Image → OpenCV Validation → Binary CNN → NORMAL / STRABISMUS
```

## ⚠️ Disclaimer
This is an AI-based screening tool and is **NOT intended for medical diagnosis or clinical use**.
Always consult a qualified ophthalmologist for accurate diagnosis and treatment.

*Developed and maintained by Knoxy Nexus*
