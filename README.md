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

# 👁️ Strabismus Detection System

AI-powered eye alignment screening application built using Deep Learning, TensorFlow, OpenCV, and Gradio for preliminary strabismus detection and classification.

---

# 🌐 Live Demo

https://octane12v1-strabismus.hf.space/

---

# 📄 Documentation

https://drive.google.com/file/d/16w4agjpw9y0Z8e-mzaP_H1Sc-p76pqF5/view?usp=sharing

---

# 📌 Overview

The **Strabismus Detection System** is an AI-powered healthcare screening application developed using **Deep Learning** and **Computer Vision** techniques.

The system analyzes eye alignment from facial images and predicts whether the eyes are:

* NORMAL
* ESOTROPIA
* EXOTROPIA
* HYPERTROPIA
* HYPOTROPIA

The application then determines whether the person has:

* ✅ Normal Eye Alignment
* ⚠️ Strabismus Detected

This project demonstrates the practical implementation of Artificial Intelligence in medical image classification and preliminary healthcare screening.

---

# 🚀 Features

## 🔍 Detection Features

* Detects **Normal vs Strabismus**
* Multi-class eye condition classification
* Confidence-based prediction system
* Argmax-based final prediction logic
* Probability distribution graph visualization

---

## 📷 Input Features

* 📁 Image Upload Support
* 📷 Webcam Snapshot Capture
* Browser Camera Integration
* Live Snapshot Prediction

---

## 🌐 Accessibility Features

* Multi-language Interface Support
* User-friendly AI screening workflow
* Interactive Gradio UI

### Supported Languages

* English
* हिन्दी (Hindi)
* मराठी (Marathi)
* தமிழ் (Tamil)
* తెలుగు (Telugu)
* বাংলা (Bengali)
* Español (Spanish)
* Français (French)
* العربية (Arabic)
* 中文 (Chinese)
* മലയാളം (Malayalam)
* ಕನ್ನಡ (Kannada)

---

## 🛡️ AI Safety & Reliability Features

* OpenCV-based eye validation
* Human face and eye detection
* Invalid image rejection
* Safe TensorFlow model loading
* Warm-up inference optimization
* Thread-safe prediction handling
* Memory-safe matplotlib rendering
* CPU optimization for Hugging Face Spaces

---

# 🧠 Model Details

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Architecture Type | CNN (Convolutional Neural Network) |
| Technique | Transfer Learning |
| Input Size | 224 × 224 |
| Output Classes | 5 |
| Prediction Logic | Argmax-based Classification |
| Deployment | Gradio + Hugging Face Spaces |

---

# 🧠 AI Concepts Used

* Deep Learning
* CNN (Convolutional Neural Networks)
* Transfer Learning
* Image Classification
* Feature Extraction
* Softmax Classification
* Computer Vision

---

# 📂 Project Structure

```text
Strabismus/
│
├── app.py
├── requirements.txt
├── README.md
├── eye.ipynb
│
├── models/
│   ├── strabismus_model.keras
│   └── class_indices.json
