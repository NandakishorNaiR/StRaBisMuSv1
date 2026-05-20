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



⚙️ Installation (Local Setup)
1️⃣ Clone Repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
python app.py
🌐 Deployment

The application is deployed using:

Gradio
Hugging Face Spaces
🔗 Live Application

https://octane12v1-strabismus.hf.space/

🧪 How the System Works
User uploads image or captures webcam snapshot
OpenCV validates face and eye presence
Image is resized and normalized
TensorFlow model predicts class probabilities
Highest probability class is selected using Argmax
Application displays:
Final Prediction
Predicted Class
Confidence Score
Probability Distribution Graph
📊 Example Output
Prediction   : ⚠️ STRABISMUS DETECTED
Class        : HYPOTROPIA
Confidence   : 82.45%

The system also generates a probability graph for all classes.

📷 Input Modes
📁 Upload Image

Users can upload facial or eye images directly from local storage.

📷 Webcam Capture

Users can capture live snapshots using browser webcam support.

🛠️ Tech Stack
Programming & AI
Python
TensorFlow
Keras
NumPy
Computer Vision & Image Processing
OpenCV
Pillow (PIL)
Visualization
Matplotlib
Interface & Deployment
Gradio
Hugging Face Spaces
Version Control
Git
GitHub
📌 Dataset
Dataset Source

https://www.kaggle.com/datasets/druthvikvarma/strabismus-dataset

Classes Used
NORMAL
ESOTROPIA
EXOTROPIA
HYPERTROPIA
HYPOTROPIA

Dataset is not included in the repository due to size limitations.

⚡ Performance Optimizations

The application includes several deployment and runtime optimizations:

TensorFlow CPU optimization
Controlled threading configuration
Concurrent request handling
Warmed-up model inference
Hugging Face Spaces compatibility patches
Memory-safe graph rendering
Reduced CPU overhead

These optimizations improve stability for multiple concurrent users.

⚠️ Limitations
Not intended for clinical diagnosis
Accuracy depends on image quality
Blurry or low-light images may reduce prediction reliability
Dataset diversity limitations
Sensitive to improper face positioning
⭐ Future Improvements
Real-time video stream detection
Eye landmark detection
OpenCV blur detection
Automatic eye-region cropping
Larger medical datasets
Attention-based deep learning models
Mobile application integration
Clinical validation support
Real-time eye tracking
Telemedicine integration
⚠️ Disclaimer

This is an AI-based screening and educational tool.

It is NOT intended for:

Medical diagnosis
Clinical treatment
Professional healthcare replacement

Please consult a qualified ophthalmologist or healthcare professional for proper diagnosis and treatment.

👨‍💻 Author

Nandakishore Nair

📚 References
TensorFlow Documentation
Keras Documentation
OpenCV Documentation
Gradio Documentation
Hugging Face Spaces Documentation
MobileNetV2 Research Paper
Kaggle Dataset Resources
