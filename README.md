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

## 🌐 Live Demo

https://octane12v1-strabismus.hf.space/

## 📄 Documentation

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

This project demonstrates the practical use of AI in medical image classification and preliminary healthcare screening.

---

# 🚀 Features

* 🔍 Detects **Normal vs Strabismus**
* 🧠 Multi-class eye condition classification
* 📊 Displays confidence-based predictions
* 📈 Generates probability distribution graphs
* 📷 Supports **Webcam Capture**
* 📁 Supports **Image Upload**
* ⚡ Interactive Gradio UI
* 🌐 Deployed on Hugging Face Spaces
* 🛡️ Safe model loading and error handling
* 🧹 Memory-safe matplotlib rendering

---

# 🧠 Model Details

| Parameter        | Value                       |
| ---------------- | --------------------------- |
| Model            | MobileNetV2                 |
| Framework        | TensorFlow / Keras          |
| Technique        | Transfer Learning           |
| Input Size       | 224 × 224                   |
| Output Classes   | 5                           |
| Prediction Logic | Argmax-based Classification |

---

# 📂 Project Structure

```text
Strabismus/
│
├── app.py
├── requirements.txt
├── models/
│   ├── strabismus_model.keras
│   └── class_indices.json
├── eye.ipynb
├── README.md
```

---

# ⚙️ Installation (Local Setup)

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run Application

```bash
python app.py
```

---

# 🌐 Deployment

The application is deployed using:

* Gradio
* Hugging Face Spaces

### 🔗 Live App

https://octane12v1-strabismus.hf.space/

---

# 🧪 How the System Works

1. User uploads image or captures webcam snapshot
2. Image is resized and normalized
3. Deep learning model predicts probabilities
4. Highest probability class is selected using **Argmax**
5. Application displays:

   * Final prediction
   * Predicted class
   * Confidence score
   * Probability graph

---

# 📊 Example Output

```text
Prediction   : ⚠️ STRABISMUS DETECTED
Class        : HYPOTROPIA
Confidence   : 82.45%
```

The system also generates a probability distribution graph for all classes.

---

# 📷 Input Modes

## 📁 Upload Image

Users can upload eye or facial images directly from local storage.

## 📷 Webcam Capture

Users can capture live snapshots using browser webcam support.

---

# 🛠️ Tech Stack

## Programming & AI

* Python
* TensorFlow
* Keras
* NumPy

## Visualization & Image Processing

* Matplotlib
* Pillow (PIL)

## Deployment & Interface

* Gradio
* Hugging Face Spaces

## Version Control

* Git
* GitHub

---

# 📌 Dataset

Dataset Source:
https://www.kaggle.com/datasets/druthvikvarma/strabismus-dataset

### Classes Used

* NORMAL
* ESOTROPIA
* EXOTROPIA
* HYPERTROPIA
* HYPOTROPIA

> Dataset is not included in the repository due to size limitations.

---

# ⚠️ Limitations

* Not intended for clinical diagnosis
* Accuracy depends on image quality
* Blurry or low-light images may reduce prediction reliability
* Limited by dataset diversity and size

---

# ⭐ Future Improvements

* Real-time video stream detection
* Eye landmark detection
* OpenCV-based blur detection
* Automatic eye-region cropping
* Improved dataset diversity
* Mobile application integration
* Clinical validation support
* Attention-based deep learning models

---

# ⚠️ Disclaimer

This is an AI-based screening and educational tool.

It is **NOT intended for:**

* Medical diagnosis
* Clinical treatment
* Professional healthcare replacement

Please consult a qualified ophthalmologist or healthcare professional for proper diagnosis and treatment.

---

# 👨‍💻 Author

**Nandakishore Nair**

---

# 📚 References

* TensorFlow Documentation
* Keras Documentation
* Gradio Documentation
* Hugging Face Spaces Documentation
* Kaggle Dataset Resources

---
