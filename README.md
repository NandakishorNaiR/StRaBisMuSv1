# StRaBisMuSv1 App live link: https://octane12v1-strabismus.hf.space/?__theme=system&deep_link=qjYHTZBxsAc
An ML opensource Project to check weather the human have Strabismus
# 👁️ Strabismus Detection System

## 📌 Overview

This project is an AI-based system that detects strabismus (eye misalignment) from facial images using deep learning.

It uses a Convolutional Neural Network (MobileNetV2) to analyze eye alignment and classify whether the eyes are normal or show signs of strabismus.

---

## 🚀 Features

* Detects **Normal vs Strabismus**
* Classifies into:

  * Esotropia
  * Exotropia
  * Hypertropia
  * Hypotropia
* Displays **confidence scores**
* Interactive **Streamlit web app**
* Real-time image upload and prediction

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* PIL

---

## 📂 Project Structure

```
strabismus_project/
│
├── app.py
├── eye.ipynb
├── models/
│   ├── strabismus_model.keras
│   └── class_indices.json
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

---

## 📊 How it Works

1. User uploads an image
2. Image is preprocessed
3. CNN model analyzes eye alignment
4. System outputs:

   * Normal / Strabismus
   * Confidence percentage
   * Detailed class probabilities

---

## ⚠️ Disclaimer

This system is intended for **educational and screening purposes only**.
It is **not a substitute for professional medical diagnosis**.

---

## 📌 Dataset

Dataset used from Kaggle (Strabismus image dataset).
(Not included in repository due to size constraints)

---

## 👨‍💻 Author

Your Name

---

