# 👁️ Strabismus Detection System
https://octane12v1-strabismus.hf.space/?__theme=system&deep_link=IsNLDynXtt8

## 📌 Overview

This project is an AI-powered system that detects **strabismus (eye misalignment)** from facial images using deep learning.

The model analyzes eye alignment and classifies the image into:

* Normal
* Esotropia
* Exotropia
* Hypertropia
* Hypotropia

The system then determines whether the person has **normal vision or strabismus**.

---

## 🚀 Features

* 🔍 Detects **Normal vs Strabismus**
* 🧠 Multi-class classification of eye conditions
* 📊 Displays **class probabilities**
* 📈 Visualizes predictions with **probability graph**
* 🌐 Deployed using Gradio (Hugging Face Spaces)
* ⚡ Fast and interactive UI

---

## 🧠 Model Details

* Model: MobileNetV2 (Transfer Learning)
* Framework: TensorFlow / Keras
* Input Size: 224 × 224
* Output: 5 classes

---

## 📂 Project Structure

```
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

## ⚙️ Installation (Local Setup)

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
python app.py
```

---

## 🌐 Deployment

This project is deployed using **Gradio on Hugging Face Spaces**.

👉 Live Demo: https://octane12v1-strabismus.hf.space/?__theme=system&deep_link=IsNLDynXtt8

---

## 🧪 How It Works

1. Upload an image
2. Image is preprocessed (resized and normalized)
3. Model predicts probabilities for each class
4. System selects the **highest probability class (argmax)**
5. Output:

   * Final prediction (Normal / Strabismus)
   * Predicted class
   * Confidence score
   * Probability graph

---

## 📊 Example Output

* Prediction: NORMAL
* Confidence: 92.45%
* Graph showing class probabilities

---

## ⚠️ Disclaimer

This is an AI-based screening tool and is **NOT intended for medical diagnosis**.
Please consult a qualified medical professional for accurate evaluation.

---

## 📌 Dataset

Dataset used from Kaggle :- https://www.kaggle.com/datasets/druthvikvarma/strabismus-dataset
(Not included in repository due to size constraints)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Gradio
* NumPy
* Pillow
* Matplotlib

---

## 👨‍💻 Author

Nandakishore Nair

---

## ⭐ Future Improvements

* Real-time webcam detection
* Mobile app integration
* Improved dataset and accuracy
* Medical-grade validation

---
