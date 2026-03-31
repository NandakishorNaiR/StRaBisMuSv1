
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from pathlib import Path

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Strabismus Detection", layout="centered")

st.title("👁️ Strabismus Detection System")
st.write("Upload an image to detect whether the eyes are normal or show strabismus.")

# =========================
# Load Model (cached)
# =========================
model_path = Path("models/strabismus_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# =========================
# Load Class Labels
# =========================
class_path = Path("models/class_indices.json")

with open(class_path, "r") as f:
    class_indices = json.load(f)

classes = [None] * len(class_indices)
for k, v in class_indices.items():
    classes[v] = k

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# =========================
# Prediction Function
# =========================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================
# Run Prediction
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_data = preprocess_image(img)

    # Predict
    prediction = model.predict(input_data)[0]

    # Convert to readable format
    prob_dict = {classes[i]: float(prediction[i]) * 100 for i in range(len(classes))}

    # =========================
    # Final Decision Logic
    # =========================
    normal_prob = prob_dict.get("NORMAL", 0)
    strabismus_prob = 100 - normal_prob

    if normal_prob > strabismus_prob:
        final_result = "NORMAL"
    else:
        final_result = "STRABISMUS"

    # =========================
    # Display Result
    # =========================
    st.subheader("📊 Final Result")

    if final_result == "NORMAL":
        st.success(f"Prediction: {final_result}")
    else:
        st.error(f"Prediction: {final_result}")

    st.write(f"**Normal Confidence:** {normal_prob:.2f}%")
    st.write(f"**Strabismus Confidence:** {strabismus_prob:.2f}%")

    # =========================
    # Detailed Probabilities
    # =========================
    st.subheader("🔍 Detailed Class Probabilities")

    for cls, prob in prob_dict.items():
        st.write(f"{cls}: {prob:.2f}%")

    # =========================
    # Bar Chart Visualization
    # =========================
    st.subheader("📈 Probability Distribution")
    st.bar_chart(prob_dict)

st.warning("⚠️ This is an AI-based screening tool and not a medical diagnosis.")