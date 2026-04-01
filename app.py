import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib.pyplot as plt

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model("models/strabismus_model.keras")

# =========================
# Load Class Labels
# =========================
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)

classes = [None] * len(class_indices)
for k, v in class_indices.items():
    classes[v] = k

# =========================
# Preprocess Function
# =========================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================
# Prediction Function
# =========================

def predict(img):
    img = img.convert("RGB")
    input_data = preprocess_image(img)

    prediction = model.predict(input_data)[0]

    # Probabilities
    prob_dict = {classes[i]: float(prediction[i]) for i in range(len(classes))}

    # ✅ Correct Decision: Argmax
    predicted_index = int(np.argmax(prediction))
    predicted_class = classes[predicted_index]

    if predicted_class == "NORMAL":
        final_result = "NORMAL"
    else:
        final_result = "STRABISMUS"

    # Confidence of predicted class
    confidence = prediction[predicted_index] * 100

    # Graph
    labels = list(prob_dict.keys())
    values = [v * 100 for v in prob_dict.values()]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=30)
    plt.ylabel("Probability (%)")
    plt.title("Class Probability Distribution")

    result_text = f"""
Prediction: {final_result}
Predicted Class: {predicted_class}
Confidence: {confidence:.2f}%
"""

    return result_text, prob_dict, plt



# =========================
# Gradio UI
# =========================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Class Probabilities"),
        gr.Plot(label="Probability Graph")
    ],
    title="👁️ Strabismus Detection System",
    description="Upload an image to detect whether the eyes are normal or show strabismus.",
)

# =========================
# Disclaimer
# =========================
gr.Markdown(
"""
⚠️ **Disclaimer:**  
This is an AI-based screening tool and is **NOT intended for medical diagnosis or clinical use**.  
Please consult a qualified medical professional for accurate diagnosis.
"""
)

# =========================
# Launch (FIXED)
# =========================
interface.launch(share=False, debug=False)
