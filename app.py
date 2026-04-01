
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import json

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
# Preprocess
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

    prob_dict = {classes[i]: float(prediction[i]) * 100 for i in range(len(classes))}

    normal_prob = prob_dict.get("NORMAL", 0)
    strabismus_prob = 100 - normal_prob

    final_result = "NORMAL" if normal_prob > strabismus_prob else "STRABISMUS"

    result_text = f"""
    Prediction: {final_result}
    
    Normal Confidence: {normal_prob:.2f}%
    Strabismus Confidence: {strabismus_prob:.2f}%
    """

    return result_text, prob_dict

# =========================
# Gradio UI
# =========================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Class Probabilities")
    ],
    title="👁️ Strabismus Detection System",
    description="Upload an image to detect whether the eyes are normal or show strabismus.",
)

interface.launch()

