import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
os.environ["GRADIO_TEMP_DIR"] = "/tmp"
os.environ["GRADIO_WATCH_DIRS"] = ""

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Load Model
# =========================
model = None
classes = []

def load_model_safe():
    global model, classes
    try:
        model = tf.keras.models.load_model("models/strabismus_model.keras")
        with open("models/class_indices.json", "r") as f:
            class_indices = json.load(f)
        classes = [None] * len(class_indices)
        for k, v in class_indices.items():
            classes[v] = k
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None
        classes = []

load_model_safe()

# =========================
# Shared Preprocess
# =========================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================
# Shared Prediction Core
# =========================
def run_prediction(img):
    """
    Accepts a PIL Image.
    Returns (result_text, prob_dict, fig)
    """
    if model is None:
        return "❌ Model not loaded. Please restart the app.", {}, None
    if img is None:
        return "⚠️ No image provided.", {}, None

    try:
        img = img.convert("RGB")
        input_data = preprocess_image(img)
        prediction = model.predict(input_data)[0]

        prob_dict = {classes[i]: float(prediction[i]) for i in range(len(classes))}
        predicted_index = int(np.argmax(prediction))
        predicted_class = classes[predicted_index]

        final_result = "✅ NORMAL" if predicted_class == "NORMAL" else "⚠️ STRABISMUS DETECTED"
        confidence = prediction[predicted_index] * 100

        fig, ax = plt.subplots(figsize=(5, 3))
        labels = list(prob_dict.keys())
        values = [v * 100 for v in prob_dict.values()]
        colors = ["#2ecc71" if l == "NORMAL" else "#e74c3c" for l in labels]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Probability (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Class Probability Distribution", fontsize=10)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8
            )
        fig.tight_layout()

        result_text = (
            f"Prediction   : {final_result}\n"
            f"Class        : {predicted_class}\n"
            f"Confidence   : {confidence:.2f}%"
        )
        return result_text, prob_dict, fig

    except Exception as e:
        return f"❌ Prediction error: {str(e)}", {}, None

    finally:
        plt.close("all")

# =========================
# Wrappers per tab
# =========================
def predict_upload(img):
    return run_prediction(img)

def predict_webcam(img):
    """img comes in as numpy array from gr.Image(type='numpy')"""
    if img is None:
        return "⚠️ No snapshot captured. Click the 📷 capture button first.", {}, None
    pil_img = Image.fromarray(img.astype("uint8"))
    return run_prediction(pil_img)

# =========================
# CSS
# =========================
CUSTOM_CSS = """
#title-md { text-align: center; }
#title-md h1 { font-size: 2rem; }
.tab-nav button { font-size: 1rem; font-weight: 600; }
.result-box textarea { font-family: monospace; font-size: 0.95rem; }
#cam-tip { font-size: 0.85rem; color: #888; margin-top: 4px; }
"""

# =========================
# Gradio UI
# =========================
with gr.Blocks(title="👁️ Strabismus Detection System") as demo:

    gr.Markdown(
        """
        # 👁️ Strabismus Detection System
        Detect whether the eyes are **normal** or show signs of **strabismus** — via image upload or live webcam.
        """,
        elem_id="title-md"
    )

    with gr.Tabs():

        # ── Tab 1 : Upload ──────────────────────────────────────────────
        with gr.TabItem("📁 Upload Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        type="pil",
                        label="Upload Eye Image",
                        sources=["upload"],
                    )
                    upload_btn = gr.Button("🔍 Analyze Uploaded Image", variant="primary")

                with gr.Column(scale=1):
                    upload_result = gr.Textbox(label="Result", lines=4, elem_classes="result-box")
                    upload_probs  = gr.Label(label="Class Probabilities")
                    upload_plot   = gr.Plot(label="Probability Graph")

            upload_btn.click(
                fn=predict_upload,
                inputs=upload_input,
                outputs=[upload_result, upload_probs, upload_plot],
            )
            upload_input.change(
                fn=predict_upload,
                inputs=upload_input,
                outputs=[upload_result, upload_probs, upload_plot],
            )

        # ── Tab 2 : Webcam ──────────────────────────────────────────────
        with gr.TabItem("📷 Webcam Capture"):
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        type="numpy",
                        label="Live Camera Feed",
                        sources=["webcam"],    # webcam-only source
                        streaming=False,        # snapshot mode, not live stream
                    )
                    gr.Markdown(
                        "ℹ️ Allow camera access when prompted. "
                        "Click the **📷 snapshot** button inside the camera box, "
                        "then press **Analyze Snapshot** below.",
                        elem_id="cam-tip"
                    )
                    webcam_btn = gr.Button("🔍 Analyze Snapshot", variant="primary")

                with gr.Column(scale=1):
                    webcam_result = gr.Textbox(label="Result", lines=4, elem_classes="result-box")
                    webcam_probs  = gr.Label(label="Class Probabilities")
                    webcam_plot   = gr.Plot(label="Probability Graph")

            webcam_btn.click(
                fn=predict_webcam,
                inputs=webcam_input,
                outputs=[webcam_result, webcam_probs, webcam_plot],
            )
            # Auto-predict when user takes a snapshot
            webcam_input.change(
                fn=predict_webcam,
                inputs=webcam_input,
                outputs=[webcam_result, webcam_probs, webcam_plot],
            )

    # ── Disclaimer ─────────────────────────────────────────────────────
    gr.Markdown(
        """
        ---
        > ⚠️ **Disclaimer:** This is an AI-based screening tool and is **NOT intended for medical
        > diagnosis or clinical use**. Always consult a qualified ophthalmologist or medical
        > professional for accurate diagnosis and treatment.
        """
    )

# =========================
# Launch
# =========================
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
    show_error=True,
    css=CUSTOM_CSS,
)
