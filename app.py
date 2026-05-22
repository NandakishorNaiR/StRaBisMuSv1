import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_ALLOW_FLAGGING"]    = "never"
os.environ["GRADIO_TEMP_DIR"]          = "/tmp"
os.environ["GRADIO_WATCH_DIRS"]        = ""
# Tuned for HF CPU Basic: 2 vCPU / 16 GB RAM
os.environ["TF_NUM_INTRAOP_THREADS"]   = "1"
os.environ["TF_NUM_INTEROP_THREADS"]   = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]    = "0"
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["OPENBLAS_NUM_THREADS"]     = "1"

# ── Stub spaces package (Python 3.13 + codefind crash fix) ───────────────────
import sys, types
def _noop(*args, **kwargs): return args[0] if args else None
_spaces                          = types.ModuleType("spaces")
_spaces.gradio_auto_wrap         = _noop
_spaces.GPU                      = _noop
_spaces.zero                     = types.SimpleNamespace(gradio_auto_wrap=_noop)
_spaces_reload                   = types.ModuleType("spaces.reloading")
_spaces_reload.start_reload_server = lambda **kw: None
sys.modules["spaces"]            = _spaces
sys.modules["spaces.reloading"]  = _spaces_reload
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2, threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_predict_lock = threading.Lock()

# =============================================================================
# Model  —  Binary sigmoid output  (0 = NORMAL, 1 = STRABISMUS)
# Model file: models/strabismus_binary_model.keras
# Input:  224 × 224 × 3  float32  (values 0–1)
# Output: shape (1,)  sigmoid  →  probability of STRABISMUS
# =============================================================================
model = None

def load_model_safe():
    global model
    try:
        model = tf.keras.models.load_model("models/strabismus_binary_model.keras")
        # Warm-up: one dummy pass so first real user isn't slow
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
        print("Model loaded and warmed up.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        model = None

load_model_safe()

# =============================================================================
# OpenCV eye / face validation  (gate: reject non-eye images)
# =============================================================================
_EYE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _contains_human_eye(pil_img):
    """
    Returns (True, "") if human eye found, else (False, reason).
    Two-pass: face→eye first, then full-image eye scan for close-up shots.
    """
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            eyes = _EYE_CASCADE.detectMultiScale(
                gray[fy:fy+fh, fx:fx+fw], 1.1, 4, minSize=(20, 20))
            if len(eyes) > 0:
                return True, ""
        return False, (
            "Face detected but eyes not visible.\n"
            "Please ensure both eyes are open and well-lit."
        )

    eyes = _EYE_CASCADE.detectMultiScale(gray, 1.05, 6, minSize=(30, 30))
    if len(eyes) > 0:
        return True, ""

    return False, (
        "No human eyes detected in this image.\n"
        "Please upload a clear eye photo with good lighting.\n"
        "Tips: face the camera directly, keep eyes open, avoid blur."
    )

# =============================================================================
# Preprocessing
# =============================================================================
def _preprocess(pil_img):
    """Resize to 224×224, normalise to [0,1], add batch dimension."""
    if max(pil_img.size) > 512:
        pil_img.thumbnail((512, 512), Image.BILINEAR)
    arr = np.array(pil_img.resize((224, 224), Image.BILINEAR), dtype=np.float32)
    return np.expand_dims(arr / 255.0, axis=0)

# =============================================================================
# Core prediction
# =============================================================================
def _predict(pil_img, lang):
    """
    Returns (result_text, confidence_bar_fig, result_class)
    result_class: "normal" | "strabismus" | "error" | "invalid"
    """
    if model is None:
        return t(lang, "err_no_model"), None, "error"
    if pil_img is None:
        return "", None, "idle"

    try:
        img = pil_img.convert("RGB")
        if max(img.size) > 640:
            img.thumbnail((640, 640), Image.BILINEAR)

        # Gate: must contain a human eye
        ok, reason = _contains_human_eye(img)
        if not ok:
            return f"{t(lang, 'err_no_eye')}\n\n{reason}", None, "invalid"

        # Inference
        with _predict_lock:
            prob_strab = float(model.predict(_preprocess(img), verbose=0)[0][0])

        prob_normal  = 1.0 - prob_strab
        is_strabismus = prob_strab >= 0.5
        label         = "STRABISMUS" if is_strabismus else "NORMAL"
        confidence    = prob_strab * 100 if is_strabismus else prob_normal * 100
        result_class  = "strabismus" if is_strabismus else "normal"
        icon          = "⚠️" if is_strabismus else "✅"

        result_text = (
            f"{t(lang, 'pred_label')}   : {icon} {label}\n"
            f"{t(lang, 'conf_label')}   : {confidence:.1f}%\n"
            f"{t(lang, 'prob_strab')}   : {prob_strab*100:.1f}%\n"
            f"{t(lang, 'prob_normal')}  : {prob_normal*100:.1f}%"
        )

        # Confidence bar chart
        fig, ax = plt.subplots(figsize=(5, 2.8))
        labels  = [t(lang, "class_normal"), t(lang, "class_strab")]
        values  = [prob_normal * 100, prob_strab * 100]
        colors  = ["#2ecc71", "#e74c3c"]
        bars    = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel(t(lang, "chart_ylabel"), fontsize=9)
        ax.set_title(t(lang, "chart_title"), fontsize=10)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        fig.tight_layout()

        return result_text, fig, result_class

    except Exception as e:
        return f"❌ {t(lang, 'err_predict')}: {e}", None, "error"
    finally:
        plt.close("all")

# =============================================================================
# Gradio tab wrappers
# =============================================================================
def predict_upload(img, lang):
    return _predict(img, lang)

def predict_webcam(img, lang):
    if img is None:
        return t(lang, "no_snap"), None, "idle"
    return _predict(Image.fromarray(img.astype("uint8")), lang)

# =============================================================================
# Translations
# =============================================================================
LANGUAGES = {
    "English":  "en", "हिन्दी":  "hi", "मराठी":  "mr",
    "தமிழ்":    "ta", "తెలుగు": "te", "বাংলা":  "bn",
    "Español":  "es", "Français":"fr", "العربية": "ar",
    "中文":     "zh", "മലയാളം": "ml", "ಕನ್ನಡ":   "kn",
}

T = {
    "en": {
        "title":         "👁️ Strabismus Detection System",
        "subtitle":      "Upload an eye image or use your webcam to detect whether the eyes are **normal** or show signs of **strabismus**.",
        "what_title":    "### 👁️ What is Strabismus?",
        "what_body":     "**Strabismus** (crossed eyes) is a condition where both eyes do not look at the same point at the same time. One eye may turn inward, outward, upward, or downward while the other looks straight ahead.",
        "types_title":   "#### Types",
        "types_body":    "- **Esotropia** — eye turns *inward* (most common in children)\n- **Exotropia** — eye turns *outward*\n- **Hypertropia** — eye turns *upward*\n- **Hypotropia** — eye turns *downward*\n\n> Early detection matters — untreated strabismus can lead to amblyopia (lazy eye) or permanent vision loss.",
        "tab_upload":    "📁 Upload Image",
        "tab_webcam":    "📷 Webcam Capture",
        "upload_label":  "Upload Eye Image",
        "btn_upload":    "🔍 Analyze Image",
        "btn_webcam":    "🔍 Analyze Snapshot",
        "result_label":  "Result",
        "plot_label":    "Confidence Chart",
        "cam_label":     "Live Camera Feed",
        "cam_tip":       "ℹ️ Allow camera access, click the **📷 snapshot** button, then press **Analyze Snapshot**.",
        "no_snap":       "⚠️ No snapshot captured. Click the 📷 button first.",
        "disclaimer":    "⚠️ **Disclaimer:** This is an AI-based screening tool and is **NOT intended for medical diagnosis or clinical use**. Always consult a qualified ophthalmologist for accurate diagnosis and treatment.",
        "developer_info":"Developed and maintained by Knoxy Nexus",
        "lang_label":    "🌐 Language",
        "pred_label":    "Prediction",
        "conf_label":    "Confidence",
        "prob_strab":    "P(Strabismus)",
        "prob_normal":   "P(Normal)   ",
        "class_normal":  "Normal",
        "class_strab":   "Strabismus",
        "chart_title":   "Class Probabilities",
        "chart_ylabel":  "Probability (%)",
        "err_no_model":  "❌ Model not loaded. Please restart the app.",
        "err_no_eye":    "❌ Invalid Image",
        "err_predict":   "Prediction error",
    },
    "hi": {
        "title":         "👁️ स्ट्रैबिस्मस डिटेक्शन सिस्टम",
        "subtitle":      "आंखों की छवि अपलोड करें या वेबकैम का उपयोग करें।",
        "what_title":    "### 👁️ स्ट्रैबिस्मस क्या है?",
        "what_body":     "**स्ट्रैबिस्मस** (भेंगापन) — दोनों आँखें एक ही बिंदु पर नहीं देखतीं।",
        "types_title":   "#### प्रकार",
        "types_body":    "- **एसोट्रोपिया** — आँख *अंदर*\n- **एक्सोट्रोपिया** — आँख *बाहर*\n- **हाइपरट्रोपिया** — आँख *ऊपर*\n- **हाइपोट्रोपिया** — आँख *नीचे*\n\n> जल्दी पहचान ज़रूरी है।",
        "tab_upload":    "📁 छवि अपलोड", "tab_webcam": "📷 वेबकैम",
        "upload_label":  "आँख की छवि", "btn_upload": "🔍 विश्लेषण करें", "btn_webcam": "🔍 स्नैपशॉट विश्लेषण",
        "result_label":  "परिणाम", "plot_label": "आत्मविश्वास चार्ट",
        "cam_label":     "लाइव कैमरा", "cam_tip": "ℹ️ 📷 बटन दबाएं, फिर विश्लेषण करें।",
        "no_snap":       "⚠️ कोई स्नैपशॉट नहीं।",
        "disclaimer":    "⚠️ **अस्वीकरण:** यह AI स्क्रीनिंग टूल है, चिकित्सा निदान के लिए नहीं।",
        "developer_info":"नॉक्सी नेक्सस द्वारा विकसित",
        "lang_label":    "🌐 भाषा",
        "pred_label":    "भविष्यवाणी", "conf_label": "आत्मविश्वास",
        "prob_strab":    "P(स्ट्रैबिस्मस)", "prob_normal": "P(सामान्य)    ",
        "class_normal":  "सामान्य", "class_strab": "स्ट्रैबिस्मस",
        "chart_title":   "वर्ग संभावनाएँ", "chart_ylabel": "संभावना (%)",
        "err_no_model":  "❌ मॉडल लोड नहीं हुआ।", "err_no_eye": "❌ अमान्य छवि", "err_predict": "भविष्यवाणी त्रुटि",
    },
    "mr": {
        "title":         "👁️ स्ट्रॅबिस्मस डिटेक्शन सिस्टम",
        "subtitle":      "डोळ्याची प्रतिमा अपलोड करा किंवा वेबकॅम वापरा।",
        "what_title":    "### 👁️ स्ट्रॅबिस्मस म्हणजे काय?",
        "what_body":     "**स्ट्रॅबिस्मस** (तिरळे डोळे) — एक डोळा आत, बाहेर, वर किंवा खाली वळतो.",
        "types_title":   "#### प्रकार",
        "types_body":    "- **एसोट्रोपिया** — आत\n- **एक्सोट्रोपिया** — बाहेर\n- **हायपरट्रोपिया** — वर\n- **हायपोट्रोपिया** — खाली\n\n> लवकर ओळख महत्त्वाची.",
        "tab_upload":    "📁 प्रतिमा अपलोड", "tab_webcam": "📷 वेबकॅम",
        "upload_label":  "डोळ्याची प्रतिमा", "btn_upload": "🔍 विश्लेषण करा", "btn_webcam": "🔍 स्नॅपशॉट विश्लेषण",
        "result_label":  "निकाल", "plot_label": "विश्वास चार्ट",
        "cam_label":     "लाइव्ह कॅमेरा", "cam_tip": "ℹ️ 📷 बटण दाबा नंतर विश्लेषण करा.",
        "no_snap":       "⚠️ स्नॅपशॉट नाही.",
        "disclaimer":    "⚠️ **अस्वीकरण:** वैद्यकीय निदानासाठी नाही.",
        "developer_info":"नॉक्सी नेक्सस द्वारे विकसित",
        "lang_label":    "🌐 भाषा",
        "pred_label":    "भविष्यवाणी", "conf_label": "आत्मविश्वास",
        "prob_strab":    "P(स्ट्रॅबिस्मस)", "prob_normal": "P(सामान्य)      ",
        "class_normal":  "सामान्य", "class_strab": "स्ट्रॅबिस्मस",
        "chart_title":   "वर्ग संभाव्यता", "chart_ylabel": "संभाव्यता (%)",
        "err_no_model":  "❌ मॉडेल लोड झाले नाही.", "err_no_eye": "❌ अवैध प्रतिमा", "err_predict": "भविष्यवाणी त्रुटी",
    },
    "ta": {
        "title":         "👁️ ஸ்ட்ரபிஸ்மஸ் கண்டறிதல் அமைப்பு",
        "subtitle":      "கண் படத்தை பதிவேற்றவும் அல்லது வெப்கேமை பயன்படுத்தவும்.",
        "what_title":    "### 👁️ ஸ்ட்ரபிஸ்மஸ் என்றால் என்ன?",
        "what_body":     "**ஸ்ட்ரபிஸ்மஸ்** — ஒரு கண் உள்ளே, வெளியே, மேலே அல்லது கீழே திரும்பும்.",
        "types_title":   "#### வகைகள்",
        "types_body":    "- **எசோட்ரோபியா** — உள்ளே\n- **எக்சோட்ரோபியா** — வெளியே\n- **ஹைபர்ட்ரோபியா** — மேலே\n- **ஹைபோட்ரோபியா** — கீழே",
        "tab_upload":    "📁 படம் பதிவேற்று", "tab_webcam": "📷 வெப்கேம்",
        "upload_label":  "கண் படம்", "btn_upload": "🔍 பகுப்பாய்வு", "btn_webcam": "🔍 பகுப்பாய்வு",
        "result_label":  "முடிவு", "plot_label": "நம்பிக்கை விளக்கப்படம்",
        "cam_label":     "நேரடி கேமரா", "cam_tip": "ℹ️ 📷 அழுத்தி பகுப்பாய்வு செய்யவும்.",
        "no_snap":       "⚠️ ஸ்னாப்ஷாட் இல்லை.",
        "disclaimer":    "⚠️ **மறுப்பு:** மருத்துவ நோயறிதலுக்காக அல்ல.",
        "developer_info":"Knoxy Nexus ஆல் உருவாக்கப்பட்டது",
        "lang_label":    "🌐 மொழி",
        "pred_label":    "கணிப்பு", "conf_label": "நம்பிக்கை",
        "prob_strab":    "P(ஸ்ட்ரபிஸ்மஸ்)", "prob_normal": "P(சாதாரணம்)   ",
        "class_normal":  "சாதாரணம்", "class_strab": "ஸ்ட்ரபிஸ்மஸ்",
        "chart_title":   "வகுப்பு நிகழ்தகவுகள்", "chart_ylabel": "நிகழ்தகவு (%)",
        "err_no_model":  "❌ மாதிரி ஏற்றப்படவில்லை.", "err_no_eye": "❌ தவறான படம்", "err_predict": "கணிப்பு பிழை",
    },
    "te": {
        "title":         "👁️ స్ట్రాబిస్మస్ డిటెక్షన్ సిస్టమ్",
        "subtitle":      "కంటి చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా వెబ్‌క్యామ్ ఉపయోగించండి.",
        "what_title":    "### 👁️ స్ట్రాబిస్మస్ అంటే ఏమిటి?",
        "what_body":     "**స్ట్రాబిస్మస్** — ఒక కన్ను లోపలికి, బయటికి, పైకి లేదా కిందికి తిరగవచ్చు.",
        "types_title":   "#### రకాలు",
        "types_body":    "- **ఎసోట్రోపియా** — లోపలికి\n- **ఎక్సోట్రోపియా** — బయటికి\n- **హైపర్‌ట్రోపియా** — పైకి\n- **హైపోట్రోపియా** — కిందికి",
        "tab_upload":    "📁 అప్‌లోడ్", "tab_webcam": "📷 వెబ్‌క్యామ్",
        "upload_label":  "కంటి చిత్రం", "btn_upload": "🔍 విశ్లేషించు", "btn_webcam": "🔍 విశ్లేషించు",
        "result_label":  "ఫలితం", "plot_label": "విశ్వాస చార్ట్",
        "cam_label":     "లైవ్ కెమెరా", "cam_tip": "ℹ️ 📷 నొక్కి విశ్లేషించండి.",
        "no_snap":       "⚠️ స్నాప్‌షాట్ లేదు.",
        "disclaimer":    "⚠️ **నిరాకరణ:** వైద్య నిర్ధారణ కోసం కాదు.",
        "developer_info":"Knoxy Nexus ద్వారా అభివృద్ధి",
        "lang_label":    "🌐 భాష",
        "pred_label":    "అంచనా", "conf_label": "విశ్వాసం",
        "prob_strab":    "P(స్ట్రాబిస్మస్)", "prob_normal": "P(సాధారణం)    ",
        "class_normal":  "సాధారణం", "class_strab": "స్ట్రాబిస్మస్",
        "chart_title":   "తరగతి సంభావ్యతలు", "chart_ylabel": "సంభావ్యత (%)",
        "err_no_model":  "❌ మోడల్ లోడ్ కాలేదు.", "err_no_eye": "❌ చెల్లని చిత్రం", "err_predict": "అంచనా లోపం",
    },
    "bn": {
        "title":         "👁️ স্ট্র্যাবিসমাস ডিটেকশন সিস্টেম",
        "subtitle":      "চোখের ছবি আপলোড করুন বা ওয়েবক্যাম ব্যবহার করুন।",
        "what_title":    "### 👁️ স্ট্র্যাবিসমাস কী?",
        "what_body":     "**স্ট্র্যাবিসমাস** — একটি চোখ ভেতরে, বাইরে, উপরে বা নিচে ঘোরে।",
        "types_title":   "#### ধরন",
        "types_body":    "- **এসোট্রোপিয়া** — ভেতরে\n- **এক্সোট্রোপিয়া** — বাইরে\n- **হাইপারট্রোপিয়া** — উপরে\n- **হাইপোট্রোপিয়া** — নিচে",
        "tab_upload":    "📁 আপলোড", "tab_webcam": "📷 ওয়েবক্যাম",
        "upload_label":  "চোখের ছবি", "btn_upload": "🔍 বিশ্লেষণ", "btn_webcam": "🔍 বিশ্লেষণ",
        "result_label":  "ফলাফল", "plot_label": "আস্থা চার্ট",
        "cam_label":     "লাইভ ক্যামেরা", "cam_tip": "ℹ️ 📷 বোতাম চাপুন তারপর বিশ্লেষণ করুন।",
        "no_snap":       "⚠️ স্ন্যাপশট নেই।",
        "disclaimer":    "⚠️ **দাবিত্যাগ:** চিকিৎসা নির্ণয়ের জন্য নয়।",
        "developer_info":"Knoxy Nexus দ্বারা উন্নত",
        "lang_label":    "🌐 ভাষা",
        "pred_label":    "পূর্বাভাস", "conf_label": "আস্থা",
        "prob_strab":    "P(স্ট্র্যাবিসমাস)", "prob_normal": "P(স্বাভাবিক)   ",
        "class_normal":  "স্বাভাবিক", "class_strab": "স্ট্র্যাবিসমাস",
        "chart_title":   "শ্রেণী সম্ভাবনা", "chart_ylabel": "সম্ভাবনা (%)",
        "err_no_model":  "❌ মডেল লোড হয়নি।", "err_no_eye": "❌ অবৈধ ছবি", "err_predict": "পূর্বাভাস ত্রুটি",
    },
    "es": {
        "title":         "👁️ Sistema de Detección de Estrabismo",
        "subtitle":      "Sube una imagen del ojo o usa la cámara web.",
        "what_title":    "### 👁️ ¿Qué es el Estrabismo?",
        "what_body":     "**Estrabismo** — un ojo puede girar hacia adentro, afuera, arriba o abajo.",
        "types_title":   "#### Tipos",
        "types_body":    "- **Esotropia** — adentro\n- **Exotropia** — afuera\n- **Hipertropia** — arriba\n- **Hipotropia** — abajo",
        "tab_upload":    "📁 Subir Imagen", "tab_webcam": "📷 Cámara Web",
        "upload_label":  "Imagen del ojo", "btn_upload": "🔍 Analizar", "btn_webcam": "🔍 Analizar",
        "result_label":  "Resultado", "plot_label": "Gráfico de Confianza",
        "cam_label":     "Cámara en Vivo", "cam_tip": "ℹ️ Haz clic en 📷 y luego analiza.",
        "no_snap":       "⚠️ Sin foto capturada.",
        "disclaimer":    "⚠️ **Aviso:** No para diagnóstico médico.",
        "developer_info":"Desarrollado por Knoxy Nexus",
        "lang_label":    "🌐 Idioma",
        "pred_label":    "Predicción", "conf_label": "Confianza",
        "prob_strab":    "P(Estrabismo)", "prob_normal": "P(Normal)     ",
        "class_normal":  "Normal", "class_strab": "Estrabismo",
        "chart_title":   "Probabilidades", "chart_ylabel": "Probabilidad (%)",
        "err_no_model":  "❌ Modelo no cargado.", "err_no_eye": "❌ Imagen inválida", "err_predict": "Error de predicción",
    },
    "fr": {
        "title":         "👁️ Système de Détection du Strabisme",
        "subtitle":      "Téléchargez une image ou utilisez la webcam.",
        "what_title":    "### 👁️ Qu'est-ce que le Strabisme ?",
        "what_body":     "**Strabisme** — un œil se tourne vers l'intérieur, l'extérieur, le haut ou le bas.",
        "types_title":   "#### Types",
        "types_body":    "- **Ésotropie** — intérieur\n- **Exotropie** — extérieur\n- **Hypertropie** — haut\n- **Hypotropie** — bas",
        "tab_upload":    "📁 Télécharger", "tab_webcam": "📷 Webcam",
        "upload_label":  "Image de l'œil", "btn_upload": "🔍 Analyser", "btn_webcam": "🔍 Analyser",
        "result_label":  "Résultat", "plot_label": "Graphique de Confiance",
        "cam_label":     "Caméra en Direct", "cam_tip": "ℹ️ Cliquez sur 📷 puis analyser.",
        "no_snap":       "⚠️ Pas de photo capturée.",
        "disclaimer":    "⚠️ **Avertissement:** Non destiné au diagnostic médical.",
        "developer_info":"Développé par Knoxy Nexus",
        "lang_label":    "🌐 Langue",
        "pred_label":    "Prédiction", "conf_label": "Confiance",
        "prob_strab":    "P(Strabisme)", "prob_normal": "P(Normal)    ",
        "class_normal":  "Normal", "class_strab": "Strabisme",
        "chart_title":   "Probabilités", "chart_ylabel": "Probabilité (%)",
        "err_no_model":  "❌ Modèle non chargé.", "err_no_eye": "❌ Image invalide", "err_predict": "Erreur de prédiction",
    },
    "ar": {
        "title":         "👁️ نظام الكشف عن الحول",
        "subtitle":      "قم بتحميل صورة للعين أو استخدم الكاميرا.",
        "what_title":    "### 👁️ ما هو الحول؟",
        "what_body":     "**الحول** — تنحرف إحدى العينين للداخل أو الخارج أو الأعلى أو الأسفل.",
        "types_title":   "#### الأنواع",
        "types_body":    "- **داخلي** — للداخل\n- **خارجي** — للخارج\n- **علوي** — للأعلى\n- **سفلي** — للأسفل",
        "tab_upload":    "📁 تحميل", "tab_webcam": "📷 كاميرا",
        "upload_label":  "صورة العين", "btn_upload": "🔍 تحليل", "btn_webcam": "🔍 تحليل",
        "result_label":  "النتيجة", "plot_label": "مخطط الثقة",
        "cam_label":     "الكاميرا المباشرة", "cam_tip": "ℹ️ انقر على 📷 ثم تحليل.",
        "no_snap":       "⚠️ لا توجد لقطة.",
        "disclaimer":    "⚠️ **إخلاء:** غير مخصص للتشخيص الطبي.",
        "developer_info":"تم التطوير بواسطة Knoxy Nexus",
        "lang_label":    "🌐 اللغة",
        "pred_label":    "التنبؤ", "conf_label": "الثقة",
        "prob_strab":    "P(الحول)   ", "prob_normal": "P(طبيعي)   ",
        "class_normal":  "طبيعي", "class_strab": "حول",
        "chart_title":   "احتمالات الفئة", "chart_ylabel": "الاحتمال (%)",
        "err_no_model":  "❌ النموذج غير محمل.", "err_no_eye": "❌ صورة غير صالحة", "err_predict": "خطأ في التنبؤ",
    },
    "zh": {
        "title":         "👁️ 斜视检测系统",
        "subtitle":      "上传眼睛图片或使用摄像头检测斜视。",
        "what_title":    "### 👁️ 什么是斜视？",
        "what_body":     "**斜视** — 一只眼睛可能向内、向外、向上或向下偏转。",
        "types_title":   "#### 类型",
        "types_body":    "- **内斜视** — 向内\n- **外斜视** — 向外\n- **上斜视** — 向上\n- **下斜视** — 向下",
        "tab_upload":    "📁 上传", "tab_webcam": "📷 摄像头",
        "upload_label":  "眼睛图片", "btn_upload": "🔍 分析", "btn_webcam": "🔍 分析",
        "result_label":  "结果", "plot_label": "置信度图",
        "cam_label":     "实时摄像头", "cam_tip": "ℹ️ 点击 📷 然后分析。",
        "no_snap":       "⚠️ 没有快照。",
        "disclaimer":    "⚠️ **免责声明：** 不用于医疗诊断。",
        "developer_info":"由 Knoxy Nexus 开发",
        "lang_label":    "🌐 语言",
        "pred_label":    "预测", "conf_label": "置信度",
        "prob_strab":    "P(斜视)  ", "prob_normal": "P(正常)  ",
        "class_normal":  "正常", "class_strab": "斜视",
        "chart_title":   "类别概率", "chart_ylabel": "概率 (%)",
        "err_no_model":  "❌ 模型未加载。", "err_no_eye": "❌ 无效图片", "err_predict": "预测错误",
    },
    "ml": {
        "title":         "👁️ സ്ട്രാബിസ്മസ് നിർണ്ണയ സംവിധാനം",
        "subtitle":      "കണ്ണിന്റെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക അല്ലെങ്കിൽ വെബ്‌ക്യാം ഉപയോഗിക്കുക.",
        "what_title":    "### 👁️ എന്താണ് സ്ട്രാബിസ്മസ്?",
        "what_body":     "**സ്ട്രാബിസ്മസ്** — ഒരു കണ്ണ് ഉള്ളിലേക്കോ, പുറത്തേക്കോ, മുകളിലേക്കോ, താഴേക്കോ തിരിഞ്ഞിരിക്കും.",
        "types_title":   "#### തരങ്ങൾ",
        "types_body":    "- **എസോട്രോപ്പിയ** — ഉള്ളിലേക്ക്\n- **എക്സോട്രോപ്പിയ** — പുറത്തേക്ക്\n- **ഹൈപ്പർട്രോപ്പിയ** — മുകളിലേക്ക്\n- **ഹൈപ്പോട്രോപ്പിയ** — താഴേക്ക്",
        "tab_upload":    "📁 അപ്‌ലോഡ്", "tab_webcam": "📷 വെബ്‌ക്യാം",
        "upload_label":  "കണ്ണിന്റെ ചിത്രം", "btn_upload": "🔍 പരിശോധിക്കുക", "btn_webcam": "🔍 പരിശോധിക്കുക",
        "result_label":  "ഫലം", "plot_label": "ആത്മവിശ്വാസ ചാർട്ട്",
        "cam_label":     "ലൈവ് ക്യാമറ", "cam_tip": "ℹ️ 📷 ബട്ടൺ ക്ലിക്ക് ചെയ്ത് പരിശോധിക്കുക.",
        "no_snap":       "⚠️ സ്നാപ്പ്ഷോട്ട് ഇല്ല.",
        "disclaimer":    "⚠️ **നിരാകരണം:** വൈദ്യ നിർണ്ണയത്തിനല്ല.",
        "developer_info":"Knoxy Nexus വികസിപ്പിച്ചത്",
        "lang_label":    "🌐 ഭാഷ",
        "pred_label":    "പ്രവചനം", "conf_label": "ആത്മവിശ്വാസം",
        "prob_strab":    "P(സ്ട്രാബിസ്മസ്)", "prob_normal": "P(സാധാരണം)    ",
        "class_normal":  "സാധാരണം", "class_strab": "സ്ട്രാബിസ്മസ്",
        "chart_title":   "ക്ലാസ് സാധ്യതകൾ", "chart_ylabel": "സാധ്യത (%)",
        "err_no_model":  "❌ മോഡൽ ലോഡ് ആയില്ല.", "err_no_eye": "❌ അസാധുവായ ചിത്രം", "err_predict": "പ്രവചന പിഴവ്",
    },
    "kn": {
        "title":         "👁️ ಓರೆಗಣ್ಣು ಪತ್ತೆ ವ್ಯವಸ್ಥೆ",
        "subtitle":      "ಕಣ್ಣಿನ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಅಥವಾ ವೆಬ್‌ಕ್ಯಾಮ್ ಬಳಸಿ.",
        "what_title":    "### 👁️ ಓರೆಗಣ್ಣು ಎಂದರೇನು?",
        "what_body":     "**ಓರೆಗಣ್ಣು** — ಒಂದು ಕಣ್ಣು ಒಳಗೆ, ಹೊರಗೆ, ಮೇಲಕ್ಕೆ ಅಥವಾ ಕೆಳಕ್ಕೆ ತಿರುಗಿರಬಹುದು.",
        "types_title":   "#### ವಿಧಗಳು",
        "types_body":    "- **ಈಸೋಟ್ರೋಪಿಯಾ** — ಒಳಮುಖ\n- **ಎಕ್ಸೋಟ್ರೋಪಿಯಾ** — ಹೊರಮುಖ\n- **ಹೈಪರ್‌ಟ್ರೋಪಿಯಾ** — ಮೇಲ್ಮುಖ\n- **ಹೈಪೋಟ್ರೋಪಿಯಾ** — ಕೆಳಮುಖ",
        "tab_upload":    "📁 ಅಪ್‌ಲೋಡ್", "tab_webcam": "📷 ವೆಬ್‌ಕ್ಯಾಮ್",
        "upload_label":  "ಕಣ್ಣಿನ ಚಿತ್ರ", "btn_upload": "🔍 ವಿಶ್ಲೇಷಿಸಿ", "btn_webcam": "🔍 ವಿಶ್ಲೇಷಿಸಿ",
        "result_label":  "ಫಲಿತಾಂಶ", "plot_label": "ವಿಶ್ವಾಸ ಚಾರ್ಟ್",
        "cam_label":     "ಲೈವ್ ಕ್ಯಾಮೆರಾ", "cam_tip": "ℹ️ 📷 ಕ್ಲಿಕ್ ಮಾಡಿ ವಿಶ್ಲೇಷಿಸಿ.",
        "no_snap":       "⚠️ ಸ್ನ್ಯಾಪ್‌ಶಾಟ್ ಇಲ್ಲ.",
        "disclaimer":    "⚠️ **ನಿರಾಕರಣೆ:** ವೈದ್ಯಕೀಯ ನಿರ್ಣಯಕ್ಕಲ್ಲ.",
        "developer_info":"Knoxy Nexus ಅಭಿವೃದ್ಧಿ",
        "lang_label":    "🌐 ಭಾಷೆ",
        "pred_label":    "ಭವಿಷ್ಯವಾಣಿ", "conf_label": "ವಿಶ್ವಾಸ",
        "prob_strab":    "P(ಓರೆಗಣ್ಣು)", "prob_normal": "P(ಸಾಮಾನ್ಯ)   ",
        "class_normal":  "ಸಾಮಾನ್ಯ", "class_strab": "ಓರೆಗಣ್ಣು",
        "chart_title":   "ಸಂಭವನೀಯತೆಗಳು", "chart_ylabel": "ಸಂಭವನೀಯತೆ (%)",
        "err_no_model":  "❌ ಮಾದರಿ ಲೋಡ್ ಆಗಿಲ್ಲ.", "err_no_eye": "❌ ಅಮಾನ್ಯ ಚಿತ್ರ", "err_predict": "ಭವಿಷ್ಯವಾಣಿ ದೋಷ",
    },
}

def t(lang_name, key):
    code = LANGUAGES.get(lang_name, "en")
    return T.get(code, T["en"]).get(key, T["en"][key])

# =============================================================================
# Language switcher  — must match outputs list exactly (16 items)
# =============================================================================
def change_language(lang):
    return (
        gr.update(value=f"# {t(lang,'title')}\n{t(lang,'subtitle')}"),
        gr.update(value=(t(lang,"what_title") + "\n\n" + t(lang,"what_body") +
                         "\n\n" + t(lang,"types_title") + "\n" + t(lang,"types_body"))),
        # upload tab
        gr.update(label=t(lang,"upload_label")),
        gr.update(value=t(lang,"btn_upload")),
        gr.update(label=t(lang,"result_label")),
        gr.update(label=t(lang,"plot_label")),
        # webcam tab
        gr.update(label=t(lang,"cam_label")),
        gr.update(value=t(lang,"cam_tip")),
        gr.update(value=t(lang,"btn_webcam")),
        gr.update(label=t(lang,"result_label")),
        gr.update(label=t(lang,"plot_label")),
        # footer
        gr.update(value=f"---\n> {t(lang,'disclaimer')}\n\n*{t(lang,'developer_info')}*"),
    )

# =============================================================================
# Gradio UI
# =============================================================================
with gr.Blocks(title="👁️ Strabismus Detection System") as demo:

    # ── Header + language selector ───────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4):
            header_md = gr.Markdown(f"# {t('English','title')}\n{t('English','subtitle')}")
        with gr.Column(scale=1, min_width=160):
            lang_dd = gr.Dropdown(
                choices=list(LANGUAGES.keys()), value="English",
                label=t("English","lang_label"), interactive=True,
            )

    # ── Info section ─────────────────────────────────────────────────────────
    info_md = gr.Markdown(
        t("English","what_title") + "\n\n" + t("English","what_body") + "\n\n" +
        t("English","types_title") + "\n" + t("English","types_body")
    )
    gr.Markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # Tab 1 — Upload
        with gr.TabItem(t("English","tab_upload")):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        type="pil", label=t("English","upload_label"), sources=["upload"])
                    upload_btn   = gr.Button(t("English","btn_upload"), variant="primary")
                with gr.Column(scale=1):
                    upload_result = gr.Textbox(label=t("English","result_label"), lines=5)
                    upload_plot   = gr.Plot(label=t("English","plot_label"))

            upload_btn.click(
                fn=predict_upload, inputs=[upload_input, lang_dd],
                outputs=[upload_result, upload_plot, gr.State()])
            upload_input.change(
                fn=predict_upload, inputs=[upload_input, lang_dd],
                outputs=[upload_result, upload_plot, gr.State()])

        # Tab 2 — Webcam
        with gr.TabItem(t("English","tab_webcam")):
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        type="numpy", label=t("English","cam_label"),
                        sources=["webcam"], streaming=False)
                    cam_tip_md   = gr.Markdown(t("English","cam_tip"))
                    webcam_btn   = gr.Button(t("English","btn_webcam"), variant="primary")
                with gr.Column(scale=1):
                    webcam_result = gr.Textbox(label=t("English","result_label"), lines=5)
                    webcam_plot   = gr.Plot(label=t("English","plot_label"))

            webcam_btn.click(
                fn=predict_webcam, inputs=[webcam_input, lang_dd],
                outputs=[webcam_result, webcam_plot, gr.State()])
            webcam_input.change(
                fn=predict_webcam, inputs=[webcam_input, lang_dd],
                outputs=[webcam_result, webcam_plot, gr.State()])

    # ── Footer ───────────────────────────────────────────────────────────────
    disclaimer_md = gr.Markdown(
        f"---\n> {t('English','disclaimer')}\n\n*{t('English','developer_info')}*"
    )

    # ── Wire language switcher ───────────────────────────────────────────────
    lang_dd.change(
        fn=change_language, inputs=[lang_dd],
        outputs=[
            header_md, info_md,
            upload_input, upload_btn, upload_result, upload_plot,
            webcam_input, cam_tip_md, webcam_btn, webcam_result, webcam_plot,
            disclaimer_md,
        ],
    )

# =============================================================================
# Launch
# =============================================================================
demo.queue(max_size=100, default_concurrency_limit=2)
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, show_error=True)
