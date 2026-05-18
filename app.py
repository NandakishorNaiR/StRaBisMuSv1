import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
os.environ["GRADIO_TEMP_DIR"] = "/tmp"
os.environ["GRADIO_WATCH_DIRS"] = ""
# Tuned for HF CPU Basic: 2 vCPU / 16 GB RAM
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"

# ── Patch: stub out the entire `spaces` package before gradio imports it ────
# Problem: Python 3.13 + spaces/_vendor/codefind crashes on TF _DictWrapper.
# Gradio also calls spaces.gradio_auto_wrap() — so we must stub the full
# spaces module with all attributes gradio expects, not just spaces.reloading.
import sys, types

def _noop(*args, **kwargs):
    # passthrough wrapper — returns the function unchanged
    return args[0] if args else None

_spaces = types.ModuleType("spaces")
_spaces.gradio_auto_wrap  = _noop   # called by gradio block_function.py
_spaces.GPU               = _noop   # decorator used in some HF demos
_spaces.zero              = types.SimpleNamespace(gradio_auto_wrap=_noop)

_spaces_reload = types.ModuleType("spaces.reloading")
_spaces_reload.start_reload_server = lambda **kw: None

sys.modules["spaces"]           = _spaces
sys.modules["spaces.reloading"] = _spaces_reload
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import json, cv2, threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Thread-safe lock ───────────────────────────────────────────────────────────
_predict_lock = threading.Lock()

# =============================================================================
# Translations
# =============================================================================
LANGUAGES = {
    "English":    "en",
    "हिन्दी":      "hi",
    "मराठी":       "mr",
    "தமிழ்":       "ta",
    "తెలుగు":      "te",
    "বাংলা":       "bn",
    "Español":    "es",
    "Français":   "fr",
    "العربية":    "ar",
    "中文":        "zh",
    "മലയാളം":   "ml",
}

T = {
    "en": {
        "title":          "👁️ Strabismus Detection System",
        "subtitle":       "Upload an eye image or use your webcam to detect whether the eyes are **normal** or show signs of **strabismus**.",
        "what_title":     "### 👁️ What is Strabismus?",
        "what_body":      "**Strabismus** (crossed eyes) is a condition where both eyes do not look at the same point at the same time. One eye may turn inward, outward, upward, or downward while the other looks straight ahead.",
        "types_title":    "#### Types",
        "types_body":     "- **Esotropia** — eye turns *inward* (most common in children)\n- **Exotropia** — eye turns *outward*\n- **Hypertropia** — eye turns *upward*\n- **Hypotropia** — eye turns *downward*\n\n> Early detection matters — untreated strabismus can lead to amblyopia (lazy eye) or permanent vision loss.",
        "tab_upload":     "📁 Upload Image",
        "tab_webcam":     "📷 Webcam Capture",
        "upload_label":   "Upload Eye Image",
        "analyze_upload": "🔍 Analyze Uploaded Image",
        "result_label":   "Result",
        "probs_label":    "Class Probabilities",
        "plot_label":     "Probability Graph",
        "cam_label":      "Live Camera Feed",
        "cam_tip":        "ℹ️ Allow camera access when prompted. Click the **📷 snapshot** button inside the camera box, then press **Analyze Snapshot**.",
        "analyze_webcam": "🔍 Analyze Snapshot",
        "no_snap":        "⚠️ No snapshot captured. Click the 📷 capture button first.",
        "disclaimer":     "⚠️ **Disclaimer:** This is an AI-based screening tool and is **NOT intended for medical diagnosis or clinical use**. Always consult a qualified ophthalmologist or medical professional for accurate diagnosis and treatment.",
        "chart_title":    "Class Probability Distribution",
        "chart_ylabel":   "Probability (%)",
        "lang_label":     "🌐 Language",
    },
    "hi": {
        "title":          "👁️ स्ट्रैबिस्मस डिटेक्शन सिस्टम",
        "subtitle":       "आंखों की छवि अपलोड करें या वेबकैम का उपयोग करें — यह पता लगाने के लिए कि आंखें **सामान्य** हैं या **स्ट्रैबिस्मस** के लक्षण हैं।",
        "what_title":     "### 👁️ स्ट्रैबिस्मस क्या है?",
        "what_body":      "**स्ट्रैबिस्मस** (भेंगापन) एक ऐसी स्थिति है जिसमें दोनों आँखें एक ही समय में एक ही बिंदु पर नहीं देखतीं। एक आँख अंदर, बाहर, ऊपर या नीचे मुड़ सकती है।",
        "types_title":    "#### प्रकार",
        "types_body":     "- **एसोट्रोपिया** — आँख *अंदर* मुड़ती है (बच्चों में सबसे आम)\n- **एक्सोट्रोपिया** — आँख *बाहर* मुड़ती है\n- **हाइपरट्रोपिया** — आँख *ऊपर* मुड़ती है\n- **हाइपोट्रोपिया** — आँख *नीचे* मुड़ती है\n\n> जल्दी पहचान ज़रूरी है — उपचार न होने पर आलसी आँख या स्थायी दृष्टि हानि हो सकती है।",
        "tab_upload":     "📁 छवि अपलोड करें",
        "tab_webcam":     "📷 वेबकैम",
        "upload_label":   "आँख की छवि अपलोड करें",
        "analyze_upload": "🔍 विश्लेषण करें",
        "result_label":   "परिणाम",
        "probs_label":    "वर्ग संभावनाएँ",
        "plot_label":     "संभावना ग्राफ",
        "cam_label":      "लाइव कैमरा",
        "cam_tip":        "ℹ️ कैमरा अनुमति दें। कैमरा बॉक्स में 📷 बटन दबाएं, फिर **विश्लेषण करें** दबाएं।",
        "analyze_webcam": "🔍 स्नैपशॉट विश्लेषण",
        "no_snap":        "⚠️ कोई स्नैपशॉट नहीं। पहले 📷 बटन दबाएं।",
        "disclaimer":     "⚠️ **अस्वीकरण:** यह एक AI-आधारित स्क्रीनिंग टूल है और **चिकित्सा निदान के लिए नहीं** है। सटीक निदान के लिए किसी योग्य नेत्र विशेषज्ञ से परामर्श लें।",
        "chart_title":    "वर्ग संभावना वितरण",
        "chart_ylabel":   "संभावना (%)",
        "lang_label":     "🌐 भाषा",
    },
    "mr": {
        "title":          "👁️ स्ट्रॅबिस्मस डिटेक्शन सिस्टम",
        "subtitle":       "डोळ्याची प्रतिमा अपलोड करा किंवा वेबकॅम वापरा — डोळे **सामान्य** आहेत की **स्ट्रॅबिस्मस** आहे हे तपासण्यासाठी।",
        "what_title":     "### 👁️ स्ट्रॅबिस्मस म्हणजे काय?",
        "what_body":      "**स्ट्रॅबिस्मस** (तिरळे डोळे) ही अशी स्थिती आहे जिथे दोन्ही डोळे एकाच वेळी एकाच बिंदूकडे पाहत नाहीत। एक डोळा आत, बाहेर, वर किंवा खाली वळतो।",
        "types_title":    "#### प्रकार",
        "types_body":     "- **एसोट्रोपिया** — डोळा *आत* वळतो\n- **एक्सोट्रोपिया** — डोळा *बाहेर* वळतो\n- **हायपरट्रोपिया** — डोळा *वर* वळतो\n- **हायपोट्रोपिया** — डोळा *खाली* वळतो\n\n> लवकर ओळख महत्त्वाची — उपचार न केल्यास दृष्टी कमी होऊ शकते।",
        "tab_upload":     "📁 प्रतिमा अपलोड करा",
        "tab_webcam":     "📷 वेबकॅम",
        "upload_label":   "डोळ्याची प्रतिमा अपलोड करा",
        "analyze_upload": "🔍 विश्लेषण करा",
        "result_label":   "निकाल",
        "probs_label":    "वर्ग संभाव्यता",
        "plot_label":     "संभाव्यता आलेख",
        "cam_label":      "लाइव्ह कॅमेरा",
        "cam_tip":        "ℹ️ कॅमेरा परवानगी द्या. कॅमेरा बॉक्समध्ये 📷 बटण दाबा, नंतर **विश्लेषण करा** दाबा.",
        "analyze_webcam": "🔍 स्नॅपशॉट विश्लेषण",
        "no_snap":        "⚠️ स्नॅपशॉट नाही. आधी 📷 बटण दाबा.",
        "disclaimer":     "⚠️ **अस्वीकरण:** हे AI-आधारित स्क्रीनिंग साधन आहे आणि **वैद्यकीय निदानासाठी नाही**. योग्य निदानासाठी नेत्रतज्ञाचा सल्ला घ्या.",
        "chart_title":    "वर्ग संभाव्यता वितरण",
        "chart_ylabel":   "संभाव्यता (%)",
        "lang_label":     "🌐 भाषा",
    },
    "ta": {
        "title":          "👁️ ஸ்ட்ரபிஸ்மஸ் கண்டறிதல் அமைப்பு",
        "subtitle":       "கண் படத்தை பதிவேற்றவும் அல்லது வெப்கேமை பயன்படுத்தவும் — கண்கள் **சாதாரணமாக** உள்ளதா அல்லது **ஸ்ட்ரபிஸ்மஸ்** அறிகுறிகள் உள்ளதா என்று கண்டறிய.",
        "what_title":     "### 👁️ ஸ்ட்ரபிஸ்மஸ் என்றால் என்ன?",
        "what_body":      "**ஸ்ட்ரபிஸ்மஸ்** (கோணல் கண்) என்பது இரண்டு கண்களும் ஒரே நேரத்தில் ஒரே புள்ளியை பார்க்காத நிலை. ஒரு கண் உள்ளே, வெளியே, மேலே அல்லது கீழே திரும்பலாம்.",
        "types_title":    "#### வகைகள்",
        "types_body":     "- **எசோட்ரோபியா** — கண் *உள்ளே* திரும்பும்\n- **எக்சோட்ரோபியா** — கண் *வெளியே* திரும்பும்\n- **ஹைபர்ட்ரோபியா** — கண் *மேலே* திரும்பும்\n- **ஹைபோட்ரோபியா** — கண் *கீழே* திரும்பும்\n\n> முன்கூட்டிய கண்டறிதல் முக்கியம் — சிகிச்சையில்லாமல் பார்வை இழப்பு ஏற்படலாம்.",
        "tab_upload":     "📁 படம் பதிவேற்று",
        "tab_webcam":     "📷 வெப்கேம்",
        "upload_label":   "கண் படத்தை பதிவேற்றவும்",
        "analyze_upload": "🔍 பகுப்பாய்வு செய்",
        "result_label":   "முடிவு",
        "probs_label":    "வகுப்பு நிகழ்தகவுகள்",
        "plot_label":     "நிகழ்தகவு வரைபடம்",
        "cam_label":      "நேரடி கேமரா",
        "cam_tip":        "ℹ️ கேமரா அணுகலை அனுமதிக்கவும். 📷 பொத்தானை அழுத்தி, **பகுப்பாய்வு** அழுத்தவும்.",
        "analyze_webcam": "🔍 ஸ்னாப்ஷாட் பகுப்பாய்வு",
        "no_snap":        "⚠️ ஸ்னாப்ஷாட் இல்லை. முதலில் 📷 பொத்தானை அழுத்தவும்.",
        "disclaimer":     "⚠️ **மறுப்பு:** இது AI அடிப்படையிலான திரையிடல் கருவி மற்றும் **மருத்துவ நோயறிதலுக்காக அல்ல**. சரியான நோயறிதலுக்கு கண் மருத்துவரை அணுகவும்.",
        "chart_title":    "வகுப்பு நிகழ்தகவு விநியோகம்",
        "chart_ylabel":   "நிகழ்தகவு (%)",
        "lang_label":     "🌐 மொழி",
    },
    "te": {
        "title":          "👁️ స్ట్రాబిస్మస్ డిటెక్షన్ సిస్టమ్",
        "subtitle":       "కంటి చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా వెబ్‌క్యామ్ ఉపయోగించండి — కళ్ళు **సాధారణంగా** ఉన్నాయా లేదా **స్ట్రాబిస్మస్** లక్షణాలు ఉన్నాయా అని తెలుసుకోండి.",
        "what_title":     "### 👁️ స్ట్రాబిస్మస్ అంటే ఏమిటి?",
        "what_body":      "**స్ట్రాబిస్మస్** (వాలుకన్ను) అనేది రెండు కళ్ళూ ఒకే సమయంలో ఒకే బిందువును చూడని స్థితి. ఒక కన్ను లోపలికి, బయటకి, పైకి లేదా కిందికి తిరగవచ్చు.",
        "types_title":    "#### రకాలు",
        "types_body":     "- **ఎసోట్రోపియా** — కన్ను *లోపలికి* తిరుగుతుంది\n- **ఎక్సోట్రోపియా** — కన్ను *బయటికి* తిరుగుతుంది\n- **హైపర్‌ట్రోపియా** — కన్ను *పైకి* తిరుగుతుంది\n- **హైపోట్రోపియా** — కన్ను *కిందికి* తిరుగుతుంది\n\n> ముందస్తు గుర్తింపు ముఖ్యం — చికిత్స లేకుండా దృష్టి నష్టం జరగవచ్చు.",
        "tab_upload":     "📁 చిత్రం అప్‌లోడ్",
        "tab_webcam":     "📷 వెబ్‌క్యామ్",
        "upload_label":   "కంటి చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "analyze_upload": "🔍 విశ్లేషించు",
        "result_label":   "ఫలితం",
        "probs_label":    "తరగతి సంభావ్యతలు",
        "plot_label":     "సంభావ్యత గ్రాఫ్",
        "cam_label":      "లైవ్ కెమెరా",
        "cam_tip":        "ℹ️ కెమెరా అనుమతి ఇవ్వండి. 📷 బటన్ నొక్కి, **విశ్లేషించు** నొక్కండి.",
        "analyze_webcam": "🔍 స్నాప్‌షాట్ విశ్లేషణ",
        "no_snap":        "⚠️ స్నాప్‌షాట్ లేదు. ముందుగా 📷 బటన్ నొక్కండి.",
        "disclaimer":     "⚠️ **నిరాకరణ:** ఇది AI ఆధారిత స్క్రీనింగ్ సాధనం మరియు **వైద్య నిర్ధారణ కోసం కాదు**. సరైన నిర్ధారణ కోసం నేత్ర వైద్యుడిని సంప్రదించండి.",
        "chart_title":    "తరగతి సంభావ్యత పంపిణీ",
        "chart_ylabel":   "సంభావ్యత (%)",
        "lang_label":     "🌐 భాష",
    },
    "bn": {
        "title":          "👁️ স্ট্র্যাবিসমাস ডিটেকশন সিস্টেম",
        "subtitle":       "একটি চোখের ছবি আপলোড করুন বা ওয়েবক্যাম ব্যবহার করুন — চোখ **স্বাভাবিক** না **স্ট্র্যাবিসমাস** আছে কিনা জানতে।",
        "what_title":     "### 👁️ স্ট্র্যাবিসমাস কী?",
        "what_body":      "**স্ট্র্যাবিসমাস** (বাঁকা চোখ) এমন একটি অবস্থা যেখানে উভয় চোখ একই সময়ে একই বিন্দুতে দেখে না। একটি চোখ ভেতরে, বাইরে, উপরে বা নিচে ঘুরতে পারে।",
        "types_title":    "#### ধরন",
        "types_body":     "- **এসোট্রোপিয়া** — চোখ *ভেতরে* ঘোরে\n- **এক্সোট্রোপিয়া** — চোখ *বাইরে* ঘোরে\n- **হাইপারট্রোপিয়া** — চোখ *উপরে* ঘোরে\n- **হাইপোট্রোপিয়া** — চোখ *নিচে* ঘোরে\n\n> প্রাথমিক সনাক্তকরণ গুরুত্বপূর্ণ — চিকিৎসা না হলে দৃষ্টিশক্তি হারাতে পারে।",
        "tab_upload":     "📁 ছবি আপলোড",
        "tab_webcam":     "📷 ওয়েবক্যাম",
        "upload_label":   "চোখের ছবি আপলোড করুন",
        "analyze_upload": "🔍 বিশ্লেষণ করুন",
        "result_label":   "ফলাফল",
        "probs_label":    "শ্রেণী সম্ভাবনা",
        "plot_label":     "সম্ভাবনা গ্রাফ",
        "cam_label":      "লাইভ ক্যামেরা",
        "cam_tip":        "ℹ️ ক্যামেরার অনুমতি দিন। 📷 বোতাম চাপুন, তারপর **বিশ্লেষণ করুন** চাপুন।",
        "analyze_webcam": "🔍 স্ন্যাপশট বিশ্লেষণ",
        "no_snap":        "⚠️ কোনো স্ন্যাপশট নেই। প্রথমে 📷 বোতাম চাপুন।",
        "disclaimer":     "⚠️ **দাবিত্যাগ:** এটি একটি AI-ভিত্তিক স্ক্রীনিং টুল এবং **চিকিৎসা নির্ণয়ের জন্য নয়**। সঠিক নির্ণয়ের জন্য একজন চক্ষু বিশেষজ্ঞের পরামর্শ নিন।",
        "chart_title":    "শ্রেণী সম্ভাবনা বিতরণ",
        "chart_ylabel":   "সম্ভাবনা (%)",
        "lang_label":     "🌐 ভাষা",
    },
    "es": {
        "title":          "👁️ Sistema de Detección de Estrabismo",
        "subtitle":       "Sube una imagen del ojo o usa la cámara web para detectar si los ojos son **normales** o muestran signos de **estrabismo**.",
        "what_title":     "### 👁️ ¿Qué es el Estrabismo?",
        "what_body":      "El **estrabismo** (ojos bizcos) es una condición en la que ambos ojos no miran al mismo punto al mismo tiempo. Un ojo puede girar hacia adentro, afuera, arriba o abajo.",
        "types_title":    "#### Tipos",
        "types_body":     "- **Esotropia** — el ojo gira *hacia adentro*\n- **Exotropia** — el ojo gira *hacia afuera*\n- **Hipertropia** — el ojo gira *hacia arriba*\n- **Hipotropia** — el ojo gira *hacia abajo*\n\n> La detección temprana es crucial — sin tratamiento puede causar ambliopía o pérdida permanente de visión.",
        "tab_upload":     "📁 Subir Imagen",
        "tab_webcam":     "📷 Cámara Web",
        "upload_label":   "Subir imagen del ojo",
        "analyze_upload": "🔍 Analizar Imagen",
        "result_label":   "Resultado",
        "probs_label":    "Probabilidades de Clase",
        "plot_label":     "Gráfico de Probabilidades",
        "cam_label":      "Cámara en Vivo",
        "cam_tip":        "ℹ️ Permite el acceso a la cámara. Haz clic en el botón 📷 y luego en **Analizar**.",
        "analyze_webcam": "🔍 Analizar Foto",
        "no_snap":        "⚠️ No hay foto. Haz clic en el botón 📷 primero.",
        "disclaimer":     "⚠️ **Aviso:** Esta es una herramienta de detección basada en IA y **NO está destinada a diagnóstico médico**. Consulta siempre a un oftalmólogo.",
        "chart_title":    "Distribución de Probabilidad",
        "chart_ylabel":   "Probabilidad (%)",
        "lang_label":     "🌐 Idioma",
    },
    "fr": {
        "title":          "👁️ Système de Détection du Strabisme",
        "subtitle":       "Téléchargez une image de l'œil ou utilisez la webcam pour détecter si les yeux sont **normaux** ou présentent des signes de **strabisme**.",
        "what_title":     "### 👁️ Qu'est-ce que le Strabisme ?",
        "what_body":      "Le **strabisme** (yeux croisés) est une condition où les deux yeux ne regardent pas le même point en même temps. Un œil peut se tourner vers l'intérieur, l'extérieur, le haut ou le bas.",
        "types_title":    "#### Types",
        "types_body":     "- **Ésotropie** — l'œil se tourne *vers l'intérieur*\n- **Exotropie** — l'œil se tourne *vers l'extérieur*\n- **Hypertropie** — l'œil se tourne *vers le haut*\n- **Hypotropie** — l'œil se tourne *vers le bas*\n\n> La détection précoce est essentielle — sans traitement, cela peut causer une amblyopie ou une perte permanente de la vision.",
        "tab_upload":     "📁 Télécharger une Image",
        "tab_webcam":     "📷 Webcam",
        "upload_label":   "Télécharger l'image de l'œil",
        "analyze_upload": "🔍 Analyser l'Image",
        "result_label":   "Résultat",
        "probs_label":    "Probabilités de Classe",
        "plot_label":     "Graphique de Probabilité",
        "cam_label":      "Caméra en Direct",
        "cam_tip":        "ℹ️ Autorisez l'accès à la caméra. Cliquez sur 📷 puis sur **Analyser**.",
        "analyze_webcam": "🔍 Analyser la Photo",
        "no_snap":        "⚠️ Pas de photo. Cliquez d'abord sur 📷.",
        "disclaimer":     "⚠️ **Avertissement:** Ceci est un outil de dépistage basé sur l'IA et **n'est pas destiné au diagnostic médical**. Consultez toujours un ophtalmologue.",
        "chart_title":    "Distribution des Probabilités",
        "chart_ylabel":   "Probabilité (%)",
        "lang_label":     "🌐 Langue",
    },
    "ar": {
        "title":          "👁️ نظام الكشف عن الحول",
        "subtitle":       "قم بتحميل صورة للعين أو استخدم كاميرا الويب للكشف عما إذا كانت العيون **طبيعية** أم تُظهر علامات **الحول**.",
        "what_title":     "### 👁️ ما هو الحول؟",
        "what_body":      "**الحول** هو حالة لا تنظر فيها العينان إلى نفس النقطة في نفس الوقت. قد تنحرف إحدى العينين إلى الداخل أو الخارج أو الأعلى أو الأسفل.",
        "types_title":    "#### الأنواع",
        "types_body":     "- **الحول الداخلي (Esotropia)** — العين تنحرف *للداخل*\n- **الحول الخارجي (Exotropia)** — العين تنحرف *للخارج*\n- **الحول العلوي (Hypertropia)** — العين تنحرف *للأعلى*\n- **الحول السفلي (Hypotropia)** — العين تنحرف *للأسفل*\n\n> الكشف المبكر مهم — بدون علاج قد يؤدي إلى كسل العين أو فقدان البصر الدائم.",
        "tab_upload":     "📁 تحميل صورة",
        "tab_webcam":     "📷 كاميرا الويب",
        "upload_label":   "تحميل صورة العين",
        "analyze_upload": "🔍 تحليل الصورة",
        "result_label":   "النتيجة",
        "probs_label":    "احتمالات الفئة",
        "plot_label":     "رسم الاحتمالات",
        "cam_label":      "الكاميرا المباشرة",
        "cam_tip":        "ℹ️ اسمح بالوصول إلى الكاميرا. انقر على 📷 ثم على **تحليل**.",
        "analyze_webcam": "🔍 تحليل اللقطة",
        "no_snap":        "⚠️ لا توجد لقطة. انقر أولاً على 📷.",
        "disclaimer":     "⚠️ **إخلاء المسؤولية:** هذه أداة فحص قائمة على الذكاء الاصطناعي و**ليست مخصصة للتشخيص الطبي**. استشر دائماً طبيب عيون متخصصاً.",
        "chart_title":    "توزيع احتمالات الفئة",
        "chart_ylabel":   "الاحتمال (%)",
        "lang_label":     "🌐 اللغة",
    },
    "zh": {
        "title":          "👁️ 斜视检测系统",
        "subtitle":       "上传眼睛图片或使用摄像头，检测眼睛是否**正常**或显示**斜视**迹象。",
        "what_title":     "### 👁️ 什么是斜视？",
        "what_body":      "**斜视**（对眼）是一种两眼无法同时注视同一点的眼部疾病。一只眼睛可能向内、向外、向上或向下偏转。",
        "types_title":    "#### 类型",
        "types_body":     "- **内斜视 (Esotropia)** — 眼睛*向内*偏转\n- **外斜视 (Exotropia)** — 眼睛*向外*偏转\n- **上斜视 (Hypertropia)** — 眼睛*向上*偏转\n- **下斜视 (Hypotropia)** — 眼睛*向下*偏转\n\n> 早期发现至关重要 — 若不治疗可能导致弱视或永久性视力丧失。",
        "tab_upload":     "📁 上传图片",
        "tab_webcam":     "📷 摄像头",
        "upload_label":   "上传眼睛图片",
        "analyze_upload": "🔍 分析图片",
        "result_label":   "结果",
        "probs_label":    "类别概率",
        "plot_label":     "概率图表",
        "cam_label":      "实时摄像头",
        "cam_tip":        "ℹ️ 允许摄像头访问。点击 📷 快照按钮，然后点击**分析**。",
        "analyze_webcam": "🔍 分析快照",
        "no_snap":        "⚠️ 没有快照。请先点击 📷 按钮。",
        "disclaimer":     "⚠️ **免责声明：** 这是一个基于AI的筛查工具，**不用于医疗诊断**。请始终咨询合格的眼科医生。",
        "chart_title":    "类别概率分布",
        "chart_ylabel":   "概率 (%)",
        "lang_label":     "🌐 语言",
    },
    "ml": {
        "title":          "👁️ സ്ട്രാബിസ്മസ് നിർണ്ണയ സംവിധാനം",
        "subtitle":       "കണ്ണുകൾ **സാധാരണ നിലയിലാണോ** അതോ **സ്ട്രാബിസ്മസിന്റെ (മാരകണ്ണ്/കോങ്കണ്ണ്)** ലക്ഷണങ്ങൾ കാണിക്കുന്നുണ്ടോ എന്ന് കണ്ടെത്തുന്നതിന് ഒരു കണ്ണിന്റെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക അല്ലെങ്കിൽ നിങ്ങളുടെ വെബ്‌ക്യാം ഉപയോഗിക്കുക.",
        "what_title":     "### 👁️ എന്താണ് സ്ട്രാബിസ്മസ് (കോങ്കണ്ണ്)?",
        "what_body":      "രണ്ട് കണ്ണുകളും ഒരേ സമയം ഒരേ ബിന്ദുവിലേക്ക് നോക്കാത്ത അവസ്ഥയാണ് **സ്ട്രാബിസ്മസ്** (കോങ്കണ്ണ്). ഒരു കണ്ണ് നേരെ നോക്കുമ്പോൾ മറ്റേ കണ്ണ് ഉള്ളിലേക്കോ, പുറത്തേക്കോ, മുകളിലേക്കോ, താഴേക്കോ തിരിഞ്ഞിരിക്കാം.",
        "types_title":    "#### തരങ്ങൾ",
        "types_body":     "- **എസോട്രോപ്പിയ (Esotropia)** — കണ്ണ് *ഉള്ളിലേക്ക്* തിരിയുന്നു (കുട്ടികളിൽ സാധാരണയായി കാണപ്പെടുന്നു)\n- **എക്സോട്രോപ്പിയ (Exotropia)** — കണ്ണ് *പുറത്തേക്ക്* തിരിയുന്നു\n- **ഹൈപ്പർട്രോപ്പിയ (Hypertropia)** — കണ്ണ് *മുകളിലേക്ക്* തിരിയുന്നു\n- **ഹൈപ്പോട്രോപ്പിയ (Hypotropia)** — കണ്ണ് *താഴേക്ക്* തിരിയുന്നു\n\n> നേരത്തെയുള്ള കണ്ടെത്തൽ പ്രധാനമാണ് — കൃത്യസമയത്ത് ചികിത്സിച്ചില്ലെങ്കിൽ സ്ട്രാബിസ്മസ് അംബ്ലിയോപ്പിയയിലേക്കോ (amblyopia/സോംബേരി കണ്ണ്) ശാശ്വതമായ കാഴ്ചനഷ്ടത്തിലേക്കോ നയിച്ചേക്കാം.",
        "tab_upload":     "📁 ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "tab_webcam":     "📷 വെബ്‌ക്യാം വഴി ചിത്രമെടുക്കുക",
        "upload_label":   "കണ്ണിന്റെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "analyze_upload": "🔍 അപ്‌ലോഡ് ചെയ്ത ചിത്രം പരിശോധിക്കുക",
        "result_label":   "ഫലം",
        "probs_label":    "സാധ്യതാനിരക്കുകൾ (Class Probabilities)",
        "plot_label":     "സാധ്യതാ ഗ്രാഫ്",
        "cam_label":      "ലൈവ് ക്യാമറ ഫീഡ്",
        "cam_tip":        "ℹ️ ചോദിക്കുമ്പോൾ ക്യാമറ ആക്സസ് അനുവദിക്കുക. ക്യാമറ ബോക്സിനുള്ളിലെ **📷 സ്നാപ്പ്ഷോട്ട് (snapshot)** ബട്ടൺ ക്ലിക്ക് ചെയ്യുക, തുടർന്ന് **Analyze Snapshot** അമർത്തുക.",
        "analyze_webcam": "🔍 സ്നാപ്പ്ഷോട്ട് പരിശോധിക്കുക",
        "no_snap":        "⚠️ സ്നാപ്പ്ഷോട്ട് എടുത്തിട്ടില്ല. ആദ്യം 📷 ക്യാപ്ചർ ബട്ടൺ ക്ലിക്ക് ചെയ്യുക.",
        "disclaimer":     "⚠️ **നിരാകരണം (Disclaimer):** ഇതൊരു AI അധിഷ്ഠിത സ്ക്രീനിംഗ് ടൂൾ മാത്രമാണ്, **മെഡിക്കൽ രോഗനിർണ്ണയത്തിനോ ക്ലിനിക്കൽ ആവശ്യങ്ങൾക്കോ ഉള്ളതല്ല**. കൃത്യമായ രോഗനിർണ്ണയത്തിനും ചികിത്സയ്ക്കും എപ്പോഴും യോഗ്യതയുള്ള ഒരു ഒഫ്താൽമോളജിസ്റ്റിനെയോ (കണ്ണ് രോഗവിദഗ്ദ്ധൻ) മെഡിക്കൽ പ്രൊഫഷണലിനെയോ സമീപിക്കുക.",
        "chart_title":    "ക്ലാസ് പ്രോബബിലിറ്റി ഡിസ്ട്രിബ്യൂഷൻ",
        "chart_ylabel":   "സാധ്യത (%)",
        "lang_label":     "🌐 ഭാഷ"
    },

}

def t(lang_name, key):
    code = LANGUAGES.get(lang_name, "en")
    return T.get(code, T["en"]).get(key, T["en"][key])

# =============================================================================
# Model loading
# =============================================================================
model  = None
classes = []

def load_model_safe():
    global model, classes
    try:
        model = tf.keras.models.load_model("models/strabismus_model.keras")
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
        with open("models/class_indices.json", "r") as f:
            class_indices = json.load(f)
        classes = [None] * len(class_indices)
        for k, v in class_indices.items():
            classes[v] = k
        print("Model loaded and warmed up.")
    except Exception as e:
        print(f"Model loading failed: {e}")
        model = None
        classes = []

load_model_safe()

# =============================================================================
# Eye Detection
# =============================================================================
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def contains_human_eye(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray   = cv2.equalizeHist(gray)
    faces  = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            roi  = gray[fy:fy+fh, fx:fx+fw]
            eyes = EYE_CASCADE.detectMultiScale(roi, 1.1, 4, minSize=(20, 20))
            if len(eyes) > 0:
                return True, ""
        return False, "A face was detected but no open eyes could be found.\nPlease ensure the eyes are open, unobstructed, and well-lit."
    eyes = EYE_CASCADE.detectMultiScale(gray, 1.05, 6, minSize=(30, 30))
    if len(eyes) > 0:
        return True, ""
    return False, "No human eyes detected in this image.\nPlease upload a clear photo showing human eyes.\nTips: ensure good lighting, eyes are open and visible."

# =============================================================================
# Preprocess
# =============================================================================
def preprocess_image(img):
    if max(img.size) > 512:
        img.thumbnail((512, 512), Image.BILINEAR)
    img       = img.resize((224, 224), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# =============================================================================
# Core Prediction
# =============================================================================
def run_prediction(img, lang):
    if model is None:
        return "❌ Model not loaded. Please restart the app.", {}, None
    if img is None:
        return "⚠️ No image provided.", {}, None
    try:
        img = img.convert("RGB")
        if max(img.size) > 640:
            img.thumbnail((640, 640), Image.BILINEAR)
        has_eye, reason = contains_human_eye(img)
        if not has_eye:
            return "Invalid Image\n\n" + reason, {}, None

        input_data = preprocess_image(img)
        with _predict_lock:
            prediction = model.predict(input_data, verbose=0)[0]

        prob_dict       = {classes[i]: float(prediction[i]) for i in range(len(classes))}
        predicted_index = int(np.argmax(prediction))
        predicted_class = classes[predicted_index]
        confidence      = prediction[predicted_index] * 100
        final_result    = "✅ NORMAL" if predicted_class == "NORMAL" else "⚠️ STRABISMUS DETECTED"

        chart_title  = t(lang, "chart_title")
        chart_ylabel = t(lang, "chart_ylabel")

        fig, ax = plt.subplots(figsize=(5, 3))
        labels  = list(prob_dict.keys())
        values  = [v * 100 for v in prob_dict.values()]
        colors  = ["#2ecc71" if l == "NORMAL" else "#e74c3c" for l in labels]
        bars    = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(chart_ylabel, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(chart_title, fontsize=10)
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

# =============================================================================
# Tab wrappers
# =============================================================================
def predict_upload(img, lang):
    return run_prediction(img, lang)

def predict_webcam(img, lang):
    if img is None:
        return t(lang, "no_snap"), {}, None
    pil_img = Image.fromarray(img.astype("uint8"))
    return run_prediction(pil_img, lang)

# =============================================================================
# UI update on language change
# =============================================================================
def change_language(lang):
    return (
        gr.update(value=f"# {t(lang,'title')}\n{t(lang,'subtitle')}"),
        gr.update(value=t(lang, "what_title") + "\n\n" + t(lang, "what_body") + "\n\n" + t(lang, "types_title") + "\n" + t(lang, "types_body")),
        gr.update(label=t(lang, "upload_label")),
        gr.update(value=t(lang, "analyze_upload")),
        gr.update(label=t(lang, "result_label")),
        gr.update(label=t(lang, "probs_label")),
        gr.update(label=t(lang, "plot_label")),
        gr.update(label=t(lang, "cam_label")),
        gr.update(value=t(lang, "cam_tip")),
        gr.update(value=t(lang, "analyze_webcam")),
        gr.update(label=t(lang, "result_label")),
        gr.update(label=t(lang, "probs_label")),
        gr.update(label=t(lang, "plot_label")),
        gr.update(value=f"---\n> {t(lang, 'disclaimer')}"),
    )

# =============================================================================
# Gradio UI
# =============================================================================
with gr.Blocks(title="👁️ Strabismus Detection System") as demo:

    # Language selector
    with gr.Row():
        with gr.Column(scale=4):
            header_md = gr.Markdown(
                f"# {t('English','title')}\n{t('English','subtitle')}"
            )
        with gr.Column(scale=1, min_width=160):
            lang_dd = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="English",
                label=t("English", "lang_label"),
                interactive=True,
            )

    # About / Info section
    info_md = gr.Markdown(
        t("English", "what_title") + "\n\n" +
        t("English", "what_body")  + "\n\n" +
        t("English", "types_title") + "\n" +
        t("English", "types_body")
    )

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1: Upload ────────────────────────────────────────────────────
        with gr.TabItem(t("English", "tab_upload")):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        type="pil",
                        label=t("English", "upload_label"),
                        sources=["upload"],
                    )
                    upload_btn = gr.Button(t("English", "analyze_upload"), variant="primary")

                with gr.Column(scale=1):
                    upload_result = gr.Textbox(label=t("English", "result_label"), lines=5)
                    upload_probs  = gr.Label(label=t("English", "probs_label"))
                    upload_plot   = gr.Plot(label=t("English", "plot_label"))

            upload_btn.click(
                fn=predict_upload,
                inputs=[upload_input, lang_dd],
                outputs=[upload_result, upload_probs, upload_plot],
            )
            upload_input.change(
                fn=predict_upload,
                inputs=[upload_input, lang_dd],
                outputs=[upload_result, upload_probs, upload_plot],
            )

        # ── Tab 2: Webcam ────────────────────────────────────────────────────
        with gr.TabItem(t("English", "tab_webcam")):
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        type="numpy",
                        label=t("English", "cam_label"),
                        sources=["webcam"],
                        streaming=False,
                    )
                    cam_tip_md = gr.Markdown(t("English", "cam_tip"))
                    webcam_btn = gr.Button(t("English", "analyze_webcam"), variant="primary")

                with gr.Column(scale=1):
                    webcam_result = gr.Textbox(label=t("English", "result_label"), lines=5)
                    webcam_probs  = gr.Label(label=t("English", "probs_label"))
                    webcam_plot   = gr.Plot(label=t("English", "plot_label"))

            webcam_btn.click(
                fn=predict_webcam,
                inputs=[webcam_input, lang_dd],
                outputs=[webcam_result, webcam_probs, webcam_plot],
            )
            webcam_input.change(
                fn=predict_webcam,
                inputs=[webcam_input, lang_dd],
                outputs=[webcam_result, webcam_probs, webcam_plot],
            )

    disclaimer_md = gr.Markdown(f"---\n> {t('English', 'disclaimer')}")

    # Wire language switcher
    lang_dd.change(
        fn=change_language,
        inputs=[lang_dd],
        outputs=[
            header_md, info_md,
            upload_input, upload_btn, upload_result, upload_probs, upload_plot,
            webcam_input, cam_tip_md, webcam_btn, webcam_result, webcam_probs, webcam_plot,
            disclaimer_md,
        ],
    )

# =============================================================================
# Launch
# =============================================================================
demo.queue(
    max_size=100,
    default_concurrency_limit=2,
)
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
    show_error=True,
    share=True,
)
