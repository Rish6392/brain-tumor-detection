import streamlit as st
import os

import tensorflow as tf

import numpy as np
import cv2
from PIL import Image
import sys

# for models 
# detection =   https://drive.google.com/file/d/1EzN6co3Ig7QfgF6WTktkWfmj4RTWQEg0/view?usp=sharing
# type =        https://drive.google.com/file/d/1v_8STwRZx-09zuzFXUWm7c7pcj3xt6og/view?usp=sharing



import requests

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
def download_file(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    if response.status_code != 200:
        raise Exception("Download failed. Check file permissions.")

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def download_models():
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    det_path = os.path.join(model_dir, "tumor_detection_model.keras")
    cls_path = os.path.join(model_dir, "tumor_type_model.keras")

    if not os.path.exists(det_path):
        download_file("1EzN6co3Ig7QfgF6WTktkWfmj4RTWQEg0", det_path)

    if not os.path.exists(cls_path):
        download_file("1v_8STwRZx-09zuzFXUWm7c7pcj3xt6og", cls_path)




# Page Config

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    color: #4CAF50;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.success-box {
    background-color: #e6ffe6;
    color: #2e7d32;
}

.error-box {
    background-color: #ffe6e6;
    color: #c62828;
}

.confidence-text {
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# Adding project root to path

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.preprocessing import preprocess_detection_image, preprocess_mat_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Load Models

@st.cache_resource
def load_models():
    download_models()
    detection_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "models", "tumor_detection_model.keras"),
        compile=False
    )

    type_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "models", "tumor_type_model.keras"),
        compile=False
    )

    return detection_model, type_model


model_det, model_type = load_models()


# Header

st.markdown("<div class='main-title'>🧠 Brain Tumor Detection</div>", unsafe_allow_html=True)
st.markdown("---")
st.info("Upload an MRI image and click Predict")


# Upload Section

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with col2:
        st.markdown("### Prediction Panel")

        if st.button("🔍 Predict"):

            
            # Detection
            
            img_det = preprocess_detection_image(img)
            img_det = img_det.reshape(1, 224, 224, 3)

            pred_det = model_det.predict(img_det)
            prediction = float(pred_det[0][0])

            confidence = prediction if prediction > 0.5 else (1 - prediction)

            
            # Confidence Meter
           
            st.progress(int(confidence * 100))
            st.markdown(
                f"<div class='confidence-text'>Confidence: {confidence*100:.2f}%</div>",
                unsafe_allow_html=True
            )

           
            # Result Display
          
            if prediction > 0.5:

                st.markdown(
                    "<div class='result-box error-box'>⚠️ Tumor Detected</div>",
                    unsafe_allow_html=True
                )

               
                # Classification
              
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_type = preprocess_mat_image(img_gray)
                img_type = img_type.reshape(1, 224, 224, 1)

                pred_type = model_type.predict(img_type)
                class_idx = np.argmax(pred_type)

                classes = ["Glioma", "Meningioma", "Pituitary"]

                st.markdown(
                    f"<div class='result-box success-box'>🧬 {classes[class_idx]}</div>",
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    "<div class='result-box success-box'>✅ No Tumor Detected</div>",
                    unsafe_allow_html=True
                )




# streamlit run streamlit_app.py