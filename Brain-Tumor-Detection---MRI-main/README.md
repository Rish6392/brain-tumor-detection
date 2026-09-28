# 🧠 Brain Tumor Detection using Deep Learning

## 🚀 Overview

This project is an AI-powered system that detects brain tumors from MRI images and classifies tumor types using deep learning.

The system performs:
- ✅ Tumor Detection (Yes / No)
- ✅ Tumor Type Classification (Glioma, Meningioma, Pituitary)

Built using **TensorFlow, MobileNetV2, and Streamlit**.

---

## 🎯 Features

- 🧠 Binary Tumor Detection
- 🧬 Multi-class Tumor Classification
- 📊 Confidence Score Display
- 🖼️ Image Upload Interface
- ⚡ Fast Predictions

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Scikit-learn

---

## 🧠 Model Details

### 🔍 Detection Model
- Transfer Learning using **MobileNetV2**
- Input: `224 × 224 × 3`
- Output: Tumor / No Tumor

### 🧬 Classification Model
- Custom CNN
- Input: `224 × 224 × 1`
- Output: 3 Classes (Glioma, Meningioma, Pituitary)

---

## 📁 Project Structure

```
brain-tumor-detection-mri/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   └── model.py
│
├── notebooks/
│
├── models/              ← Place downloaded models here
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ABHINAY945/Brain-Tumor-Detection-MRI.git
cd Brain-Tumor-Detection-MRI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run streamlit_app.py
```

---

## ⚠️ Models (Important)

Trained models are **not included** in this repository due to GitHub file size limits.

👉 Download models from here:
https://drive.google.com/drive/folders/1AhzYeVl13vbyo7OjtD3pNNuSmkteWz5q?usp=sharing

After downloading, place them inside the `models/` directory:

```
models/
├── detection_model.h5
└── classification_model.h5
```

---

## 🖥️ Usage

1. Upload an MRI image
2. Click **Predict**
3. View results:
   - Tumor detected or not
   - Confidence score
   - Tumor type (if detected)

---

## 📈 Future Improvements

- Improve accuracy with a larger dataset
- Add Grad-CAM visualization
- Deploy on cloud (AWS / GCP / Hugging Face Spaces)
- Add performance metrics dashboard

---

## 🤝 Contributing

Feel free to fork this repository and submit a pull request with your improvements!

---

Author 

Himanshu Singh Yadav
