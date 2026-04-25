import streamlit as st
import cv2
import numpy as np
import os

# 🔽 Download model from Google Drive if not present
if not os.path.exists("mask_detector.keras"):
    import gdown
    url = "https://drive.google.com/file/d/1WQjSlvYS93qRFBAnabkGz6XDNbHYK5zT/view?usp=sharing"   
    gdown.download(url, "mask_detector.keras", quiet=False)

from tensorflow.keras.models import load_model

# Load model
model = load_model("mask_detector.keras")

# Streamlit UI
st.set_page_config(page_title="Face Mask Detector", layout="centered")

st.title("😷 Face Mask Detection App")
st.write("Upload an image to check whether a person is wearing a mask or not.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, 224, 224, 3))

    # Prediction
    pred = model.predict(img_resized, verbose=0)[0][0]

    # Result
    if pred < 0.5:
        st.success("Mask 😷")
    else:
        st.error("No Mask ❌")