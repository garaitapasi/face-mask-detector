import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# 🔽 Download model from Google Drive if not present
if not os.path.exists("mask_detector.keras"):
    import gdown
    url = "https://drive.google.com/uc?id=1WQjSlvYS93qRFBAnabkGz6XDNbHYK5zT"
    gdown.download(url, "mask_detector.keras", quiet=False)

# Load model
model = load_model("mask_detector.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

st.title("😷 Face Mask Detection")
st.write("Upload an image to detect mask on faces")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # Preprocess
        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized / 255.0
        face_resized = np.reshape(face_resized, (1, 224, 224, 3))

        # Predict
        pred = model.predict(face_resized, verbose=0)[0][0]

        if pred < 0.5:
            label = "Mask 😷"
            color = (0, 255, 0)
        else:
            label = "No Mask ❌"
            color = (0, 0, 255)

        # Draw box
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.image(img, caption="Result", use_container_width=True)
