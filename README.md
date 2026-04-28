# 😷 Face Mask Detection System

## 📌 Description

This project is a **real-time Face Mask Detection System** built using Deep Learning and Computer Vision techniques.
It detects whether a person is wearing a mask or not using a trained **MobileNetV2 model**.

The system supports both:

* 🎥 **Real-time webcam detection**
* 🌐 **Image-based web application**

---

## 🚀 Features

### 🔴 Webcam Version (Main Feature)

* Real-time face detection using OpenCV
* Mask / No Mask classification
* Live bounding boxes and labels
* 🚨 Audio alert when no mask detected
* 👥 Counts number of people without mask

### 🌐 Web App Version

* Upload image and get prediction
* Simple UI using Streamlit

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit

---

## 📂 Project Structure

```
face-mask-detector/
│
├── webcam.py                # Real-time detection (MAIN)
├── app.py                   # Streamlit web app
├── face_mask_detection.ipynb
├── requirements.txt
├── runtime.txt
├── README.md
```

---

## ▶️ How to Run

### 🥇 Webcam Version (Recommended)

1. Install dependencies:

```
pip install opencv-python numpy tensorflow
```

2. Run:

```
python webcam.py
```

3. Press **q** to exit

---

### 🥈 Web App Version

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run:

```
python -m streamlit run app.py
```

---

## 🎥 Demo

👉 (Add your demo video link here)

Example:

```
https://drive.google.com/your-demo-link
```

---

## 🎯 Output

* 😷 Mask detected (Green box)
* ❌ No Mask detected (Red box)
* 🔊 Alert sound for violations
* 👥 Live count of people without mask

---

## 💡 Future Improvements

* Deploy real-time webcam version on web
* Improve accuracy with larger dataset
* Add face tracking and logging system

---

## 👨‍💻 Author

Tapasi

---

## 📢 Note

* Webcam version works on **local system only**
* Web version supports **image-based prediction**
* Best results when face is clearly visible

---
