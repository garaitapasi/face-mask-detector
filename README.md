# 😷 Face Mask Detection System

## 📌 Description

This project is a **real-time Face Mask Detection System** built using Deep Learning and Computer Vision. It detects whether a person is wearing a mask or not using a trained **MobileNetV2 model**.

The system supports both:

* 🎥 Real-time webcam detection (main feature)
* 🌐 Image-based prediction using a simple web UI

---

## 🚀 Features

### 🔴 Webcam Version (Main Feature)

* Real-time face detection using OpenCV
* Mask / No Mask classification
* Live bounding boxes and labels
* 🔊 Audio alert when no mask is detected
* 👥 Counts number of people without mask

### 🌐 Web App Version

* Upload image and get prediction
* Detects faces and predicts mask status
* Simple UI built using Streamlit

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

Install dependencies:

```
pip install opencv-python numpy tensorflow
```

Run:

```
python webcam.py
```

Press **q** to exit

---

### 🥈 Web App Version

Install dependencies:

```
pip install -r requirements.txt
```

Run:

```
python -m streamlit run app.py
```

---

## 🎥 Demo

```
https://drive.google.com/file/d/1YDOP65el5W6_7v4vjCR7oGvNf_jzgEti/view?usp=sharing
```

---

## 🎯 Output

* 😷 Mask detected → Green bounding box
* ❌ No Mask detected → Red bounding box
* 🔊 Audio alert for violations
* 👥 Real-time count of people without mask

---

## 💡 Future Improvements

* Deploy full real-time system on web
* Improve model accuracy with more data
* Add face tracking and logging system

---


## 📢 Note

* The webcam version runs locally and requires camera access
* The Streamlit app is for image-based prediction
* Best results when faces are clearly visible

---
