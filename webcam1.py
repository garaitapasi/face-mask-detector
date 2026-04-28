import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound
import time

# Load trained model
model = load_model("mask_detector.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Alert control (avoid continuous beeping)
last_alert_time = 0

def play_alert():
    global last_alert_time
    if time.time() - last_alert_time > 2:  # 2 sec delay
        winsound.Beep(1000, 500)  # frequency, duration
        last_alert_time = time.time()

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    no_mask_count = 0  # counter

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Skip invalid face
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # Preprocess
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        # Predict
        pred = model.predict(face, verbose=0)[0][0]

        if pred < 0.5:
            label = "Mask 😷"
            color = (0, 255, 0)
        else:
            label = "No Mask ❌"
            color = (0, 0, 255)
            no_mask_count += 1

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put label
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display count
    cv2.putText(frame, f"No Mask Count: {no_mask_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Play alert if needed
    if no_mask_count > 0:
        play_alert()

    # Show window
    cv2.imshow("Face Mask Detector (Live)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()