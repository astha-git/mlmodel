# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

st.title("🖐️ Real-Time Sign Language Detection (English + Hindi)")

# Load trained model
model = pickle.load(open("real_model.pkl","rb"))
le = pickle.load(open("le.pkl","rb"))

HINDI_MAP = {
    "hello": "नमस्ते",
    "thank_you": "धन्यवाद",
    "yes": "हाँ",
    "no": "नहीं",
    "love": "प्यार",
    "stop": "रुको"
}

language = st.radio("Choose Language", ["English", "Hindi"])
run = st.checkbox("Start Camera")
frame_window = st.image([])

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Streamlit container for displaying detected gesture
gesture_placeholder = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not found")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    display_label = "No hand detected"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            # Flatten landmarks
            landmarks = [lm.x for lm in handLms.landmark] + [lm.y for lm in handLms.landmark] + [lm.z for lm in handLms.landmark]
            X = np.array(landmarks).reshape(1,-1)
            pred = model.predict(X)[0]
            label = le.inverse_transform([pred])[0]
            display_label = HINDI_MAP.get(label, label) if language=="Hindi" else label

    # Update gesture text in Streamlit
    gesture_placeholder.markdown(f"**Detected Gesture:** {display_label}")

    # Show camera frame
    frame_window.image(frame, channels="BGR")

cap.release()
