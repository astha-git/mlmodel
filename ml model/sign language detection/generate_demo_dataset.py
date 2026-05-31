# collect_data.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Gestures to collect
GESTURES = ["hello","thank_you","yes","no","love","stop"]
SAMPLES_PER_GESTURE = 50
LANDMARKS = 21
DATA = []

os.makedirs("data", exist_ok=True)
cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    input(f"Prepare gesture '{gesture}' and press Enter to start collecting {SAMPLES_PER_GESTURE} samples...")
    count = 0
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                row = [gesture] + [coord for lm in handLms.landmark for coord in [lm.x, lm.y, lm.z]]
                DATA.append(row)
                count += 1

        cv2.putText(frame, f"{gesture}: {count}/{SAMPLES_PER_GESTURE}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save dataset
columns = ["label_en"] + [f"{axis}{i}" for i in range(LANDMARKS) for axis in ["x","y","z"]]
df = pd.DataFrame(DATA, columns=columns)
df.to_csv("data/real_signs.csv", index=False)
print("Data collection complete: data/real_signs.csv")
