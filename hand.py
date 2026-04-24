import cv2
import numpy as np
from math import hypot

import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (5,9),(9,10),(10,11),(11,12),   # Middle
    (9,13),(13,14),(14,15),(15,16), # Ring
    (13,17),(17,18),(18,19),(19,20),# Pinky
    (0,17)                          # Palm
]

def draw_hand_landmarks(img, hand_landmarks):
    h, w, _ = img.shape

    # Draw connections (lines)
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw points (circles)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)

# Load model
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_timestamp = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to Mediapipe Image
    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb)

    # Run detection
    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            draw_hand_landmarks(frame, hand_landmarks)

            # Thumb tip = 4, Index tip = 8
            x1, y1 = hand_landmarks[4].x, hand_landmarks[4].y
            x2, y2 = hand_landmarks[8].x, hand_landmarks[8].y

            h, w, _ = frame.shape
            cx1, cy1 = int(x1 * w), int(y1 * h)
            cx2, cy2 = int(x2 * w), int(y2 * h)

            cv2.circle(frame, (cx1, cy1), 10, (255, 0, 255), -1)
            cv2.circle(frame, (cx2, cy2), 10, (255, 0, 255), -1)
            cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 3)

            length = hypot(cx2 - cx1, cy2 - cy1)
            print(length)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()