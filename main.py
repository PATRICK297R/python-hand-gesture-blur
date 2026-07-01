import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ---------- Constants ----------
MODEL_PATH = "hand_landmarker.task"
MAX_HANDS = 2
FPS_POS = (10, 30)
GESTURE_POS = (10, 60)
BLUR_POS = (10, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
BLUR_KERNEL = (51, 51)
PEACE_COLOR = (0, 255, 255)
NONE_COLOR = (0, 0, 255)
TOGGLE_COOLDOWN = 1.5  # seconds

# --- Jika video terlihat mirror, set ke True untuk membalik secara horizontal ---
FLIP_IMAGE = False  # True = tampilan mirror (seperti kaca), False = tampilan normal

# Landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17), (0, 17)    # palm
]


# ---------- Helper functions ----------
def draw_hand(image, hand_landmarks):
    """Draw landmarks and connections for a single hand."""
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for point in points:
        cv2.circle(image, point, 5, (0, 0, 255), -1)


def detect_peace(hand_landmarks):
    """
    Determine if the hand gesture is the Peace sign (V-sign).
    Conditions:
        - index finger extended
        - middle finger extended
        - ring finger closed
        - pinky closed
        - thumb closed
    Uses ratio of distances from wrist to tip vs wrist to PIP.
    """
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    wrist = hand_landmarks[0]

    idx_tip = hand_landmarks[8]
    idx_pip = hand_landmarks[6]
    idx_ratio = distance(wrist, idx_tip) / max(distance(wrist, idx_pip), 1e-6)

    mid_tip = hand_landmarks[12]
    mid_pip = hand_landmarks[10]
    mid_ratio = distance(wrist, mid_tip) / max(distance(wrist, mid_pip), 1e-6)

    ring_tip = hand_landmarks[16]
    ring_pip = hand_landmarks[14]
    ring_ratio = distance(wrist, ring_tip) / max(distance(wrist, ring_pip), 1e-6)

    pinky_tip = hand_landmarks[20]
    pinky_pip = hand_landmarks[18]
    pinky_ratio = distance(wrist, pinky_tip) / max(distance(wrist, pinky_pip), 1e-6)

    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_ratio = distance(wrist, thumb_tip) / max(distance(wrist, thumb_ip), 1e-6)

    THRESHOLD = 1.2
    THUMB_THRESHOLD = 1.1

    index_open = idx_ratio > THRESHOLD
    middle_open = mid_ratio > THRESHOLD
    ring_closed = ring_ratio <= THRESHOLD
    pinky_closed = pinky_ratio <= THRESHOLD
    thumb_closed = thumb_ratio <= THUMB_THRESHOLD

    return index_open and middle_open and ring_closed and pinky_closed and thumb_closed


def draw_text(image, text, position, color=TEXT_COLOR):
    """Draw text on the image."""
    cv2.putText(image, text, position, FONT, FONT_SCALE, color, FONT_THICKNESS)


# ---------- Main ----------
def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("http://192.168.1.100:8080/video")  # IP camera example

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    prev_time = start_time

    # Toggle state
    blur_active = False
    prev_peace_detected = False
    last_toggle_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            # --- Opsional: balik gambar jika kamera terlihat mirror ---
            if FLIP_IMAGE:
                frame = cv2.flip(frame, 1)   # 1 = flip horizontal (efek cermin)

            current_time = time.time()

            # Convert to RGB and create MediaPipe image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Monotonically increasing timestamp
            timestamp_ms = int((current_time - start_time) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Check if peace gesture is present
            is_peace = False
            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    if detect_peace(landmarks):
                        is_peace = True
                        break

            # Toggle logic: rising edge of peace detection with cooldown
            if is_peace and not prev_peace_detected:
                if current_time - last_toggle_time > TOGGLE_COOLDOWN:
                    blur_active = not blur_active
                    last_toggle_time = current_time
            prev_peace_detected = is_peace

            # Apply blur if active
            if blur_active:
                frame = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

            # Draw landmarks on top
            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    draw_hand(frame, landmarks)

            # FPS
            elapsed = current_time - prev_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            draw_text(frame, f"FPS: {int(fps)}", FPS_POS)

            # Gesture text
            gesture_text = "Gesture : Peace" if is_peace else "Gesture : None"
            gesture_color = PEACE_COLOR if is_peace else NONE_COLOR
            draw_text(frame, gesture_text, GESTURE_POS, gesture_color)

            # Blur status
            blur_text = f"Blur: {'ON' if blur_active else 'OFF'}"
            blur_color = (0, 255, 0) if blur_active else (0, 0, 255)
            draw_text(frame, blur_text, BLUR_POS, blur_color)

            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_time = current_time

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()