import cv2
import mediapipe as mp
from ultralytics import YOLO
from playsound import playsound
import time
import random
import os
import threading

# Config
ALERT_THRESHOLD = 60  # seconds of bad behavior before alert
ALERT_COOLDOWN = 30   # seconds between alerts
SLOUCH_THRESHOLD = 0.1  # forward lean sensitivity
TILT_THRESHOLD = 0.05   # side tilt sensitivity
SOUNDS_DIR = "sounds"

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# YOLO setup (phone detection)
yolo = YOLO("yolov8n.pt")
PHONE_CLASS_ID = 67  # cell phone in COCO

# State tracking
posture_bad_since = None
phone_detected_since = None
last_alert_time = 0


def get_random_sound():
    """Get random sound file from sounds directory."""
    if not os.path.exists(SOUNDS_DIR):
        return None
    sounds = [f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.mp3', '.wav'))]
    return os.path.join(SOUNDS_DIR, random.choice(sounds)) if sounds else None


def play_alert():
    """Play alert sound in background thread."""
    sound = get_random_sound()
    if sound:
        threading.Thread(target=playsound, args=(sound,), daemon=True).start()


def check_posture(landmarks):
    """Check for slouching and side tilt. Returns (is_bad, reason)."""
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Forward slouch: nose Z closer to camera than shoulders
    shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
    forward_lean = shoulder_z - nose.z
    
    # Side tilt: shoulder height difference
    tilt = abs(left_shoulder.y - right_shoulder.y)
    
    if forward_lean > SLOUCH_THRESHOLD:
        return True, "Slouching forward"
    if tilt > TILT_THRESHOLD:
        return True, "Tilting sideways"
    return False, None


def detect_phone(frame):
    """Detect phone in frame using YOLO."""
    results = yolo(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == PHONE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return True, (x1, y1, x2, y2)
    return False, None


def format_time(seconds):
    """Format seconds as M:SS."""
    return f"{int(seconds)//60}:{int(seconds)%60:02d}"


def main():
    global posture_bad_since, phone_detected_since, last_alert_time
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("PostureGuard running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()
        
        # Posture detection
        pose_results = pose.process(rgb)
        posture_bad = False
        posture_reason = None
        
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            posture_bad, posture_reason = check_posture(pose_results.pose_landmarks.landmark)
        
        # Phone detection
        phone_detected, phone_box = detect_phone(frame)
        if phone_box:
            x1, y1, x2, y2 = phone_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "PHONE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update timers
        if posture_bad:
            if posture_bad_since is None:
                posture_bad_since = now
        else:
            posture_bad_since = None
        
        if phone_detected:
            if phone_detected_since is None:
                phone_detected_since = now
        else:
            phone_detected_since = None
        
        # Check alerts
        posture_duration = (now - posture_bad_since) if posture_bad_since else 0
        phone_duration = (now - phone_detected_since) if phone_detected_since else 0
        
        can_alert = (now - last_alert_time) > ALERT_COOLDOWN
        
        if can_alert and (posture_duration >= ALERT_THRESHOLD or phone_duration >= ALERT_THRESHOLD):
            play_alert()
            last_alert_time = now
        
        # Status overlay
        status_y = 30
        if posture_bad:
            remaining = max(0, ALERT_THRESHOLD - posture_duration)
            color = (0, 0, 255) if remaining == 0 else (0, 165, 255)
            cv2.putText(frame, f"! {posture_reason}: {format_time(posture_duration)}/1:00", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            status_y += 30
        
        if phone_detected:
            remaining = max(0, ALERT_THRESHOLD - phone_duration)
            color = (0, 0, 255) if remaining == 0 else (0, 165, 255)
            cv2.putText(frame, f"! Phone detected: {format_time(phone_duration)}/1:00",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            status_y += 30
        
        if not posture_bad and not phone_detected:
            cv2.putText(frame, "Good posture!", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("PostureGuard", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
