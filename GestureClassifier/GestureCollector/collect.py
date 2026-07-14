import csv
import math
import os
import sys
import time

import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_DETECT_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)
PYTHONSERVER_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..")
)

sys.path.insert(0, PYTHONSERVER_DIR)

from Gesture_Detect.gesture_detector import gestureDetector

SAVE_PATH = os.path.join(GESTURE_DETECT_DIR, "GestureClassifier","dataset", "raw.csv")
SAVE_INTERVAL = 0.3
AUTO_SAVE_COUNT = 100
AUTO_SAVE_FRAME_INTERVAL = 3
COUNTDOWN_SECONDS = 3
# Landmark coordinates are normalized; frames below this average distance are
# treated as the same pose.
SIMILARITY_THRESHOLD = 0.015

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)


def flatten_landmarks(landmarks):
    """Flatten the 21 (x, y, z) landmarks into one CSV-compatible list."""
    return [value for landmark in landmarks for value in landmark]


def is_similar_sample(candidate, samples):
    """Return True when a pose is nearly identical to an already kept pose."""
    for sample in samples:
        squared_distance = sum((a - b) ** 2 for a, b in zip(candidate, sample))
        average_distance = math.sqrt(squared_distance / len(candidate))
        if average_distance < SIMILARITY_THRESHOLD:
            return True
    return False


def ensure_csv_exists():
    if os.path.exists(SAVE_PATH):
        return

    header = ["label", "hand", "score"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    with open(SAVE_PATH, "w", newline="") as file:
        csv.writer(file).writerow(header)


ensure_csv_exists()

sample_count = [0] * 10
with open(SAVE_PATH, "r", newline="") as file:
    for row in csv.DictReader(file):
        try:
            sample_count[int(row["label"])] += 1
        except (KeyError, TypeError, ValueError):
            pass


def save_sample(data, label):
    landmarks = flatten_landmarks(data["landmarks"])
    row = [label, data["handedness"], data["score"], *landmarks]
    with open(SAVE_PATH, "a", newline="") as file:
        csv.writer(file).writerow(row)
    sample_count[label] += 1
    return landmarks


detector = gestureDetector()
current_label = 0
last_save_time = 0
auto_collect = False
auto_label = 0
countdown = False
countdown_start = 0
saved_count = 0
frame_counter = 0
auto_samples = []

print("=" * 40)
print("Gesture Dataset Collector")
print("Press a to start automatic collection.")
print("=" * 40)

while True:
    data = detector.get_training_data()
    if data is None:
        continue

    frame = data["frame"]
    now = time.time()

    cv2.putText(frame, f"Label : {current_label}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples : {sample_count[current_label]}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if data["hand_detected"]:
        cv2.putText(frame, f"Hand : {data['handedness']}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Score : {data['score']:.2f}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No Hand", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if countdown:
        remaining = max(0, COUNTDOWN_SECONDS - (now - countdown_start))
        cv2.putText(frame, f"Auto collect in: {math.ceil(remaining)}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        if remaining <= 0:
            countdown = False
            auto_collect = True
            frame_counter = 0
            print(f"Auto collection started: label {auto_label}, target {AUTO_SAVE_COUNT} unique samples.")

    if auto_collect:
        cv2.putText(frame, f"Auto {auto_label}: {saved_count}/{AUTO_SAVE_COUNT}", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("Dataset Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord("a"):
        if auto_collect or countdown:
            print("Auto collection is already running.")
        else:
            countdown = True
            countdown_start = time.time()
            saved_count = 0
            frame_counter = 0
            auto_samples = []
            auto_label = current_label
            print(f"Auto collection will start in {COUNTDOWN_SECONDS} seconds.")
    elif key == 32:
        if now - last_save_time < SAVE_INTERVAL:
            continue
        if not data["hand_detected"]:
            print("No hand detected.")
            continue

        save_sample(data, current_label)
        last_save_time = now
        print(f"Saved Label {current_label} ({sample_count[current_label]})")
    elif key == ord("c"):
        print("\n========== Dataset ==========")
        total = 0
        for label, count in enumerate(sample_count):
            print(f"{label}: {count}")
            total += count
        print("-----------------------------")
        print("Total:", total)
        print("=============================\n")
    elif ord("0") <= key <= ord("9"):
        current_label = key - ord("0")
        print(f"Current Label -> {current_label}")

    if auto_collect:
        frame_counter += 1
        if frame_counter >= AUTO_SAVE_FRAME_INTERVAL:
            frame_counter = 0
            if not data["hand_detected"]:
                continue

            landmarks = flatten_landmarks(data["landmarks"])
            if is_similar_sample(landmarks, auto_samples):
                print("Skipped similar frame.")
                continue

            save_sample(data, auto_label)
            auto_samples.append(landmarks)
            saved_count += 1
            print(f"Auto saved {saved_count}/{AUTO_SAVE_COUNT} (label {auto_label})")

            if saved_count >= AUTO_SAVE_COUNT:
                auto_collect = False
                print("Auto collection completed.")

detector.cap.release()
cv2.destroyAllWindows()
