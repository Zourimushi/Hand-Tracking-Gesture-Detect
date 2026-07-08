import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import time
import gesture_detector 

def main():
    tracker = gesture_detector.gestureDetector(
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    tracker.run(
        camera_id=0,
        window_name='MediaPipe Hand Tracking - Gesture Control',
        draw_connections=True,
        draw_indices=False,
        show_finger_info=True,
        flip_frame=True
    )


if __name__ == "__main__":
    main()