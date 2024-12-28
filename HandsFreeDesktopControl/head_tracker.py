# head_tracker.py

import cv2
import numpy as np
import pyautogui
import dlib
import os

class HeadTracker:
    def __init__(self, sensitivity=50):
        self.sensitivity = sensitivity
        self.detector = dlib.get_frontal_face_detector()

        # Model file path
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.predictor = dlib.shape_predictor(model_path)
        self.screen_width, self.screen_height = pyautogui.size()

        # Previous nose position
        self.prev_nose_point = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            nose_point = (landmarks.part(30).x, landmarks.part(30).y)

            # Calculate movement
            if self.prev_nose_point is not None:
                dx = nose_point[0] - self.prev_nose_point[0]
                dy = nose_point[1] - self.prev_nose_point[1]

                # Adjust sensitivity
                dx = dx * (self.sensitivity / 50)
                dy = dy * (self.sensitivity / 50)

                # Get current mouse position
                mouse_x, mouse_y = pyautogui.position()
                new_x = mouse_x - dx * 5  # Multiply for speed adjustment
                new_y = mouse_y + dy * 5

                # Keep mouse within screen bounds
                new_x = max(0, min(self.screen_width - 1, new_x))
                new_y = max(0, min(self.screen_height - 1, new_y))

                pyautogui.moveTo(new_x, new_y, duration=0.01)

            self.prev_nose_point = nose_point

            # Draw circle on nose for visualization
            cv2.circle(frame, nose_point, 5, (0, 255, 0), -1)
        return frame

    def set_sensitivity(self, value):
        self.sensitivity = value
