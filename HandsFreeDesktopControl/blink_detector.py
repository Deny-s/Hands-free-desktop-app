# blink_detector.py

import cv2
import numpy as np
import pyautogui
from scipy.spatial import distance as dist
import dlib
import time
import os
from collections import deque

class BlinkDetector:
    def __init__(self, sensitivity='Medium'):
        self.sensitivity = sensitivity
        self.detector = dlib.get_frontal_face_detector()

        # Model dosya yolu
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        self.predictor = dlib.shape_predictor(model_path)

        # Eşik değerleri ve sabitler
        self.right_EYE_AR_THRESH = None  # Kalibrasyondan sonra ayarlanacak
        self.left_EYE_AR_THRESH = None
        self.EYE_AR_CONSEC_FRAMES = 2  # Göz kırpma için gereken ardışık frame sayısı

        # Sayaçlar ve değişkenler
        self.blink_counter = 0
        self.last_blink_time = 0
        self.calibration_frames = 50  # Kalibrasyon için kullanılacak frame sayısı
        self.frame_count = 0
        self.right_EAR_list = []
        self.left_EAR_list = []
        self.right_EAR_series = deque(maxlen=10)  # Hareketli ortalama için
        self.left_EAR_series = deque(maxlen=10)

    def eye_aspect_ratio(self, eye):
        # Öklid mesafelerini hesaplama
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Yatay mesafe
        C = dist.euclidean(eye[0], eye[3])

        # EAR hesaplama
        ear = (A + B) / (2.0 * C)
        return ear

    def calibrate(self):
        # Sağ ve sol gözlerin EAR ortalamalarını hesapla
        right_EAR_avg = np.mean(self.right_EAR_list)
        left_EAR_avg = np.mean(self.left_EAR_list)

        # Eşik değerlerini ortalamaların %75'i olarak ayarla
        self.right_EYE_AR_THRESH = right_EAR_avg * 0.75
        self.left_EYE_AR_THRESH = left_EAR_avg * 0.75

        print(f"Eşik değerleri kalibre edildi:")
        print(f"Right eye threshold value: {self.right_EYE_AR_THRESH:.3f}")
        print(f"Left eye threshold value: {self.left_EYE_AR_THRESH:.3f}")

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if self.right_EYE_AR_THRESH is None or self.left_EYE_AR_THRESH is None:
            # Kalibrasyon aşaması
            if len(faces) > 0:
                face = faces[0]
                landmarks = self.predictor(gray, face)

                # Sağ göz koordinatları
                right_eye = np.array([
                    (landmarks.part(36).x, landmarks.part(36).y),
                    (landmarks.part(37).x, landmarks.part(37).y),
                    (landmarks.part(38).x, landmarks.part(38).y),
                    (landmarks.part(39).x, landmarks.part(39).y),
                    (landmarks.part(40).x, landmarks.part(40).y),
                    (landmarks.part(41).x, landmarks.part(41).y)
                ])

                # Sol göz koordinatları
                left_eye = np.array([
                    (landmarks.part(42).x, landmarks.part(42).y),
                    (landmarks.part(43).x, landmarks.part(43).y),
                    (landmarks.part(44).x, landmarks.part(44).y),
                    (landmarks.part(45).x, landmarks.part(45).y),
                    (landmarks.part(46).x, landmarks.part(46).y),
                    (landmarks.part(47).x, landmarks.part(47).y)
                ])

                right_ear = self.eye_aspect_ratio(right_eye)
                left_ear = self.eye_aspect_ratio(left_eye)

                self.right_EAR_list.append(right_ear)
                self.left_EAR_list.append(left_ear)
                self.frame_count += 1

                cv2.putText(frame, "Start Calibration, Please don't close your eyes...", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.frame_count >= self.calibration_frames:
                    self.calibrate()
            else:
                cv2.putText(frame, "Face is not detected, Hold your face to camera for calibration.", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        for face in faces:
            landmarks = self.predictor(gray, face)

            # Sağ göz koordinatları
            right_eye = np.array([
                (landmarks.part(36).x, landmarks.part(36).y),
                (landmarks.part(37).x, landmarks.part(37).y),
                (landmarks.part(38).x, landmarks.part(38).y),
                (landmarks.part(39).x, landmarks.part(39).y),
                (landmarks.part(40).x, landmarks.part(40).y),
                (landmarks.part(41).x, landmarks.part(41).y)
            ])

            # Sol göz koordinatları
            left_eye = np.array([
                (landmarks.part(42).x, landmarks.part(42).y),
                (landmarks.part(43).x, landmarks.part(43).y),
                (landmarks.part(44).x, landmarks.part(44).y),
                (landmarks.part(45).x, landmarks.part(45).y),
                (landmarks.part(46).x, landmarks.part(46).y),
                (landmarks.part(47).x, landmarks.part(47).y)
            ])

            right_ear = self.eye_aspect_ratio(right_eye)
            left_ear = self.eye_aspect_ratio(left_eye)

            # EAR değerlerini hareketli ortalama için sakla
            self.right_EAR_series.append(right_ear)
            self.left_EAR_series.append(left_ear)

            right_ear_avg = np.mean(self.right_EAR_series)
            left_ear_avg = np.mean(self.left_EAR_series)

            # Görselleştirme için göz çevrelerine çizgi çizme
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)

            # Göz kırpma kontrolü
            # Sadece sağ göz kırpmasıyla tıklama işlemi
            if (right_ear_avg < self.right_EYE_AR_THRESH) and (left_ear_avg > self.left_EYE_AR_THRESH):
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.5:  # İki tıklama arasındaki minimum süre
                        print("Right eye click is detected")
                        pyautogui.click()
                        self.last_blink_time = current_time
                self.blink_counter = 0

        return frame

    def set_sensitivity(self, value):
        self.sensitivity = value
        # Hassasiyet ayarına göre eşik değerini güncelleyebilirsiniz
        # Bu örnekte otomatik kalibrasyon kullanıyoruz
