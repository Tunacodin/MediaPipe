# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:57:20 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe poz tahmini ve çizim fonksiyonları
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Kamera başlatma
cap = cv2.VideoCapture(0)

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=2,
                  smooth_landmarks=True,
                  min_detection_confidence=0.7,
                  min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # Görüntüyü RGB formatına çevirme
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Görüntüyü geri BGR'ye çevir
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Poz tespit edildiyse
        if results.pose_landmarks:
            # Anahtar noktaları çiz
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Görüntüyü göster
        cv2.imshow("Poz Tahmin Uygulaması", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
