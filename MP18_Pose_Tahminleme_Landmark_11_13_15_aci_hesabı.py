# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:05:57 2024

@author: muozi
"""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe poz tahmini ve çizim fonksiyonları
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Açı hesaplama fonksiyonu
def calculate_angle(a, b, c):
    # Açı hesaplama
    a = np.array(a)  # 1. nokta
    b = np.array(b)  # 2. nokta (açının tepe noktası)
    c = np.array(c)  # 3. nokta
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

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
            # 11, 13, 15 numaralı landmark'ları al
            landmarks = results.pose_landmarks.landmark
            point11 = (landmarks[7].x, landmarks[7].y)
            point13 = (landmarks[9].x, landmarks[9].y)
            point15 = (landmarks[10].x, landmarks[10].y)

            # Noktaların gerçek koordinatlarını elde et
            h, w, _ = image.shape
            p11 = (int(point11[0] * w), int(point11[1] * h))
            p13 = (int(point13[0] * w), int(point13[1] * h))
            p15 = (int(point15[0] * w), int(point15[1] * h))

            # Noktaları çiz
            cv2.circle(image, p11, 5, (255, 0, 0), -1)  # Mavi - 11
            cv2.circle(image, p13, 5, (0, 255, 0), -1)  # Yeşil - 13
            cv2.circle(image, p15, 5, (0, 0, 255), -1)  # Kırmızı - 15

            # 11, 13, 15 noktalarını bağla
            cv2.line(image, p11, p13, (255, 255, 0), 2)  # Mavi - 11 -> 13
            cv2.line(image, p13, p15, (255, 255, 0), 2)  # Mavi - 13 -> 15

            # Açı hesapla
            angle = calculate_angle(p11, p13, p15)

            # Açıyı yazdır
            cv2.putText(image, f'Angle: {int(angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Oval şekil çiz
            center = p13
            axes = (40, 20)  # Yarım eksenler
            angle = int(angle)  # Açı
            cv2.ellipse(image, center, axes, angle, 0, 360, (0, 255, 255), 2)

        # Görüntüyü göster
        cv2.imshow("Poz Tahmin Uygulaması", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
