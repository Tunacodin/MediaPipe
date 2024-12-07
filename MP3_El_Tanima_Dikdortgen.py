# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:51:40 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# Mediapipe el modeli ve çizim fonksiyonları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Mediapipe el modeli ile el tespiti
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # OpenCV görüntüyü RGB formatına çevirme
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Görüntüyü geri BGR'ye çevir
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Eller tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Noktaları çiz
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Landmark koordinatlarını al
                landmarks = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in hand_landmarks.landmark]

                # En küçük kutunun koordinatlarını hesapla
                x_coords = [landmark[0] for landmark in landmarks]
                y_coords = [landmark[1] for landmark in landmarks]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                # En küçük dikdörtgeni çiz
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("El Takibi - En Küçük Kutu", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
