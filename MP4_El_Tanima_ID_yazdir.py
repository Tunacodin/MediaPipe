# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:18:12 2024

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
                # Her bir noktayı kırmızı renkte çiz ve ID'yi yanına ekle
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Noktayı kırmızı renkte çiz
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Noktanın ID'sini yaz
                    cv2.putText(image, str(id), (cx + 10, cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Bağlantıları çiz
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Görüntüyü göster
        cv2.imshow("El Takibi", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
