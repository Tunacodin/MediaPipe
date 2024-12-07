# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:20:58 2024

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
    max_num_hands=1,  # Sadece bir el için
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

        # Parmak sayısı
        finger_count = 0

        # Eller tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Noktaları çiz
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Parmak sayısını belirle
                landmarks = hand_landmarks.landmark

                # Başparmak (Tip nokta: 4)
                if landmarks[4].x < landmarks[3].x:  # Sağ el için kontrol
                    finger_count += 1

                # Diğer dört parmak (Tip noktalar: 8, 12, 16, 20)
                if landmarks[8].y < landmarks[6].y:  # İşaret parmağı açık
                    finger_count += 1
                if landmarks[12].y < landmarks[10].y:  # Orta parmak açık
                    finger_count += 1
                if landmarks[16].y < landmarks[14].y:  # Yüzük parmağı açık
                    finger_count += 1
                if landmarks[20].y < landmarks[18].y:  # Serçe parmak açık
                    finger_count += 1

        # Sağ üst köşeye parmak sayısını yazdır
        cv2.putText(image, str(finger_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        # Görüntüyü göster
        cv2.imshow("El Takibi - Parmak Sayma", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
