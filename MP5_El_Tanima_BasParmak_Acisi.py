# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:32:07 2024

@author: muozi
"""

import cv2
import mediapipe as mp
import math

# Mediapipe el modeli ve çizim fonksiyonları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# İki vektör arasındaki açıyı hesaplama fonksiyonu
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle)

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Mediapipe el modeli ile el tespiti
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Eller tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Noktaları çiz
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # İşaret parmağı için 3 nokta (5, 6 ve 8 numaralı noktalar)
                index_finger_joint1 = hand_landmarks.landmark[5]
                index_finger_joint2 = hand_landmarks.landmark[6]
                index_finger_tip = hand_landmarks.landmark[8]

                # Bu noktaların ekran koordinatları
                a = [index_finger_joint1.x * image.shape[1], index_finger_joint1.y * image.shape[0]]
                b = [index_finger_joint2.x * image.shape[1], index_finger_joint2.y * image.shape[0]]
                c = [index_finger_tip.x * image.shape[1], index_finger_tip.y * image.shape[0]]

                # Açı hesaplama
                angle = calculate_angle(a, b, c)

                # Sarı renkte noktaları işaretleme
                cv2.circle(image, (int(a[0]), int(a[1])), 10, (0, 255, 255), -1)  # 5. nokta
                cv2.circle(image, (int(b[0]), int(b[1])), 10, (0, 255, 255), -1)  # 6. nokta
                cv2.circle(image, (int(c[0]), int(c[1])), 10, (0, 255, 255), -1)  # 8. nokta

                # Açı bilgisini görüntüye yazdırma
                cv2.putText(image, f"Aci: {int(angle)} derece", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Görüntüyü göster
        cv2.imshow("El Takibi - Açı Ölçümü", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
