# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:40:28 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe el modeli ve çizim fonksiyonları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_hand_open(landmarks):
    """
    Elin açık olup olmadığını kontrol eden fonksiyon.
    Açık el için parmak uçları (8, 12, 16, 20) ile kök eklemleri (5, 9, 13, 17) arasındaki uzaklık karşılaştırılır.
    """
    open_fingers = 0
    # Parmak uçları ve kök eklemlerine göre açık parmak sayısını hesaplama
    for tip_id, base_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        if landmarks[tip_id].y < landmarks[base_id].y:  # Parmak ucu kök eklemden yukarıda ise açık
            open_fingers += 1
    # Başparmak kontrolü (4 ve 3 numaralı noktalar)
    if landmarks[4].x > landmarks[3].x:  # Sağ el için başparmak açık
        open_fingers += 1
    
    # En az 4 parmak açıksa eli açık kabul edelim
    return open_fingers >= 4

# Kamera başlatma
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=3,
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
            # Ekrana yazdırmak için başlangıç konumu
            y_position = 50  
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Noktaları çiz
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Elin açık veya kapalı olduğunu kontrol et
                if is_hand_open(hand_landmarks.landmark):
                    status_text = f"{'Sag' if hand_no == 0 else 'Sol'} el: Acik"
                else:
                    status_text = f"{'Sag' if hand_no == 0 else 'Sol'} el: Kapali"

                # Ekrana yazdır
                cv2.putText(image, status_text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_position += 40  # Her el için y konumunu artır

        # Görüntüyü göster
        cv2.imshow("Eller Acik/Kapali Uygulamasi", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
