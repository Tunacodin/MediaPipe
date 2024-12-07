# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 01:06:18 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe yüz ve çizim fonksiyonları
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Kamera başlatma
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # OpenCV görüntüyü RGB formatına çevirme
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Yüz tespit edildiyse
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Yüzün köşe noktaları
                h, w, _ = image.shape
                x_min = int(min(landmark.x for landmark in face_landmarks.landmark) * w)
                y_min = int(min(landmark.y for landmark in face_landmarks.landmark) * h)
                x_max = int(max(landmark.x for landmark in face_landmarks.landmark) * w)
                y_max = int(max(landmark.y for landmark in face_landmarks.landmark) * h)

                # Yüzü dikdörtgen içine al
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Landmark noktalarını çiz ve say
                landmark_count = 0
                for landmark in face_landmarks.landmark:
                    landmark_x = int(landmark.x * w)
                    landmark_y = int(landmark.y * h)
                    cv2.circle(image, (landmark_x, landmark_y), 2, (0, 0, 255), -1)  # Kırmızı nokta
                    landmark_count += 1  # Her nokta için sayacı artır

                # Landmark sayısını ekrana yazdır
                cv2.putText(image, f'Landmark Sayisi: {landmark_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Görüntüyü göster
        cv2.imshow("Yuz Tanima ve Landmark Cizme", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()

