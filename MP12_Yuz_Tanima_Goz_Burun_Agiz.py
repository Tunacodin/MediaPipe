# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 01:34:29 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe yüz mesh ve çizim yardımcıları
mp_face_mesh = mp.solutions.face_mesh

# Landmark grupları
LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]
NOSE_LANDMARKS = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274]
MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]

# Kamera başlatma
cap = cv2.VideoCapture(0)

# MediaPipe FaceMesh modeliyle başlat
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # Görüntüyü RGB formatına çevirme
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Yüz tespit edildiyse ilgili landmarkları çiz
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Fonksiyon: Landmark kontrolü ve çizim
                def draw_landmarks(landmarks, color):
                    for i in landmarks:
                        # Landmark indeksinin mevcut olup olmadığını kontrol et
                        if i < len(face_landmarks.landmark):
                            x = int(face_landmarks.landmark[i].x * image.shape[1])
                            y = int(face_landmarks.landmark[i].y * image.shape[0])
                            cv2.circle(image, (x, y), 2, color, -1)

                # Sol gözü çiz (Yeşil)
                draw_landmarks(LEFT_EYE_LANDMARKS, (0, 255, 0))
                # Sağ gözü çiz (Yeşil)
                draw_landmarks(RIGHT_EYE_LANDMARKS, (0, 255, 0))
                # Sol iris çiz (Kırmızı)
                draw_landmarks(LEFT_IRIS_LANDMARKS, (0, 0, 255))
                # Sağ iris çiz (Kırmızı)
                draw_landmarks(RIGHT_IRIS_LANDMARKS, (0, 0, 255))
                # Burnu çiz (Mavi)
                draw_landmarks(NOSE_LANDMARKS, (255, 0, 0))
                # Ağzı çiz (Sarı)
                draw_landmarks(MOUTH_LANDMARKS, (255, 255, 0))

        # Görüntüyü göster
        cv2.imshow("Yuz Landmarklari", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
