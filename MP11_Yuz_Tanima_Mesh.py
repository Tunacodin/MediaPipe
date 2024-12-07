# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 01:18:54 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe yüz mesh ve çizim fonksiyonları
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
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # Mesh bağlantılarını kullan
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Yeşil nokta
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)  # Kırmızı bağlantı
                )

        # Görüntüyü göster
        cv2.imshow("Yuz Tanima ve Mesh Gösterimi", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
