# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:55:06 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe yüz modeli ve çizim fonksiyonları
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Kamera başlatma
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # OpenCV görüntüyü RGB formatına çevirme
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Yüzler tespit edildiyse
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Yüzün merkezi ve yarıçapı
                center_x = int(x + width / 2)
                center_y = int(y + height / 2)
                radius = int(max(width, height) / 2)

                # Yüzün etrafına daire çiz
                cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)

                # Yüzün güven puanını yazdır
                score = int(detection.score[0] * 100)
                cv2.putText(image, f'Score: {score}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("Yüz Tanıma Uygulaması", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
