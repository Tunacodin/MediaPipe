# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:31:24 2024

@author: muozi
"""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Selfie Segmentation modüllerini başlat
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Webcam'den görüntü almak için video kaynağını başlat
cap = cv2.VideoCapture(0)

# Selfie Segmentation için yapılandırma
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera görüntüsü alınamadı.")
            break

        # Görüntüyü BGR'den RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Segmentasyonu uygula
        results = selfie_segmentation.process(image_rgb)

        # Görüntüyü yeniden yazılabilir hale getir
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Segmentasyon sonucunu alma
        condition = np.zeros(image_bgr.shape, dtype=np.uint8)
        condition[results.segmentation_mask > 0.5] = (255, 255, 255)  # Beyaz arka plan

        # Mavi arka plan
        blue_background = np.zeros(image_bgr.shape, dtype=np.uint8)
        blue_background[:] = (0, 0, 255)  # BGR formatında mavi

        # Son görüntüyü birleştirme
        output_image = np.where(condition == 255, image_bgr, blue_background)

        # Sonuç görüntüsünü göster
        cv2.imshow('Selfie Segmentation with Blue Background', output_image)

        if cv2.waitKey(5) & 0xFF == 27:  # 'ESC' tuşuna basıldığında döngüyü kır
            break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
