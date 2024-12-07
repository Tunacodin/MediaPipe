# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:24:16 2024

@author: muozi
"""

import cv2
import mediapipe as mp

# MediaPipe Objectron için gerekli modüller
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# Video kaynağını başlat (web kamerası)
cap = cv2.VideoCapture(0)

# Tanınabilir nesne türlerinin listesi
model_names = ['Cup', 'Shoe', 'Chair', 'Laptop', 'Book', 'Bottle', 'Box', 'TV']

# Her nesne için döngü başlat
for model_name in model_names:
    with mp_objectron.Objectron(static_image_mode=False,
                                 max_num_objects=5,
                                 min_detection_confidence=0.1,  # Düşük bir güven değeri
                                 model_name=model_name) as objectron:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Kamera görüntüsü alınamadı.")
                break

            # Görüntüyü BGR'den RGB'ye çevir
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Nesne tespitini uygula
            results = objectron.process(image_rgb)

            # Görüntüyü yeniden yazılabilir hale getir
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Tespit edilen nesneleri çiz
            if results.detected_objects:
                print(f"Tespit edilen nesne: {model_name}")
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(image_bgr, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

            # Sonuç görüntüsünü göster
            cv2.imshow(f'Objectron - 3D Nesne Tespiti ({model_name})', image_bgr)

            if cv2.waitKey(5) & 0xFF == 27:  # 'ESC' tuşuna basıldığında döngüyü kır
                break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
