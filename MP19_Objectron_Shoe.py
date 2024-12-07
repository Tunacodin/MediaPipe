# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:17:54 2024

@author: muozi
"""

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

cap = cv2.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False,
                             max_num_objects=5,
                             min_detection_confidence=0.3,  # Düşük bir güven değeri
                             model_name='Box') as objectron:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera görüntüsü alınamadı.")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = objectron.process(image_rgb)

        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(image_bgr, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

        cv2.imshow('Objectron - 3D Nesne Tespiti', image_bgr)

        if cv2.waitKey(5) & 0xFF == 27:  # 'ESC' tuşuna basıldığında döngüyü kır
            break

cap.release()
cv2.destroyAllWindows()

