import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Mediapipe el modeli ve çizim fonksiyonları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Ekran çözünürlüğünü al
screen_width, screen_height = pyautogui.size()

# Hareket kontrolü için değişkenler
prev_x, prev_y = 0, 0
stay_start_time = None  # Parmağın belirli bir bölgede durmaya başladığı zaman

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Mediapipe el modeli ile el tespiti
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.8) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamadı.")
            break

        # Y eksenine göre görüntüyü yansıt
        image = cv2.flip(image, 1)

        # OpenCV görüntüyü RGB formatına çevirme
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Görüntüyü geri BGR'ye çevir
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

        # Eller tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # İşaret parmağı ucu (tip 8)
                index_finger_tip = hand_landmarks.landmark[8]
                index_x = int(index_finger_tip.x * width)
                index_y = int(index_finger_tip.y * height)

                # Ekran koordinatlarına ölçekleme
                screen_x = np.interp(index_x, [0, width], [0, screen_width])
                screen_y = np.interp(index_y, [0, height], [0, screen_height])

                # Fareyi hareket ettir
                pyautogui.moveTo(screen_x, screen_y)

                # Parmağın hareketsiz olup olmadığını kontrol et
                if abs(index_x - prev_x) < 5 and abs(index_y - prev_y) < 5:  # Hareket eşiği
                    if stay_start_time is None:
                        stay_start_time = time.time()  # Durma başlangıç zamanını kaydet
                    elif time.time() - stay_start_time > 1:  # 2 saniyeden fazla hareketsiz
                        pyautogui.click()  # Fare tıklaması yap
                        stay_start_time = None  # Zamanı sıfırla
                else:
                    stay_start_time = None  # Hareket varsa zamanı sıfırla

                # Önceki konumu güncelle
                prev_x, prev_y = index_x, index_y

                # İşaret parmağı ucuna bir daire çizin
                cv2.circle(image, (index_x, index_y), 10, (0, 255, 0), -1)

        # Görüntüyü göster
        cv2.imshow("Parmak ile Fare Kontrolu", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
