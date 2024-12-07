import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# MediaPipe yüz mesh
mp_face_mesh = mp.solutions.face_mesh

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Ekran çözünürlüğünü al
screen_width, screen_height = pyautogui.size()

# Gözlerin mesh noktaları
LEFT_EYE = [33, 133, 160, 159, 158, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 373]

# Hassasiyet katsayısı (Hareket oranını artırır)
SENSITIVITY = 65.5  # 2.0 veya daha yüksek değerlerle oynayabilirsiniz

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

        # Görüntüyü yansıt (mirror effect)
        image = cv2.flip(image, 1)
        height, width, _ = image.shape

        # Mesh noktalarını işleme
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Sol gözün orta noktasını hesaplama
                left_eye = np.mean([[face_landmarks.landmark[i].x * width,
                                     face_landmarks.landmark[i].y * height] for i in LEFT_EYE], axis=0)

                # Sağ gözün orta noktasını hesaplama
                right_eye = np.mean([[face_landmarks.landmark[i].x * width,
                                      face_landmarks.landmark[i].y * height] for i in RIGHT_EYE], axis=0)

                # Gözlerin ortalamasını alarak bakış noktasını bulma
                gaze_point = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

                # Fare hareketi için ekran boyutuna ölçekleme
                screen_x = np.interp(gaze_point[0], [0, width], [0, screen_width * SENSITIVITY])
                screen_y = np.interp(gaze_point[1], [0, height], [0, screen_height * SENSITIVITY])

                # Fareyi hareket ettir
                pyautogui.moveTo(screen_x / SENSITIVITY, screen_y / SENSITIVITY)

                # Göz noktalarını çiz
                cv2.circle(image, (int(left_eye[0]), int(left_eye[1])), 5, (0, 255, 0), -1)
                cv2.circle(image, (int(right_eye[0]), int(right_eye[1])), 5, (0, 255, 0), -1)

        # Görüntüyü göster
        cv2.imshow("Goz Takibi ile Fare Kontrolu", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
