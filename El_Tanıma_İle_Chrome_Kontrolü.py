import cv2
import mediapipe as mp
import pygetwindow as gw
import pyautogui
import time

# MediaPipe el modeli ve çizim fonksiyonları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_hand_open(landmarks):
    """Elin açık olup olmadığını kontrol eder."""
    open_fingers = 0
    for tip_id, base_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        if landmarks[tip_id].y < landmarks[base_id].y:
            open_fingers += 1
    if landmarks[4].x > landmarks[3].x:  # Sağ el için başparmak açık
        open_fingers += 1
    return open_fingers >= 4

def is_two_fingers_up(landmarks):
    """İki parmağın yukarıda olup olmadığını kontrol eder."""
    return (
        landmarks[8].y < landmarks[6].y and  # İşaret parmağı
        landmarks[12].y < landmarks[10].y   # Orta parmak
    )

def is_thumb_up(landmarks):
    """Başparmağın yukarıda olup olmadığını kontrol eder."""
    return landmarks[4].y < landmarks[3].y

def is_thumb_down(landmarks):
    """Başparmağın aşağıda olup olmadığını kontrol eder."""
    return landmarks[4].y > landmarks[3].y

# Kamera başlatma
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    is_chrome_minimized = False  # Başlangıç durumu

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
            hand_landmarks = results.multi_hand_landmarks[0]

            # Kontroller
            if is_hand_open(hand_landmarks.landmark):
                status_text = "El Acik"
                if not is_chrome_minimized:
                    chrome_windows = [w for w in gw.getWindowsWithTitle("Chrome") if w.title]
                    for window in chrome_windows:
                        window.minimize()
                    is_chrome_minimized = True
            elif not is_hand_open(hand_landmarks.landmark):
                status_text = "El Kapali"
                if is_chrome_minimized:
                    chrome_windows = [w for w in gw.getWindowsWithTitle("Chrome") if w.title]
                    for window in chrome_windows:
                        window.maximize()
                    is_chrome_minimized = False

            elif is_two_fingers_up(hand_landmarks.landmark):
                status_text = "Ses Artir"
                pyautogui.press("volumeup")
            elif not is_two_fingers_up(hand_landmarks.landmark):
                status_text = "Ses Azalt"
                pyautogui.press("volumedown")

            elif is_thumb_up(hand_landmarks.landmark):
                status_text = "Yeni Sekme"
                pyautogui.hotkey("ctrl", "t")
            elif is_thumb_down(hand_landmarks.landmark):
                status_text = "Sekmeyi Kapat"
                pyautogui.hotkey("ctrl", "w")

        else:
            status_text = "El Bulunamadi"

        # Görüntü üzerine durum yazısı
        cv2.putText(image, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("Eller Acik/Kapali Uygulamasi", image)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
