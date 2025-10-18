import cv2
import mediapipe as mp
import csv
import os

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Crear el archivo CSV si no existe
file_path = "dataset.csv"
if not os.path.exists(file_path):
    with open(file_path, mode="w", newline="") as f:
        csv.writer(f).writerow(["label"] + [f"v{i}" for i in range(42)])

# Configurar cámara
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

current_label = 0  # Etiqueta actual (puedes cambiarla con teclas)

print("📸 Presiona las teclas 0-9 para cambiar etiqueta.")
print("🖐 Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen para efecto espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convertir a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Si detecta mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener las coordenadas normalizadas (x, y)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            # Guardar al CSV
            with open(file_path, mode="a", newline="") as f:
                csv.writer(f).writerow([current_label] + coords)

    # Mostrar la etiqueta actual
    cv2.putText(frame, f"Etiqueta: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Captura Dataset", frame)

    # Leer teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif 48 <= key <= 57:  # teclas 0-9
        current_label = key - 48

cap.release()
cv2.destroyAllWindows()
print("✅ Dataset guardado en dataset.csv")
