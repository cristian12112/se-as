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
        csv.writer(f).writerow(["label"] + [f"v{i}" for i in range(84)])  # 2 manos = 42 * 2

# Configurar cámara
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 👈 Permitir dos manos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

current_label = 0  # Etiqueta actual (puedes cambiarla con teclas)

print("📸 Presiona las teclas 0-9 para cambiar etiqueta.")
print("🖐 Presiona 'q' para salir.")
print("✋ Usa ambas manos si quieres capturar señas con las dos.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen (efecto espejo)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convertir a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Si detecta manos
    if results.multi_hand_landmarks:
        coords_totales = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Guardar coordenadas (x, y) para cada mano
            coords = []
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            coords_totales.extend(coords)

        # 🔒 Asegurar longitud exacta (dos manos = 84 coordenadas)
        # Si solo hay una mano, se rellena con ceros
        if len(coords_totales) < 84:
            coords_totales += [0.0] * (84 - len(coords_totales))

        # Guardar en el CSV solo si hay 84 valores
        if len(coords_totales) == 84:
            with open(file_path, mode="a", newline="") as f:
                csv.writer(f).writerow([current_label] + coords_totales)
        else:
            print(f"⚠️ Coordenadas incompletas ({len(coords_totales)})")

    # Mostrar etiqueta actual
    cv2.putText(frame, f"Etiqueta: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Captura Dataset (2 manos)", frame)

    # Leer teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif 48 <= key <= 57:  # teclas 0–9
        current_label = key - 48

cap.release()
cv2.destroyAllWindows()
print("✅ Dataset guardado en dataset.csv")
