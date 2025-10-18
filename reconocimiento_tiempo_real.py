import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
from collections import deque

# Cargar modelo entrenado
model = pickle.load(open("modelo_señas.pkl", "rb"))

# Diccionario de etiquetas a palabras (ajusta según tu dataset)
etiquetas = {
    0: "hola",
    1: "como",
    2: "estas",
    3: "gracias",
    4: "bien",
    5: "adios"
}

# Inicializar MediaPipe y voz
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Variables para control de secuencia
ultima_prediccion = ""
frase_actual = ""
tiempo_ultima_palabra = time.time()
buffer_predicciones = deque(maxlen=8)  # promediamos últimas predicciones

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                puntos = []
                for lm in hand_landmarks.landmark:
                    puntos.append(lm.x)
                    puntos.append(lm.y)

                pred = model.predict([puntos])[0]
                palabra = etiquetas.get(int(pred), "desconocido")

                buffer_predicciones.append(palabra)
                # obtenemos la palabra más común en las últimas N predicciones
                palabra_estable = max(set(buffer_predicciones), key=buffer_predicciones.count)

                # Si la palabra cambia y se mantiene estable por un momento, agregarla
                if palabra_estable != ultima_prediccion and palabra_estable != "desconocido":
                    if time.time() - tiempo_ultima_palabra > 1.5:
                        frase_actual += palabra_estable + " "
                        print("🧩 Frase parcial:", frase_actual)
                        ultima_prediccion = palabra_estable
                        tiempo_ultima_palabra = time.time()

                        # Decir la frase acumulada
                        engine.say(frase_actual)
                        engine.runAndWait()

        # Mostrar texto en pantalla
        cv2.putText(frame, f"Palabra: {ultima_prediccion}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Frase: {frase_actual.strip()}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("Lenguaje de Señas - Traducción Fluida", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            frase_actual = ""  # limpiar frase

cap.release()
cv2.destroyAllWindows()
