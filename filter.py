import cv2
import numpy as np
from PIL import Image

# Abre a webcam (0 = webcam padrão)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível acessar a câmera.")
else:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Aplica filtro de escala de cinza
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplica filtro de suavização
        frame = cv2.blur(frame, (15, 15))


        cv2.imshow("Webcam", frame)

        # Sai ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Libera a câmera
camera.release()
