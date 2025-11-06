import cv2
import numpy as np

# Carrega o classificador Haar Cascade (incluso no OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Abre a webcam (0 = padrão)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível acessar a câmera.")
else:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Converte para escala de cinza (necessário para o Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostos
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Cria uma cópia borrada da imagem inteira
        blurred = cv2.GaussianBlur(frame, (15, 15), 30)

        # Substitui cada rosto original (nítido) na imagem borrada
        for (x, y, w, h) in faces:
            rosto_nitido = frame[y:y+h, x:x+w]
            blurred[y:y+h, x:x+w] = rosto_nitido

        # Mostra o resultado
        cv2.imshow("Rostos com fundo borrado", blurred)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
