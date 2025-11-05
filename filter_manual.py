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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        altura = len(frame)
        largura = len(frame[0])

        #cria uma nova imagem com a altura e largura da imagem original
        imagem_gray_med =  [[0 for _ in range(largura)] for _ in range(altura)] 

        for i in range(1,altura - 1):       # i = linha
            for j in range(1,largura - 1):  # j = coluna
                #soma2 = np.sum(foto_gray[i-1:i+2, j-1:j+2])
                #print("soma1:",(soma2 / 9))
                soma = (
                        float(frame[i-1, j-1]) + float(frame[i-1, j]) + float(frame[i-1, j+1]) +
                        float(frame[i,   j-1]) + float(frame[i,   j]) + float(frame[i,   j+1]) +
                        float(frame[i+1, j-1]) + float(frame[i+1, j]) + float(frame[i+1, j+1])
                )
                
                imagem_gray_med[i][j] = (soma // 9)
                #print("soma2:",soma/9)

        array_med = np.array(imagem_gray_med, dtype=np.uint8)

        # Cria uma imagem em tons de cinza (modo 'L')
        imagem_med = Image.fromarray(array_med, mode='L')
        frame = cv2.cvtColor(np.array(imagem_med), cv2.COLOR_GRAY2BGR)


        cv2.imshow("Webcam", frame)

        # Sai ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Libera a câmera
camera.release()
