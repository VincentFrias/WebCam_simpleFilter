import cv2
import numpy as np
import streamlit as st

# Configuração da página
st.set_page_config(page_title="Filtros estilo Instagram", layout="wide")
st.title("📸 Filtros estilo Instagram em tempo real")
st.write("Escolha um filtro e veja o efeito em tempo real na sua webcam!")

# Lista de filtros disponíveis
filtros = ["Normal", "Preto e Branco", "Sépia", "Blur", "Negativo", "Desenho", "Cores Vivas"]
filtro_selecionado = st.selectbox("🎨 Selecione um filtro:", filtros)

# Funções de filtro
def aplicar_filtro(frame, tipo):
    if tipo == "Preto e Branco":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif tipo == "Sépia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)
    elif tipo == "Blur":
        return cv2.GaussianBlur(frame, (21, 21), 0)
    elif tipo == "Negativo":
        return cv2.bitwise_not(frame)
    elif tipo == "Desenho":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        return cv2.bitwise_and(color, color, mask=edges)
    elif tipo == "Cores Vivas":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 40)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return frame

# Captura da webcam
frame_window = st.image([])
camera = cv2.VideoCapture(0)

# Loop do vídeo
while True:
    ret, frame = camera.read()
    if not ret:
        st.warning("Não foi possível acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)  # espelha a imagem (como selfie)
    frame_filtro = aplicar_filtro(frame, filtro_selecionado)

    # Converter para RGB (Streamlit usa RGB, OpenCV usa BGR)
    if len(frame_filtro.shape) == 2:
        frame_filtro = cv2.cvtColor(frame_filtro, cv2.COLOR_GRAY2RGB)
    else:
        frame_filtro = cv2.cvtColor(frame_filtro, cv2.COLOR_BGR2RGB)

    frame_window.image(frame_filtro)

camera.release()
