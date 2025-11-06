# Exemplos de Filtros e Processamento de Imagem com OpenCV e MediaPipe

Este repositório contém diversos scripts Python que demonstram como aplicar filtros e efeitos de processamento de imagem em tempo real, utilizando a webcam como fonte de vídeo. Os exemplos variam desde filtros simples (como Preto e Branco e Desfoque) até aplicações mais avançadas, como filtros faciais com stickers e desfoque de fundo.

## Estrutura do Arquivo

| Arquivo | Descrição | Bibliotecas Principais |
| :--- | :--- | :--- |
| `app.py` | Aplicação web interativa construída com Streamlit que permite selecionar e visualizar diversos filtros em tempo real. | `cv2`, `numpy`, `streamlit` |
| `filter.py` | Exemplo básico de uso da webcam aplicando um desfoque simples (suavização). | `cv2`, `numpy`, `PIL` |
| `filter_manual.py` | Demonstra como aplicar um filtro de média (desfoque) manualmente em uma imagem em tons de cinza, iterando sobre os pixels (método didático, mas menos eficiente). | `cv2`, `numpy`, `PIL` |
| `facial_filter_blur.py` | Aplica desfoque no **fundo** (background) do vídeo, mantendo o usuário em foco. Utiliza o modelo de segmentação do MediaPipe. | `cv2`, `numpy`, `mediapipe` |
| `filter_png.py` | Adiciona um **sticker/filtro facial** (ex: óculos) sobre o rosto detectado. Utiliza o modelo de malha facial (`FaceMesh`) do MediaPipe e contém uma função robusta para sobreposição de imagens PNG transparentes (RGBA). | `cv2`, `numpy`, `mediapipe` |

---

## Como Executar

### Pré-requisitos

Certifique-se de ter o Python instalado e as seguintes bibliotecas no seu ambiente:

```bash
pip install opencv-python numpy Pillow mediapipe streamlit