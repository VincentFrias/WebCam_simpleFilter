# salvar como face_filter_sticker.py
import cv2
import mediapipe as mp
import numpy as np

# AVISO: O 'AttributeError' e 'ImportError' de protobuf/tensorflow
# persistem a menos que você atualize ou reverta o 'protobuf' (ex: pip install protobuf==3.20.3)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

def overlay_transparent(bg_img, overlay_img, x, y, overlay_size=None):
    """
    Sobrepõe uma imagem PNG (4 canais, RGBA) ou JPG (3 canais) sobre o plano de fundo.
    Gerencia automaticamente o recorte nas bordas da imagem de fundo.
    """
    bg = bg_img.copy()
    
    # 1. Redimensionamento
    if overlay_size is not None:
        overlay_img = cv2.resize(overlay_img, overlay_size, interpolation=cv2.INTER_AREA)

    h, w = overlay_img.shape[:2]
    
    # 2. Caso sem canal Alpha (apenas 3 canais)
    if overlay_img.shape[2] == 3:
        # Apenas sobrepor. Ainda precisamos verificar os limites para evitar erro
        # Atribui x1, y1, x2, y2 com recorte, se necessário.
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, bg.shape[1])
        y2 = min(y + h, bg.shape[0])
        
        # Recorta a imagem de sobreposição também para o tamanho da ROI (Região de Interesse)
        overlay_x1 = x1 - x; overlay_y1 = y1 - y
        overlay_x2 = overlay_x1 + (x2 - x1); overlay_y2 = overlay_y1 + (y2 - y1)
        
        overlay = overlay_img[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        # Realiza a sobreposição
        bg[y1:y2, x1:x2] = overlay
        return bg

    # 3. Caso com canal Alpha (4 canais)
    
    # A. Calcula as coordenadas de recorte (ROI)
    if y + h > bg.shape[0] or x + w > bg.shape[1] or x < 0 or y < 0:
        # Recortar (fora dos limites)
        x1 = max(x, 0); y1 = max(y, 0)
        x2 = min(x + w, bg.shape[1]); y2 = min(y + h, bg.shape[0])
        
        overlay_x1 = x1 - x; overlay_y1 = y1 - y
        overlay_x2 = overlay_x1 + (x2 - x1); overlay_y2 = overlay_y1 + (y2 - y1)
        
        overlay = overlay_img[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        bg_roi = bg[y1:y2, x1:x2]
    else:
        # Sem recorte (dentro dos limites)
        x1 = x; y1 = y
        x2 = x + w; y2 = y + h

        overlay = overlay_img
        bg_roi = bg[y1:y2, x1:x2] # Usa x1, y1 para consistência

    # B. Mistura Alpha
    alpha = overlay[:, :, 3] / 255.0
    inv_alpha = 1.0 - alpha
    
    # Converte bg_roi para float para o cálculo de mistura
    bg_roi_float = bg_roi.astype(float)
    
    for c in range(0, 3):
        bg_roi_float[:, :, c] = (alpha * overlay[:, :, c] + inv_alpha * bg_roi_float[:, :, c])
        
    # C. Atribuição do ROI misturado de volta ao fundo
    # Usa x1 e y1, que agora estão definidos em AMBOS os blocos 'if' e 'else'
    bg[y1:y2, x1:x2] = bg_roi_float.astype(np.uint8)
    
    return bg

# --- RESTO DO CÓDIGO PERMANECE IGUAL (A PARTIR DA LINHA 50) ---

# carregue um PNG com transparência (ex: "glasses.png")
sticker = cv2.imread("./img/glasses.png", cv2.IMREAD_UNCHANGED)
if sticker is None:
    raise FileNotFoundError("Coloque um arquivo 'glasses.png' (RGBA) no diretório '../img/'.")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        # exemplo: usar a ponta do nariz (landmark 1) e largura entre olhos (33 e 263)
        # indexes do FaceMesh: 33 = olho direito (outer), 263 = olho esquerdo (outer)
        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]  # ponta do nariz
        # converter para pixels
        left = np.array([int(left_eye.x * w), int(left_eye.y * h)])
        right = np.array([int(right_eye.x * w), int(right_eye.y * h)])
        nose_pt = (int(nose.x * w), int(nose.y * h))

        # calcular largura do sticker como função da distância dos olhos
        eye_dist = np.linalg.norm(left - right)
        sticker_width = int(eye_dist * 2.0)  # ajuste a constante para caber melhor
        sticker_height = int(sticker_width * sticker.shape[0] / sticker.shape[1])

        # posicionar centralizado acima do nariz (ou ajustar com offsets)
        x = int(nose_pt[0] - sticker_width / 2)
        y = int(nose_pt[1] - sticker_height / 2 - int(eye_dist * 0.3))

        frame = overlay_transparent(frame, sticker, x, y, overlay_size=(sticker_width, sticker_height))

    cv2.imshow("Filtro (q para sair)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()