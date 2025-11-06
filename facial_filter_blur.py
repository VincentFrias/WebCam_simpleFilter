# salvar como background_blur.py
import cv2
import mediapipe as mp
import numpy as np

mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = seg.process(rgb)
    if results.segmentation_mask is None:
        mask = np.zeros((h, w), dtype=np.float32)
    else:
        mask = results.segmentation_mask

    # binarizar / suavizar máscara
    _, mask_bin = cv2.threshold((mask * 255).astype('uint8'), 127, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.medianBlur(mask_bin, 15)
    mask_f = mask_bin.astype(float) / 255.0
    mask_3c = cv2.merge([mask_f, mask_f, mask_f])

    # criar fundo desfocado
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)

    output = (frame.astype(float) * mask_3c + blurred.astype(float) * (1 - mask_3c)).astype('uint8')

    cv2.imshow("Background Blur (q para sair)", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

