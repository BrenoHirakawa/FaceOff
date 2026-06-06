import cv2
import time
import requests 
import onnxruntime as ort
import numpy as np

FIREBASE_URL = "https://bim-detector-default-rtdb.firebaseio.com/status.json"

session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar câmera")
    exit()

cap.set(3, 640)
cap.set(4, 480)

ultimo_envio = 0
INTERVALO_ENVIO = 5 


while True:

    ret, frame = cap.read()

    inicio_frame = time.time()

    if not ret:
        break

    # ==================================================
    # YOLO ONNX
    # ==================================================

    altura_original, largura_original = frame.shape[:2]

    img = cv2.resize(frame, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(
        None,
        {session.get_inputs()[0].name: img}
    )

    output = outputs[0][0]  # (84,2100)

    boxes = []
    scores = []

    for i in range(output.shape[1]):

        classe_person = output[4, i]

        if classe_person < 0.4:
            continue

        cx = output[0, i]
        cy = output[1, i]
        w = output[2, i]
        h = output[3, i]

        x1 = int((cx - w / 2) * largura_original / 256)
        y1 = int((cy - h / 2) * altura_original / 256)

        largura_box = int(w * largura_original / 256)
        altura_box = int(h * altura_original / 256)

        boxes.append([
            x1,
            y1,
            largura_box,
            altura_box
        ])

        scores.append(float(classe_person))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.4,
        nms_threshold=0.45
    )

    pessoas = 0

    if len(indices) > 0:

        for idx in indices.flatten():

            pessoas += 1

            x, y, w, h = boxes[idx]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # cv2.putText(
    #     frame,
    #     f"Pessoas: {pessoas}",
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2
    # )

    # cv2.imshow("FaceOff", frame)

    max_pessoas = max(max_pessoas, pessoas)

    fps = 1 / (time.time() - inicio_frame)

    print(
        f"Pessoas={pessoas} | Máximo={max_pessoas} | FPS={fps:.2f}"
    )

    agora = time.time()

    deve_enviar = (
        pessoas != ultima_quantidade
        or
        (agora - ultimo_envio >= INTERVALO_ENVIO)
    )

    if deve_enviar:

        dados = {
            "quantidade": pessoas,
            "maximo": max_pessoas,
            "fps": round(fps, 2),
            "ambiente": "laboratorio",
            "timestamp": int(agora)
        }

        try:

            response = requests.post(
                FIREBASE_URL,
                json=dados,
                timeout=5
            )

            print(
                f"[FIREBASE] {response.status_code}"
            )

            ultima_quantidade = pessoas
            ultimo_envio = agora

        except Exception as e:

            print(
                f"Erro ao enviar: {e}"
            )

    # if cv2.waitKey(1) == 27: 
    #     break

    time.sleep(0.01)



cap.release()
cv2.destroyAllWindows()