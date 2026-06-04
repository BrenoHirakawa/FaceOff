import cv2
import time
import requests 
import onnxruntime as ort
import numpy as np

FIREBASE_URL = "https://bim-detector-default-rtdb.firebaseio.com/status.json"

session = ort.InferenceSession(
    "models/yolov8n.onnx",
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

    if not ret:
        break

    pessoas = 0

    cv2.putText(
        frame,
        f"Pessoas: {pessoas}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FaceOff", frame)

    agora = time.time()

    if agora - ultimo_envio >= INTERVALO_ENVIO:

        dados = {
            "quantidade": pessoas,
            "ambiente": "laboratorio",
            "timestamp": int(agora)
        }

        try:

            response = requests.put(
                FIREBASE_URL,
                json=dados,
                timeout=5
            )

            print(
                f"[FIREBASE] {response.status_code} | Pessoas: {pessoas}"
            )

        except Exception as e:

            print(f"Erro ao enviar: {e}")

        ultimo_envio = agora

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()