import cv2
from ultralytics import YOLO

# Carrega modelo leve (nano)
model = YOLO("yolov8n.pt")

# Inicializa câmera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar câmera")
    exit()

# (Importante pra Raspberry) reduzir resolução
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferência (tamanho menor = mais rápido)
    results = model(frame, imgsz=320, conf=0.4, verbose=False)

    pessoas = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # Classe 0 = pessoa (COCO dataset)
            if cls == 0:
                pessoas += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar contagem
    cv2.putText(frame, f'Pessoas: {pessoas}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - Monitoramento", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()