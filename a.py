import onnxruntime as ort

session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)

print("Entradas:")
for inp in session.get_inputs():
    print(inp.name, inp.shape)

print("\nSaídas:")
for out in session.get_outputs():
    print(out.name, out.shape)