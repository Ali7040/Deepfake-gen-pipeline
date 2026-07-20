
import onnxruntime
import os

model_path = r'.assets/models/yoloface_8n.onnx'
print(f"Testing model loading: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)} bytes")

try:
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("Success! Model loaded.")
except Exception as e:
    print(f"FAILED to load model. Error:\n{e}")
