import gdown
import os

def download_model():
    os.makedirs("backend/models", exist_ok=True)
    url = f"https://huggingface.co/sharp-y/deepdetect/resolve/main/model.pth?download=true"
    output = "backend/models/deepfake_detector.pth"
    gdown.download(url, output, quiet=False)
