import os
import requests

def download_model():
    os.makedirs("backend/models", exist_ok=True)

    url = "https://huggingface.co/sharp-y/deepdetect/resolve/main/model.pth"
    output_path = "backend/models/deepfake_detector.pth"

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)