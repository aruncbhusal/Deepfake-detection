import torch
import os
import logging
from backend.download_model import download_model
from backend.model import DeepfakeDetector
from backend.config import MODEL_PATH

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
    
# Load the trained model
def load_model():
    # Ensure model exists else download
    if not os.path.exists(MODEL_PATH):
        logging.info("Model not found locally. Downloading...")
        download_model()
        
    model = DeepfakeDetector()
    # Load only the model's state_dict
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

# Run inference for prediction
def run_inference(frames_tensor):
    model = get_model()
    
    if frames_tensor.dim() != 5:
        raise ValueError("Invalid input tensor shape for model")
    
    frames_tensor = frames_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(frames_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction