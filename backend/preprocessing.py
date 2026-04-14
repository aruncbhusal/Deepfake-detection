import torch
import torchvision.transforms as transforms
import cv2
from backend.config import FRAME_COUNT, IMAGE_SIZE

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization for RGB
])

# Function to extract frames from video
def extract_frames(video_path, num_frames=FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        raise ValueError("Not enough frames in video")

    step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) != num_frames:
        raise ValueError("Frame extraction failed")

    return torch.stack(frames).unsqueeze(0)