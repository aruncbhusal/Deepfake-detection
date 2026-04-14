import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, frames_path, labels_path):
        self.data = torch.load(frames_path)
        self.labels = torch.load(labels_path)

        if len(self.data) != len(self.labels):
            raise ValueError("Mismatch between frame and label length")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx].float()  # Shape: [T, C, H, W]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label  # Frames will be fed to ResNet, then to LSTM