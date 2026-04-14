import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from backend.model import DeepfakeDetector
from training.dataset import DeepfakeDataset
from training.config import *

def train():
    """
    NOTE:
    This training pipeline was originally used with preprocessed dataset tensors (.pt files)
    generated from the Celeb-DF dataset.

    Due to dataset size and storage constraints, the dataset is not included in this repository.

    To reproduce training:
    1. Obtain Celeb-DF dataset
    2. Run preprocessing pipeline to generate frame tensors
    3. Update paths in config.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DeepfakeDataset(FRAMES_PATH, LABELS_PATH)

    train_size = int(0.75 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = DeepfakeDetector().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch
            }, "deepfake_detector.pth")
            

if __name__ == "__main__":
    train()