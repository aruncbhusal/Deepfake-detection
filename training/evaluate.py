import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, loader, device):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")