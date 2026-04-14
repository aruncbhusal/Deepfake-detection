import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class DeepfakeDetector(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove final classification layer to use as feature extractor
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze early layers for transfer learning so it remains stable
        for layer in list(self.resnet.children())[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        # Bi-LSTM for temporal modeling across frames
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        # Final classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Input shape: (B, T, C, H, W) i.e. (Batch, Time, Channels, Height, Width)
        """
        B, T, C, H, W = x.shape

        # Flatten temporal dimension for CNN
        x = x.view(B * T, C, H, W)

        # Extract spatial features
        x = self.resnet(x)  # (B*T, 512, 1, 1)
        x = x.view(B, T, 512)

        # Temporal modeling
        lstm_out, _ = self.lstm(x)

        # Aggregate temporal features (mean pooling)
        x = torch.mean(lstm_out, dim=1)

        x = self.dropout(x)

        return self.fc(x)