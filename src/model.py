import torch
import torch.nn as nn

class ImprovedTextClassifier(nn.Module):
    """
    Custom MLP text classifier built from scratch.
    No pretrained weights used.
    Architecture: 4 fully connected layers with BatchNorm, ReLU, Dropout.
    Total parameters: ~67.7M
    """
    def __init__(self, input_dim=65536, num_classes=24):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
