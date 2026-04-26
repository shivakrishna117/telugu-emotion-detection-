# policy_nn.py
import torch
import torch.nn as nn

class CNNPolicyNN(nn.Module):
    def __init__(self, n_mfcc=13, n_frames=40, n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # Calculate features: (n_mfcc//8) x (n_frames//8) x 128
        self.flatten_dim = (n_mfcc // 8) * (n_frames // 8) * 128
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 13, 40)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # (batch, 32, 6, 20)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # (batch, 64, 3, 10)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # (batch, 128, 1, 5)
        x = x.reshape(x.size(0), -1)            # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
