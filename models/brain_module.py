import torch
from torch import nn

class BrainModule(nn.Module):
    def __init__(self, input_channels=272, hidden_size=270, output_size=768):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 270, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(270, 270, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(270, 320, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(320, 320, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(320, 2048, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.avg_pool(x) # reducing the time length to 1
        # x = x.view(x.size(0), -1)  # Flatten out the time dimension
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.fc(x)
        return x