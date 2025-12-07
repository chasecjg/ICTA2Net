
import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorTemperatureNet(nn.Module):
    def __init__(self, num_classes=128, start_channels=3):
        super(ColorTemperatureNet, self).__init__()
        # Initial Convolution Layer (adapted for 6-channel input)
        self.conv1 = nn.Conv2d(start_channels, 64, kernel_size=3, stride=1, padding=1)  # Input channels = 6
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
        
        # Feature Refinement
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global Feature Aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers for Classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial Attention
        sa = self.spatial_attention(x.mean(dim=1, keepdim=True))
        x = x * sa
        
        # Feature Refinement
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Global Aggregation
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
