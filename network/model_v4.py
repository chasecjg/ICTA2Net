import torch
import torch.nn as nn
import torchvision.models as models
import clip
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import traceback


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50





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






class MultiModalNetwork(nn.Module):
    def __init__(self):
        super(MultiModalNetwork, self).__init__()
        self.resnet = resnet
        self.clip_model = clip_model
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fc = nn.Linear(512, 1)  # 示例输出层

    def forward(self, img1, img2, text):
        try:
            # 提取视觉特征
            visual_feature1 = self.resnet(img1)
            visual_feature1 = visual_feature1.view(visual_feature1.size(0), -1)
            visual_feature2 = self.resnet(img2)
            visual_feature2 = visual_feature2.view(visual_feature2.size(0), -1)
            visual_features = torch.stack([visual_feature1, visual_feature2], dim=1)  # [batch_size, 2, 2048]
            print("#" * 100)
            print("拼接后的图像特征: {}".format(visual_features.shape))
            # 提取文本特征
            text_tokens = clip.tokenize(text).to(device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)  # [batch_size, 512]
            print("文本特征: {}".format(text_features.shape))

            # 调整视觉特征维度以匹配文本特征
            visual_features = nn.Linear(2048, 512).to(device)(visual_features)  # [batch_size, 2, 512]
            print("图文特征: {}".format(visual_features.shape))
            # 多模态注意力融合
            visual_features = visual_features.permute(1, 0, 2)  # [2, batch_size, 512]
            text_features = text_features.unsqueeze(0)  # [1, batch_size, 512]
            attn_output, _ = self.attention(text_features, visual_features, visual_features)
            attn_output = attn_output.squeeze(0)  # [batch_size, 512]

            # 输出层
            output = self.fc(attn_output)
            return output
        except Exception as e:
            print(f"An error occurred in forward pass: {e}")
            traceback.print_exc()
            return None





class NIMA_v5(nn.Module):
    def __init__(self, pretrained_base_model=True, enc_dim=128):
        super(NIMA_v5, self).__init__()
        base_model = resnet50(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.clip_model = clip_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,128),
        )
        
        self.linear_cat = nn.Linear(2 * enc_dim, enc_dim)

        self.linear_residual = nn.Linear(enc_dim, enc_dim)

        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        self.CTN = ColorTemperatureNet(start_channels=6)
    def forward(self, img1, img2, text):

        # 提取文本特征
        text_tokens = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)  # [batch_size, 512]
        

        # 两个图像拼接
        feat = torch.cat((img1, img2), dim=1)
        feat = self.CTN(feat)

        # 处理第一张图像
        feat1 = self.base_model(img1)
        feat1 = feat1.view(feat1.size(0), -1)
        feat1 = self.head(feat1)
        
        # 处理第二张图像
        feat2 = self.base_model(img2)
        feat2 = feat2.view(feat2.size(0), -1)
        feat2 = self.head(feat2)

        out1 = torch.cat((feat, feat1), dim=1)
        out1 = self.linear_cat(out1)
        out1 = feat1 + out1
        out1 = self.linear_residual(out1)
        score1 = self.score(out1)

        out2 = torch.cat((feat, feat2), dim=1)
        out2 = self.linear_cat(out2)
        out2 = feat2 + out2
        out2 = self.linear_residual(out2)
        score2 = self.score(out2)
   

        return score1, score2
    

# 模拟自定义数据集
class CustomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img1 = torch.randn(3, 224, 224)
        img2 = torch.randn(3, 224, 224)
        text = "This is a sample text"
        return img1, img2, text

if __name__ == '__main__':
    # 示例使用
    model = TAWB().to(device)
    dataset = CustomDataset(num_samples=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for img1, img2, text in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        output = model(img1, img2, text)
        if output is not None:
            print(output)