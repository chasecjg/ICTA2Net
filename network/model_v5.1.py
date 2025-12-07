import torch
import torch.nn as nn
import torchvision.models as models
import clip
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 加载 ResNet50 模型，使用 weights 参数
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()  # 去掉最后一层全连接层

# 加载 CLIP 模型并冻结参数
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model = clip_model.float()  # 将 CLIP 模型转换为单精度浮点数
for param in clip_model.parameters():
    param.requires_grad = False


class WhiteBalanceFeatureExtractor(nn.Module):
    def __init__(self):
        super(WhiteBalanceFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class TAWB(nn.Module):
    def __init__(self):
        super(TAWB, self).__init__()
        self.clip = clip_model

        # 加载 ResNet50 模型，使用 weights 参数
        self.base_model = resnet

        # 空间注意力模块
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Softmax2d()
        )

        # 多模态融合模块
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.text_proj = nn.Linear(512, 512)
        self.image_proj = nn.Linear(512, 512)

        # 评分模块
        self.score_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # 特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

        # 特征降维
        self.linear = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        )

        # 白平衡特征提取器
        self.wb_extractor = WhiteBalanceFeatureExtractor()

    def forward_single(self, img, text_feat):
        # 提取视觉特征
        visual_feat = self.base_model(img)
        visual_feat = self.wb_extractor(visual_feat)

        # 空间注意力增强
        visual_feat = visual_feat.unsqueeze(1)  # [bs, 1, 512]
        spatial_attn = self.spatial_attn(visual_feat)
        visual_feat = visual_feat * spatial_attn
        visual_feat = visual_feat.squeeze(1)

        # 特征增强
        spatial_feat = visual_feat.unsqueeze(1)  # [bs, 1, 512]
        spatial_feat = self.feature_enhancer(spatial_feat.unsqueeze(-1))
        spatial_feat = spatial_feat.squeeze()

        # 跨模态注意力
        text_feat = self.text_proj(text_feat).unsqueeze(0)  # [1, bs, 512]
        visual_feat = self.image_proj(spatial_feat).unsqueeze(0)  # [1, bs, 512]

        # 计算相似度
        similarity = F.cosine_similarity(text_feat, visual_feat, dim=-1)
        similarity = similarity.unsqueeze(-1)

        # 调整注意力权重
        attn_out, _ = self.cross_attn(text_feat * similarity, visual_feat, visual_feat)
        fused_feat = attn_out.squeeze(0)

        # 计算分数
        return self.score_predictor(fused_feat)

    def forward(self, img1, img2, text):
        # 文本特征提取
        text_tokens = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_feat = self.clip.encode_text(text_tokens)

        # 双流处理
        score1 = self.forward_single(img1, text_feat)
        score2 = self.forward_single(img2, text_feat)

        # 比较分数，输出排名
        ranking = (score1 > score2).float()  # 1 表示 img1 更优，0 表示 img2 更优
        return ranking


if __name__ == '__main__':
    # 示例使用
    model = TAWB().to(device)
    img1 = torch.randn(1, 3, 224, 224).to(device)
    img2 = torch.randn(1, 3, 224, 224).to(device)
    text = ["A well - balanced image"]
    output = model(img1, img2, text)
    print(output)