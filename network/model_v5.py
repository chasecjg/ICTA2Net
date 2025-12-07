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



# 加载 ResNet50 模型，使用 weights 参数
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()  # 去掉最后一层全连接层


# 加载 CLIP 模型并冻结参数
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model = clip_model.float()  # 将 CLIP 模型转换为单精度浮点数
for param in clip_model.parameters():
    param.requires_grad = False


class TAWB(nn.Module):
    def __init__(self):
        super(TAWB, self).__init__()
        self.clip = clip_model
        
        # 加载 ResNet50 模型，使用 weights 参数
        self.base_model = resnet
        # # 空间注意力模块
        # self.spatial_attn = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 1, kernel_size=1),
        #     nn.Softmax2d()
        # )
        
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
            nn.Linear(1024,512),
        )
    def forward_single(self, img, text_feat):
        # 提取视觉特征
        # visual_feat = self.clip.encode_image(img)  # [bs, 512]
        visual_feat = self.base_model(img)
        visual_feat = self.linear(visual_feat)
        # print(visual_feat.shape)
        # 空间注意力增强
        spatial_feat = visual_feat.unsqueeze(1)  # [bs, 1, 512]
        spatial_feat = self.feature_enhancer(spatial_feat.unsqueeze(-1))
        spatial_feat = spatial_feat.squeeze()
        # print(spatial_feat.shape)
        
        # 跨模态注意力
        text_feat = self.text_proj(text_feat).unsqueeze(0)  # [1, bs, 512]
        # print(text_feat.shape)
        visual_feat = self.image_proj(spatial_feat).unsqueeze(0)  # [1, bs, 512]
        
        attn_out, _ = self.cross_attn(text_feat, visual_feat, visual_feat)
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
        
        return score1, score2
    

class TAWB_v2(nn.Module):
    def __init__(self):
        super(TAWB_v2, self).__init__()
        self.clip = clip_model
        
        # 加载 ResNet50 模型，使用 weights 参数
        self.base_model = resnet
        # # 空间注意力模块
        # self.spatial_attn = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 1, kernel_size=1),
        #     nn.Softmax2d()
        # )
        
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
            nn.Linear(512, 512)
        )
    def forward_single(self, img, text_feat):
        # 提取视觉特征
        visual_feat = self.clip.encode_image(img)  # [bs, 512]
        # visual_feat = self.base_model(img)
        visual_feat = self.linear(visual_feat)
        # print(visual_feat.shape)
        # 空间注意力增强
        spatial_feat = visual_feat.unsqueeze(1)  # [bs, 1, 512]
        spatial_feat = self.feature_enhancer(spatial_feat.unsqueeze(-1))
        spatial_feat = spatial_feat.squeeze()
        # print(spatial_feat.shape)
        
        # 跨模态注意力
        text_feat = self.text_proj(text_feat).unsqueeze(0)  # [1, bs, 512]
        # print(text_feat.shape)
        visual_feat = self.image_proj(spatial_feat).unsqueeze(0)  # [1, bs, 512]
        
        attn_out, _ = self.cross_attn(text_feat, visual_feat, visual_feat)
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
        
        return score1, score2
    


# # 模拟自定义数据集
# class CustomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         img1 = torch.randn(3, 224, 224)
#         img2 = torch.randn(3, 224, 224)
#         text = "This is a sample text"
#         return img1, img2, text

# if __name__ == '__main__':
#     # 示例使用
#     model = TAWB_v2().to(device)
#     dataset = CustomDataset(num_samples=10)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#     for img1, img2, text in dataloader:
#         img1 = img1.to(device)
#         img2 = img2.to(device)
#         output = model(img1, img2, text)
#         if output is not None:
#             print(output)