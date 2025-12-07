import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchvision.models.resnet import ResNet50_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

class WBAssessmentModel(nn.Module):
    def __init__(self):
        super(WBAssessmentModel, self).__init__()
        # CLIP模型初始化
        self.clip, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 空间注意力模块（生成RoI建议）
        self.region_proposal = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()  # 输出注意力热图
        )
        
        # RoIAlign参数
        self.roi_align = RoIAlign(output_size=(7,7), spatial_scale=0.25, sampling_ratio=2)
        
        # 多模态融合模块
        self.multimodal_fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        
        # 耦合评分模块
        self.scoring = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 2)  # 同时输出两个耦合分数
        )
    
    def get_roi_features(self, img, text_feat):
        """文本引导的RoI特征提取"""
        # 生成基础特征
        base_feat = self.clip.encode_image(img)
        
        # 生成区域建议
        attn_map = self.region_proposal(img)  # [bs,1,H,W]
        
        # 动态区域选择（选取注意力最高区域）
        bs, _, H, W = attn_map.shape
        coords = []
        for i in range(bs):
            # 找到注意力最大值坐标
            max_idx = attn_map[i].view(-1).argmax()
            y = (max_idx // W) / H  # 归一化坐标
            x = (max_idx % W) / W
            # 生成建议框（中心坐标+宽高）
            coords.append([x, y, 0.3, 0.3])  # 假设框大小为30%
        
        # RoIAlign提取特征
        rois = torch.tensor(coords).to(device)
        roi_feat = self.roi_align(img, [rois])  # [bs,3,7,7]
        
        # 多模态特征融合
        text_feat = text_feat.unsqueeze(1)  # [bs,1,512]
        roi_feat = roi_feat.flatten(2).permute(2,0,1)  # [49,bs,3*7*7]
        fused, _ = self.multimodal_fusion(text_feat, roi_feat, roi_feat)
        return fused.mean(dim=1)  # [bs,512]

    def forward(self, img1, img2, text):
        # 文本特征提取
        text_feat = self.clip.encode_text(clip.tokenize(text).to(device))
        
        # 提取双图特征
        feat1 = self.get_roi_features(img1, text_feat)
        feat2 = self.get_roi_features(img2, text_feat)
        
        # 耦合分数计算
        combined = torch.cat([feat1, feat2], dim=1)  # [bs,1024]
        scores = self.scoring(combined)  # [bs,2]
        
        return scores

# 使用示例

from torch.utils.data import Dataset, DataLoader
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
    model = WBAssessmentModel().to(device)
    dataset = CustomDataset(num_samples=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for img1, img2, text in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        output = model(img1, img2, text)
        if output is not None:
            print(output.shape)