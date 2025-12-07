import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTFeatureExtractor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 定义跨模态注意力模块
class CrossModalAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    
    def forward(self, vis_feat, text_feat):
        # 输入：vis_feat [B, N, D], text_feat [B, M, D]
        Q = self.query(vis_feat)  # [B, N, D]
        K = self.key(text_feat)   # [B, M, D]
        V = self.value(text_feat) # [B, M, D]
        
        attn_weights = torch.matmul(Q, K.transpose(1, 2)) / (self.query.weight.shape[0] ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, N, M]
        
        fused_feat = torch.matmul(attn_weights, V)  # [B, N, D]
        return fused_feat + vis_feat  # 残差连接


# 2. 定义门控融合模块
class GatedFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, vis_feat, text_feat):
        gate = self.gate(torch.cat([vis_feat, text_feat], dim=-1))  # [B, N, D]
        return gate * vis_feat + (1 - gate) * text_feat


# 3. 定义文本引导的注意力模块
class TextDrivenAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.text_proj = nn.Linear(dim, dim)
        self.vis_proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(self, vis_feat, text_feat):
        # 文本特征扩展：text_feat [B, D] -> [B, 1, D]
        text_feat = text_feat.unsqueeze(1)
        
        # 计算相似度
        sim_matrix = torch.matmul(
            self.vis_proj(vis_feat),  # [B, N, D]
            self.text_proj(text_feat).transpose(1, 2)  # [B, D, 1]
        ) / self.temperature  # [B, N, 1]
        
        attn_map = F.softmax(sim_matrix, dim=1)  # [B, N, 1]
        return attn_map * vis_feat  # 加权特征


# 4. 定义主题敏感的区域池化模块
class ThemeAwareROI(nn.Module):
    def __init__(self, output_size=7):
        super().__init__()
        self.roi_align = RoIAlign(output_size, spatial_scale=1.0, sampling_ratio=2)
    
    def forward(self, feature_map, attn_map):
        # 从注意力图中提取候选区域
        B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
        topk_indices = torch.topk(attn_map.flatten(1), k=5, dim=1)[1]  # 取前5个区域
        
        # 将索引转换为坐标
        rois = []
        for idx in topk_indices:
            y = idx // W
            x = idx % W
            rois.append([x - 3, y - 3, x + 3, y + 3])  # 假设区域大小为7x7
        rois = torch.tensor(rois).to(feature_map.device)
        
        pooled_feat = self.roi_align(feature_map, [rois])  # [B, 5, C, 7, 7]
        return pooled_feat.mean(dim=1)  # 全局平均


# 5. 定义完整模型
class TARNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 骨干网络
        self.vision_backbone = ViTModel.from_pretrained("/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/models/vit-base-patch16-224")
        self.text_encoder = CLIPModel.from_pretrained("/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/models/clip-vit-base-patch32").text_model
        
        # 多模态融合
        self.cross_attn = CrossModalAttention(dim=768)
        self.gated_fusion = GatedFusion(dim=768)
        
        # 注意力模块
        self.text_attn = TextDrivenAttention(dim=768)
        self.roi_pool = ThemeAwareROI()
        
        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出美学评分
        )
    
    def forward(self, image, text):
        # 提取特征
        vis_feat = self.vision_backbone(image).last_hidden_state  # [B, 197, 768]
        text_feat = self.text_encoder(**text).last_hidden_state[:, 0, :]  # [B, 768]
        
        # 多模态融合
        fused_feat = self.cross_attn(vis_feat, text_feat.unsqueeze(1))
        fused_feat = self.gated_fusion(fused_feat, text_feat.unsqueeze(1))
        
        # 文本主题注意力
        attn_feat = self.text_attn(vis_feat, text_feat)
        roi_feat = self.roi_pool(attn_feat)
        
        # 最终预测
        global_feat = fused_feat.mean(dim=1)
        combined_feat = torch.cat([global_feat, roi_feat], dim=1)
        score = self.reg_head(combined_feat)
        return score


# 6. 定义损失函数
def pairwise_loss(pred_scores, pairs):
    total_loss = 0
    for (win, lose) in pairs:
        total_loss += torch.relu(1 - (pred_scores[win] - pred_scores[lose]))
    return total_loss / len(pairs)


# 7. 训练逻辑
def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        images, texts, pairs = batch
        images = images.to(device)
        texts = texts.to(device)
        
        # 前向传播
        pred_scores = model(images, texts)
        
        # 计算损失
        loss = pairwise_loss(pred_scores, pairs)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item()}")


# 8. 推理与可视化
def adjust_temp(image, model):
    # 使用 ViTFeatureExtractor 预处理图像
    feature_extractor = ViTFeatureExtractor.from_pretrained("/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/models/vit-base-patch16-224")
    
    # 将 PIL 图像转换为模型输入格式
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    # 提取特征
    with torch.no_grad():
        feat = model.vision_backbone(pixel_values)
        adj_coeff = model.temp_adjust(feat)
    
    # 将调整后的图像转换回 PIL 格式
    adjusted = (pixel_values.squeeze(0).cpu().numpy().transpose(1, 2, 0)) * adj_coeff.numpy()
    adjusted = (adjusted * 255).astype(np.uint8)  # 转换为 0-255 范围
    return Image.fromarray(adjusted)


# 9. Grad-CAM 可视化
class TempGradCAM(GradCAM):
    def __init__(self, model):
        target_layer = model.vision_backbone.blocks[-1].attention
        super().__init__(model, target_layer)
    
    def forward(self, input_img):
        return model(input_img)


# 10. 示例运行
if __name__ == "__main__":
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TARNet().to(device)
    
    # 示例数据
    image = Image.open("/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/a0001-jmac_DSC1459_gt.jpg").convert("RGB")
    text = ["a warm sunset with golden tones"]
    
    # 推理
    adjusted_image = adjust_temp(image, model)
    plt.imshow(adjusted_image)
    plt.show()
    
    # Grad-CAM 可视化
    cam = TempGradCAM(model)
    heatmap = cam(image)
    show_cam_on_image(image, heatmap)