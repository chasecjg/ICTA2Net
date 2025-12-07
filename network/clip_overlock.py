import sys, os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from os import path
d = path.dirname(__file__)
parent_path = os.path.dirname(d)
sys.path.append(parent_path)

import torch
import torch.nn as nn
import clip
from timm.models.factory import create_model
from Multi_modal_AWB.utils import option
from Multi_modal_AWB.OverLoCK.models import overlock_xt

from timm.models import create_model
def creat_over_locl_midel(opt):
    model = create_model(
        opt.model,
        pretrained=opt.pretrained,
        num_classes=opt.num_classes,
        drop_rate=opt.drop,
        drop_path_rate=opt.drop_path,
        # img_size=opt.input_size,
        use_checkpoint=opt.ckpt_stg if opt.grad_checkpoint else [0] * 4,
    )
    return model


class DualEncoder(nn.Module):
    def __init__(self, opt):
        super(DualEncoder, self).__init__()
        self.vis_model = creat_over_locl_midel(opt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.text_encoder = clip_model.encode_text
        self.fusion_layer = nn.Linear(1000 + 512, 256)  # 视觉编码器输出 1000 维，CLIP 文本特征 512 维
        self.output_layer = nn.Linear(256, 1)  # 最终输出一个分数

    def forward(self, images, texts):
        # 视觉特征提取
        vis_features = self.vis_model(images)
        if self.training:
            vis_features = vis_features["main"]
        # 文本特征提取
        # texts = clip.tokenize(texts).to(self.device)
        # print("texts.shape", texts.shape)
        with torch.no_grad():
            text_features = self.text_encoder(texts)
        text_features = text_features.to(self.device)
        # print("text_features.shape", text_features.shape)
        # print("vis_features.shape", vis_features.shape)
        # 特征融合
        combined_features = torch.cat((vis_features, text_features), dim=1)
        fused_features = torch.relu(self.fusion_layer(combined_features))

        # 输出分数
        score = self.output_layer(fused_features)
        return score
    


class DualEncoder_v2(nn.Module):
    def __init__(self, opt):
        super(DualEncoder_v2, self).__init__()
        self.vis_model = creat_over_locl_midel(opt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.text_encoder = clip_model.encode_text
        self.fusion_layer = nn.Linear(1000 + 512, 256)  # 视觉编码器输出 1000 维，CLIP 文本特征 512 维
        self.output_layer = nn.Linear(256, 1)  # 最终输出一个分数
        self.linear1 = nn.Linear(1000, 512)
        self.linear2 = nn.Linear(512, 512)
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,1),
            nn.Sigmoid()
        ) 
    def forward(self, images1, images2, texts):
        # 视觉特征提取
        vis_features1 = self.vis_model(images1)
        vis_features2 = self.vis_model(images2)
        
        if self.training:
            vis_features1 = vis_features1["main"]
            vis_features2 = vis_features2["main"]
        
        vis_features1 = self.linear1(vis_features1)
        vis_features2 = self.linear1(vis_features2)
        # texts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.text_encoder(texts)
        text_features = text_features.to(self.device)
        # 对视觉特征和文本特征l2归一化
        vis_features1 = torch.nn.functional.normalize(vis_features1, dim=-1)
        vis_features2 = torch.nn.functional.normalize(vis_features2, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        vis_txt1 = vis_features1 + text_features
        vis_txt2 = vis_features2 + text_features 
        vis_txt1 = self.linear2(vis_txt1)
        vis_txt2 = self.linear2(vis_txt2)
        # print("text_features.shape", text_features.shape)
        # print("vis_features.shape", vis_features.shape)
        # 分数计算
        score1 = self.score(vis_txt1)
        score2 = self.score(vis_txt2)

        return score1, score2


# 示例使用
# if __name__ == "__main__":
#     opt = option.init()
#     model = DualEncoder_v2(opt)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     # 模拟输入
#     images1 = torch.randn(4, 3, 224, 224).to(device)
#     images2 = torch.randn(4, 3, 224, 224).to(device)
#     texts = ["a relevant description", "another relevant description", "a third description", "a fourth description"]

#     # 前向传播
#     output1, output2 = model(images1, images2, texts)
#     print("Output shape 1:", output1.shape)
#     print("Output shape 2:", output2.shape)
#     print("Output 1:", output1)
#     print("Output 2:", output2)
