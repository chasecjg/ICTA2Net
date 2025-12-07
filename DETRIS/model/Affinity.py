import torch.nn as nn

class Affinity(nn.Module):
    def __init__(self, vis_channel, txt_dim):
        super().__init__()  # 添加这一行以正确初始化父类
        
        self.visual_projection = nn.Sequential(
            nn.Conv2d(vis_channel, 256, kernel_size=1),  # 输入来自dinov2的768维视觉特征
            nn.AdaptiveAvgPool2d(1),                      # [B,256,1,1]
            nn.Flatten()                                  # [B,256]
        )
        self.text_projection = nn.Sequential(
            nn.Linear(txt_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

        # 初始化参数
        nn.init.normal_(self.visual_projection[0].weight, std=0.02)
        for m in self.text_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, vis_feat, txt_feat):
        vis_feat = self.visual_projection(vis_feat)  # shape: [B, 256]
        txt_feat = self.text_projection(txt_feat)    # shape: [B, 256]
        return vis_feat, txt_feat


import torch
if __name__ == '__main__':
    affinity = Affinity(768, 512)
    vis_feat = torch.randn(1, 768, 16, 16)
    txt_feat = torch.randn(1, 512)
    vis_feat, txt_feat = affinity(vis_feat, txt_feat)
    print(vis_feat.shape, txt_feat.shape)  # 应输出: torch.Size([1, 256]) torch.Size([1, 256])