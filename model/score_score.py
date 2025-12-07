# import torch.nn as nn
# import torch
# class ScoreNet(nn.Module):
#     def __init__(self,enc_dim=128):
#         super().__init__()
#         # 这里可以添加初始化代码，如果需要的话

#         # Global Feature Aggregation
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

#         # Fully Connected Layers for Classification
#         self.fc = nn.Sequential(nn.Linear(512, 256),
#                                 nn.Linear(256, 128))
        
#         self.linear_cat = nn.Linear(enc_dim, enc_dim)

#         self.linear_residual = nn.Linear(enc_dim, enc_dim)
        

#         self.score = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64,1),
#             nn.Sigmoid()
#         )
#     def forward(self, fq1):
                     
#         # Global Aggregation
#         fq1 = self.global_avg_pool(fq1)
#         fq1 = fq1.view(fq1.size(0), -1)
#         fq1 = self.fc(fq1)



#         fq1 = self.linear_cat(fq1)
#         fq1 = self.linear_residual(fq1)
#         score1 = self.score(fq1)
#         return score1
    


import torch.nn as nn
import torch

class ScoreNet(nn.Module):
    def __init__(self, enc_dim=128):
        super().__init__()
        # 全局特征聚合，将输入特征图池化为 1x1 大小
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 这里不固定输入维度，后续在 forward 中动态确定
        self.fc = nn.Sequential(nn.Linear(512, 256),
                                    nn.Linear(256, 128))
        # 用于拼接特征的线性层
        self.linear_cat = nn.Linear(enc_dim, enc_dim)
        # 用于残差连接的线性层
        self.linear_residual = nn.Linear(enc_dim, enc_dim)
        # 分数预测的全连接层序列
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, fq1):
        # 全局聚合，将特征图池化为 1x1 大小
        fq1 = self.global_avg_pool(fq1)
        # 将特征图展平为一维向量
        fq1 = fq1.view(fq1.size(0), -1)
            
        # 经过全连接层
        fq1 = self.fc(fq1)
        # 经过线性变换
        fq1_cat = self.linear_cat(fq1)
        # 实现残差连接
        fq1_residual = self.linear_residual(fq1) + fq1_cat
        # 预测分数
        score1 = self.score(fq1_residual)
        return score1


# if __name__ == '__main__':
#     # 创建一个示例输入
#     input_tensor = torch.randn(1, 128, 7, 7)
#     # 创建一个示例模型
#     model = ScoreNet()
#     # 前向传播
#     output = model(input_tensor)
#     # 打印输出
#     print(output)