import torch.nn as nn
import torch
class ScoreNet(nn.Module):
    def __init__(self,enc_dim=128):
        super().__init__()
        # 这里可以添加初始化代码，如果需要的话

        # Global Feature Aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers for Classification
        self.fc = nn.Sequential(nn.Linear(512, 256),
                                nn.Linear(256, 128))
        
        self.linear_cat = nn.Linear(2 * enc_dim, enc_dim)

        self.linear_residual = nn.Linear(enc_dim, enc_dim)
        

        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    # ori
    def forward(self, cct, fq1, fq2):
                     
        # Global Aggregation
        fq1 = self.global_avg_pool(fq1)
        fq1 = fq1.view(fq1.size(0), -1)
        fq1 = self.fc(fq1)
        fq2 = self.global_avg_pool(fq2)
        fq2 = fq2.view(fq2.size(0), -1)
        fq2 = self.fc(fq2)
        # print(fq1.shape)
        # print(cct.shape)
        fq1 = torch.cat((cct, fq1), dim=1)
        fq1 = self.linear_cat(fq1)
        fq1 = fq1 + fq1
        fq1 = self.linear_residual(fq1)
        score1 = self.score(fq1)

        fq2 = torch.cat((cct, fq2), dim=1)
        fq2 = self.linear_cat(fq2)
        fq2 = fq2 + fq2
        fq2 = self.linear_residual(fq2)
        score2 = self.score(fq2)
        return score1, score2
    

    
    # def forward(self, fq1, fq2):
                     
    #     # Global Aggregation
    #     fq1 = self.global_avg_pool(fq1)
    #     fq1 = fq1.view(fq1.size(0), -1)
    #     fq1 = self.fc(fq1)
    #     fq2 = self.global_avg_pool(fq2)
    #     fq2 = fq2.view(fq2.size(0), -1)
    #     fq2 = self.fc(fq2)
    #     # print(fq1.shape)
    #     # print(cct.shape)
    #     fq1 = torch.cat((fq1, fq1), dim=1)
    #     fq1 = self.linear_cat(fq1)
    #     fq1 = fq1 + fq1
    #     fq1 = self.linear_residual(fq1)
    #     score1 = self.score(fq1)

    #     fq2 = torch.cat((fq2, fq2), dim=1)
    #     fq2 = self.linear_cat(fq2)
    #     fq2 = fq2 + fq2
    #     fq2 = self.linear_residual(fq2)
    #     score2 = self.score(fq2)
   

    #     return score1, score2
    

# 引入自适应学习参数
# import torch.nn as nn
# import torch


# class ScoreNet(nn.Module):
#     def __init__(self, enc_dim=128):
#         super().__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(nn.Linear(512, 256),
#                                 nn.Linear(256, 128))
#         self.linear_cat = nn.Linear(2 * enc_dim, enc_dim)
#         self.linear_residual = nn.Linear(enc_dim, enc_dim)
#         self.score = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#         self.weight_net = nn.Sequential(
#             nn.Linear(enc_dim, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 3),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, cct, fq1, fq2):
#         fq1 = self.global_avg_pool(fq1)
#         fq1 = fq1.view(fq1.size(0), -1)
#         fq1 = self.fc(fq1)

#         fq2 = self.global_avg_pool(fq2)
#         fq2 = fq2.view(fq2.size(0), -1)
#         fq2 = self.fc(fq2)
      
#         weights = self.weight_net(cct)
#         alpha = weights[:, 0:1]
#         beta = weights[:, 1:2]
#         gamma = weights[:, 2:3]
        
#         fq1_weighted = alpha * cct + beta * fq1 + gamma * fq2
#         fq2_weighted = alpha * cct + beta * fq2 + gamma * fq1
     
#         fq1_weighted = torch.cat((cct, fq1_weighted), dim=1)
#         fq1_weighted = self.linear_cat(fq1_weighted)
#         fq1_weighted = fq1_weighted + fq1_weighted
#         fq1_weighted = self.linear_residual(fq1_weighted)
#         score1 = self.score(fq1_weighted)
       
#         fq2_weighted = torch.cat((cct, fq2_weighted), dim=1)
#         fq2_weighted = self.linear_cat(fq2_weighted)
#         fq2_weighted = fq2_weighted + fq2_weighted
#         fq2_weighted = self.linear_residual(fq2_weighted)
#         score2 = self.score(fq2_weighted)

#         return score1, score2
    
if __name__ == '__main__':
    model = ScoreNet()
    cct = torch.randn(2, 128)
    fq1 = torch.randn(2, 512, 7, 7)
    fq2 = torch.randn(2, 512, 7, 7)
    res = model(cct, fq1, fq2)
    print(res[0])
    print(res[1])