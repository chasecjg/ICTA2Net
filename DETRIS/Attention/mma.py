# import torch
# import torch.nn as nn

# class MSCA(nn.Module):
#     def __init__(self, dim=64, r=4):
#         super(MSCA, self).__init__()
#         out_dim = int(dim // r)
#         # local_att
#         self.local_att = nn.Sequential(
#             nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_dim, dim, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(dim)
#         )

#         # global_att
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_dim, dim, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(dim)
#         )

#         self.sig = nn.Sigmoid()

#     def forward(self, x):

#         xl = self.local_att(x)
#         print(xl.shape)
#         xg = self.global_att(x)
#         print(xg.shape)
#         xlg = xl + xg
#         wei = self.sig(xlg)

#         return wei
    
# if __name__ == '__main__':
#     data = torch.randn(2, 64, 256, 256)
#     model = MSCA()
#     out = model(data)
#     print(out.shape)


import torch
import torch.nn as nn

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu=True, bn=True):
        super(BasicConv1d, self).__init__()
        self.conv =  nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        # print("#"*100)
        # print(x.shape)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        self.back_proj = nn.Linear(embed_dim, image_dim)
        
    def forward(self, image_features, text_features, attention_mask=None):


        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        query = image_features.permute(1, 0, 2)
        key = text_features.permute(1, 0, 2)
        value = text_features.permute(1, 0, 2)
        
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attention_mask)
        
        attn_output = self.back_proj(attn_output)
        
        return attn_output.permute(1, 0, 2)
    

# class MMA(nn.Module):
#     def __init__(self, channels=64, r=4):
#         super(MMA, self).__init__()
#         out_channels = int(channels // r)
#         # local_att for 1D input
#         self.local_att = nn.Sequential(
#             nn.Conv1d(channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(out_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm1d(channels)
#         )

#         # global_att for 1D input
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),  # 1D adaptive pooling
#             nn.Conv1d(channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(out_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm1d(channels)
#         )
#         # 视觉特征：v3:torch.Size([16, 768, 16, 16]) v4:torch.Size([16, 768, 16, 16]) v5:torch.Size([16, 768, 16, 16])
#         # in_channels=[512, 1024, 1024],
#         # out_channels=[256, 512, 1024],
#         # 文本特征：torch.Size([16, 77, 512])
#         self.v_att = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )

#         self.txt_att = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#         )
#         self.Conv1 = nn.Conv2d(512, 512, kernel_size=1)
#         self.sig = nn.Sigmoid()

#     def forward(self, v_feature, txt_feature):
        
#         B, C, H, W = v_feature.shape
#         v1 = self.v_att(v_feature)
#         v1 = v1.reshape(B,C,W*H).permute(0,2,1)

#         txt = self.txt_att(txt_feature)


#         xl = self.local_att(x)
#         xl = self.local_att(x)
#         xg = self.global_att(x)
#         xlg = xl + xg
#         wei = self.sig(xlg)
        
#         return wei
# ****************************************************************************************************
# torch.Size([16, 256, 128])
# torch.Size([16, 77, 512])
# ####################################################################################################
# torch.Size([16, 256, 128])
# torch.Size([16, 77, 128])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# torch.Size([256, 16, 128])
# torch.Size([77, 16, 128])
# torch.Size([77, 16, 128])


# torch.Size([16, 768, 16, 16])
# torch.Size([16, 512])
class MMA(nn.Module):
    def __init__(self, channels=768, emd_dim=77):
        super(MMA, self).__init__()

        # 视觉特征：v3:torch.Size([16, 768, 16, 16]) v4:torch.Size([16, 768, 16, 16]) v5:torch.Size([16, 768, 16, 16])
        # in_channels=[512, 1024, 1024],
        # out_channels=[256, 512, 1024],
        # 文本特征：torch.Size([16, 77, 512])
        self.v_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.txt_att = nn.Sequential(
            nn.Conv1d(emd_dim, emd_dim, kernel_size=1),
            nn.BatchNorm1d(emd_dim),
            nn.ReLU(inplace=True),
        )
        self.Conv1 = nn.Conv2d(channels, channels , kernel_size=1)
        self.sig = nn.Sigmoid()

        self.vis_proj = nn.Linear(channels, 128)
        self.txt_proj = nn.Linear(512, 128)
        self.multihead_attn = nn.MultiheadAttention(128, 8, dropout=0.1)
        self.back_proj = nn.Linear(128, 128)
        self.sig = nn.Sigmoid()
    def forward(self, v_feature, txt_feature):
        
        B, C, H, W = v_feature.shape
        v1 = self.v_att(v_feature)
        v1 = v1.reshape(B,C,W*H).permute(0,2,1)
        v1 = self.vis_proj(v1)

        txt1 = self.txt_att(txt_feature)
        txt1 = self.txt_proj(txt1)

        query = v1.permute(1, 0, 2)
        key = txt1.permute(1, 0, 2)
        value = txt1.permute(1, 0, 2)

        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=None)
        attn_output = self.back_proj(attn_output).permute(1, 0, 2)
        # print(attn_output.shape)
        attn_output = self.sig(attn_output)
        # print(attn_output)
        return attn_output


class MMFA(nn.Module):
    def __init__(self, channels=768):
        super(MMFA, self).__init__()

        self.msca = MMA()
           
        self.v_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.vis_proj = nn.Linear(channels, 128)
        self.txt_proj = nn.Linear(512, 128)
        self.conv1 = BasicConv1d(77, 256, kernel_size=1, stride=1, padding=0, relu=True)
        self.conv2 = BasicConv1d(256, 256, kernel_size=3, stride=1, padding=1, relu=True)
        
    def forward(self, v_feature, txt_feature):
        # print(v_feature.shape)
        # print(txt_feature.shape)
        B, C, H, W = v_feature.shape
        v1 = self.v_att(v_feature)
        v1 = v1.reshape(B,C,W*H).permute(0,2,1)
        v1 = self.vis_proj(v1)
        
        txt_feature_1 = self.txt_proj(txt_feature)
        # print("*"*100)
        # print(txt_feature_1.shape)
        txt_feature_1 = self.conv1(txt_feature_1)

        wei = self.msca(v_feature,txt_feature)
        # print(wei.shape)
        # print(v1.shape)
        # print(txt_feature_1.shape)
        xo = v1 * wei + txt_feature_1 * (1 - wei)
        xo = self.conv2(xo)

        return xo


if __name__ == '__main__':
    # Test with 1D input (batch_size, channels, length)
    # data = torch.randn(2, 77, 512)  # (batch_size, in_channels, input_length)
    # model = MMA(channels=77)
    # out = model(data)
    # print(out.shape)  # Expected output shape: (2, 64, 1)
    # print(out)

    vis = torch.randn(16, 768, 16, 16)
    txt = torch.randn(16, 77, 512)
    mdoel = MMFA()
    out = mdoel(vis, txt)
    # print(out[0].shape)
    # print(out[1].shape)
    print(out.shape)