import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
import numpy as np
import torch.distributed as dist

from Multi_modal_AWB.DETRIS.model.adapter_my import DenseAligner
from .layers import conv_layer, deconv_layer
import os
from functools import partial



class Fusion(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 dino_layers= 12,
                 output_dinov2 =[4, 8] ,
                ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)

        self.up_conv = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        nn.init.normal_(textual_ctx_vectors, std=0.02)
        self.initialize_parameters()


    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, img, text, txt_backbone, mamba):

        # text encode
        txt = txt_backbone.token_embedding(text).type(
            txt_backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = txt_backbone.transformer
        txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]        
        txt = txt.permute(1, 0, 2)  # BLD -> LBD
        # language
        txt = txt.permute(1, 0, 2)  # LBD -> BLD
        # normalization
        txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # Sentence level features
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ txt_backbone.text_projection# get sentence-level feature Fs


        # forward
        out_avg_pool, features = mamba.forward_features(img)
        outs = []
        # print(out_avg_pool.shape)
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        # print(features[3].shape)
        features[1] = self.up_conv(features[1])
        outs.append(features[1])
        outs.append(features[2])
        outs.append(features[3])
        # torch.Size([1, 640])
        # torch.Size([1, 160, 28, 28])
        # torch.Size([1, 320, 14, 14])
        # torch.Size([1, 640, 7, 7])
        # torch.Size([1, 640, 7, 7])

        output = out_avg_pool, outs , txt, state

        return output



# class Fusion(nn.Module):
#     def __init__(self,
#                  d_img=[768, 768, 768],
#                  d_txt=512,
#                  d_model=64,
#                  nhead=8,
#                  num_stages=3,
#                  strides=[1, 1, 1],
#                  num_layers=12,
#                  shared_weights=False,
#                  visual_layers=12,
#                  output_visual_layers=[4, 8],
#                  dim = 768,
#                  visual_adapter_dim = 128
#                 ):
#         super().__init__()
#         self.d_img = d_img
#         self.d_txt = d_txt
#         self.d_model = d_model
#         self.num_stages = num_stages
#         self.num_layers = num_layers
#         self.visual_layers = visual_layers
#         self.output_visual_layers = output_visual_layers
#         self.n_ctx_visual = 0
#         self.n_ctx_text = 1
#         self.adapter = DenseAligner(dim, visual_adapter_dim, visual_adapter_dim//2, visual_adapter_dim//16, visual_adapter_dim//4, visual_adapter_dim//16, visual_adapter_dim//4) 
#         self.up_conv = nn.Conv2d(320, 768, kernel_size=3, stride=2, padding=1)
#         self.conv_d = nn.Conv2d(640, 768, kernel_size=1, stride=1, padding=0)
#         self.linear = nn.Linear(768, 640)
#         self.initialize_parameters()

#     def initialize_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.02)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, img, text, txt_backbone, visual_backbone):
#         B = img.shape[0]
#         img = img.type(txt_backbone.dtype)
#         vis_outs = []
        
#         # 文本编码部分保持不变
#         txt = txt_backbone.token_embedding(text).type(txt_backbone.dtype)
#         txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]
#         txt = txt.permute(1, 0, 2)  # LBD -> BLD
        
#         # 处理文本编码器
#         txt_enc = txt_backbone.transformer
#         for i in range(self.num_layers):
#             txt = txt_enc.resblocks[i](txt)
#         txt = txt.permute(1, 0, 2)  # LBD -> BLD
#         txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
#         state = txt[torch.arange(txt.shape[0]), text.argmax(dim=-1)] @ txt_backbone.text_projection

#         # 根据视觉骨干类型选择处理方式
#         # if isinstance(visual_backbone, MambaVision):
#         # MambaVision 处理流程
#         avg_pool, outs = visual_backbone.forward_features(img)
        
#         outs[1] = self.up_conv(outs[1])
#         outs[2] = self.conv_d(outs[2])
#         outs[3] = self.conv_d(outs[3])
#         outs_list = []
#         outs_list.append(outs[1])
#         outs_list.append(outs[2])
#         outs_list.append(outs[3])

#         features_visual = []
        
#         # 提取指定层的特征
#         for i, feature in enumerate(outs_list):
#             # if i in self.output_visual_layers:
#             B, C, H, W = feature.shape
#             feature = feature.reshape(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
#             # print(feature.shape)
#             # print(txt.shape)
#             feature = self.adapter(feature, txt) + feature
#             feature = self.linear(feature)
#             features_visual.append(feature)

#         # 添加最后一层特征
#         # B, C, H, W = outs[-1].shape
#         # final_feature = outs[-1].reshape(B, C, H*W).permute(0, 2, 1)
#         # features_visual.append(final_feature)
            
#         # else:
#         #     # 原 DINO 处理流程
#         #     net_input = img.clone()
#         #     B, nc, w, h = net_input.shape
#         #     dino_f = visual_backbone.patch_embed(net_input)
#         #     dino_f = torch.cat((visual_backbone.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
#         #     dino_f = dino_f + visual_backbone.interpolate_pos_encoding(dino_f, w, h)
#         #     dino_f = torch.cat(
#         #         (
#         #             dino_f[:, :1],
#         #             visual_backbone.register_tokens.expand(dino_f.shape[0], -1, -1),
#         #             dino_f[:, 1:],
#         #         ),
#         #         dim=1,
#         #     )
#         #     features_visual = []
            
#         #     # 处理每一层的视觉特征
#         #     for i in range(self.visual_layers):
#         #         dino_f = visual_backbone.blocks[i](dino_f, txt)
#         #         if i in self.output_visual_layers:
#         #             features_visual.append(dino_f)
            
#         #     dino_f = visual_backbone.norm(dino_f)
#         #     features_visual.append(dino_f)
        
#         # 处理视觉特征
#         for i, feature_visual in enumerate(features_visual):
#             # if isinstance(visual_backbone, MambaVision):
#                 # MambaVision 特征已经是 [B, L, C] 格式
#             B, L, C = feature_visual.shape
#             # else:
#             #     # DINO 特征需要移除额外的 tokens
#             #     feature_visual = feature_visual[:, 4 + 1 :]  # 移除 cls_token 和 4 个额外 tokens
#             #     B, L, C = feature_visual.shape
            
#             # 重塑为空间特征
#             H = int(L ** 0.5)
#             W = L // H
#             feature_visual = feature_visual.reshape(B, H, W, C).permute(0, 3, 1, 2)
#             vis_outs.append(feature_visual)
#         # print("*"*100)
#         # print(len(vis_outs))

#         return avg_pool, vis_outs, txt, state