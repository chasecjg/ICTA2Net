import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import fvcore.nn.weight_init as weight_init

from Multi_modal_AWB.DETRIS.Attention.mma_v2 import MMFA
from Multi_modal_AWB.DETRIS.Attention.mmfe import MMFE, RF

# from Multi_modal_AWB.DETRIS.Attention.mma import MMFA

from typing import Optional
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def deconv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

class CrossAttn(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


import torch
import torch.nn as nn
from typing import Optional

class CrossAttnEnhanced(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Gating机制
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 融合后的微小MLP提升
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, vis, txt,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        """
        vis: (HW, B, C) -> Visual features
        txt: (L, B, C) -> Sentence-level text features
        """
        # Step 1: 做 Cross Attention
        attn_output, _ = self.multihead_attn(
            query=self.with_pos_embed(vis, query_pos),
            key=self.with_pos_embed(txt, pos),
            value=txt,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )  # 输出是 (HW, B, C)
        
        # Step 2: Gating机制控制交互强度
        vis_flat = vis.transpose(0,1)  # (B, HW, C)
        attn_flat = attn_output.transpose(0,1)  # (B, HW, C)
        
        concat = torch.cat([vis_flat, attn_flat], dim=-1)  # (B, HW, 2C)
        gate = self.gate_fc(concat)  # (B, HW, C)
        
        gated_attn = gate * attn_flat  # (B, HW, C)
        
        # Step 3: Residual加回原始视觉特征
        fused = vis_flat + self.dropout(gated_attn)  # (B, HW, C)
        
        # Step 4: 小MLP进一步融合特征
        fused = self.norm(fused)
        fused = fused + self.mlp(fused)  # (B, HW, C)
        
        # Step 5: 再转回 (HW, B, C)形式
        fused = fused.transpose(0,1)  # (HW, B, C)
        
        return fused





# 方案1:单词级的没用
class BiCrossModalFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        # Cross-Attn: Visual <-- Sentence Text
        self.vis2txt_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-Attn: Visual <-- Word-level Text
        self.vis2word_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-Attn: Text <-- Visual
        self.txt2vis_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Gating机制融合不同注意路径
        self.gate_proj = nn.Linear(d_model * 3, d_model)
        self.fusion_proj = nn.Linear(d_model * 3, d_model)

        # 残差块（Norm + FFN）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def with_pos(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, vis, txt_sent, txt_word,
                vis_pos=None, txt_sent_pos=None, txt_word_pos=None,
                vis_mask=None, txt_mask=None):

        # ---- Visual attends to Sentence-level Text ----
        vis2txt, _ = self.vis2txt_attn(
            query=self.with_pos(vis, vis_pos),
            key=self.with_pos(txt_sent, txt_sent_pos),
            value=txt_sent,
            key_padding_mask=txt_mask
        )

        # ---- Visual attends to Word-level Text ----
        vis2word, _ = self.vis2word_attn(
            query=self.with_pos(vis, vis_pos),
            key=self.with_pos(txt_word, txt_word_pos),
            value=txt_word,
            key_padding_mask=txt_mask
        )

        # ---- Text attends to Visual (句子级) ----
        txt2vis, _ = self.txt2vis_attn(
            query=self.with_pos(txt_sent, txt_sent_pos),
            key=self.with_pos(vis, vis_pos),
            value=vis,
            key_padding_mask=vis_mask
        )

        # ---- 融合视觉特征路径 ----
        fusion_input = torch.cat([vis, vis2txt, vis2word], dim=-1)
        gate = torch.sigmoid(self.gate_proj(fusion_input))
        fused_vis = gate * vis2txt + (1 - gate) * vis2word + vis  # 加原始残差
        fused_vis = self.norm1(fused_vis)

        # ---- FFN ----
        fused_vis = fused_vis + self.ffn(fused_vis)
        fused_vis = self.norm2(fused_vis)

        # ---- 返回视觉融合后的特征，以及文本增强后的句子特征 ----
        return fused_vis



# 方案2:单词级的使用
class BiCrossModalFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        # Cross-Attn: Visual <-- Sentence Text
        self.vis2txt_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-Attn: Visual <-- Word-level Text
        self.vis2word_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-Attn: Text <-- Visual
        self.txt2vis_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Gating机制融合不同注意路径
        self.gate_proj = nn.Linear(d_model * 3, d_model)  # Gate to control the attention paths
        self.fusion_proj = nn.Linear(d_model * 3, d_model)

        # 残差块（Norm + FFN）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def with_pos(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, vis, txt_sent, txt_word,
                vis_pos=None, txt_sent_pos=None, txt_word_pos=None,
                vis_mask=None, txt_mask=None):

        # ---- Visual attends to Sentence-level Text ----
        vis2txt, _ = self.vis2txt_attn(
            query=self.with_pos(vis, vis_pos),
            key=self.with_pos(txt_sent, txt_sent_pos),
            value=txt_sent,
            key_padding_mask=txt_mask
        )

        # ---- Visual attends to Word-level Text ----
        vis2word, _ = self.vis2word_attn(
            query=self.with_pos(vis, vis_pos),
            key=self.with_pos(txt_word, txt_word_pos),
            value=txt_word,
            key_padding_mask=txt_mask
        )

        # ---- Text attends to Visual (句子级) ----
        txt2vis, _ = self.txt2vis_attn(
            query=self.with_pos(txt_sent, txt_sent_pos),
            key=self.with_pos(vis, vis_pos),
            value=vis,
            key_padding_mask=vis_mask
        )

        # ---- 融合视觉特征路径 ----
        # 我们将 vis, vis2txt 和 vis2word 进行拼接，并通过门控机制控制每个路径的贡献
        fusion_input = torch.cat([vis, vis2txt, vis2word], dim=-1)  # (B, HW, 3C)
        
        # 通过门控网络生成门控系数
        gate = torch.sigmoid(self.gate_proj(fusion_input))  # (B, HW, C)

        # 使用门控机制融合
        # gate 控制 vis2txt 和 vis2word 对最终输出的贡献
        fused_vis = gate * vis2txt + (1 - gate) * vis2word + vis  # 加原始视觉特征作为残差
        fused_vis = self.norm1(fused_vis)

        # ---- FFN (Feed-Forward Network) ----
        fused_vis = fused_vis + self.ffn(fused_vis)
        fused_vis = self.norm2(fused_vis)

        # ---- 返回视觉融合后的特征，以及文本增强后的句子特征 ----
        return fused_vis



class Neck(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024],
                 stride = [2, 1, 2], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(Neck, self).__init__()
        # self.fusion3 = CrossAttnEnhanced(d_model=d_model, nhead=nhead)
        # self.fusion4 = CrossAttnEnhanced(d_model=d_model, nhead=nhead)
        # self.fusion5 = CrossAttnEnhanced(d_model=d_model, nhead=nhead)     
        # self.txt_proj = nn.Linear(in_channels[2], out_channels[1])   
        self.txt_proj = nn.Linear(512, out_channels[1])
        self.f3_proj = conv_layer(in_channels[0], out_channels[1], stride[0], 0, stride[0])
        self.f4_proj = conv_layer(in_channels[1], out_channels[1], stride[1], 0, stride[1])
        self.f5_proj = deconv_layer(in_channels[2], out_channels[1], stride[2], 0, stride[2])
        # aggregation
        # self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        # self.coordconv = nn.Sequential(
        #     CoordConv(out_channels[1], out_channels[1], 3, 1),
        #     conv_layer(out_channels[1], out_channels[1], 3, 1))
        

        

    def forward(self, imgs, state):

        # word = word.permute(1, 0, 2)
        # v3, v4, v5: 512, 52, 52 / 1024, 26, 26 / 512, 13, 13，大小均为：[24, 768, 16, 16]
        v3, v4, v5 = imgs
        # print(v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)
        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)
        # print(v3.shape, v4.shape, v5.shape)
        txt = self.txt_proj(txt)
        # print(v3.shape, v4.shape, v5.shape)
        # fusion v3 
        b, c, h, w = v3.shape
        v3 = v3.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c
        # fusion v4 
        b, c, h, w = v4.shape
        v4 = v4.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c  
        # fusion v5 
        b, c, h, w = v5.shape
        v5 = v5.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c       




        # ori
        # fq3 = self.fusion3(v3, txt)  
        # # print(fq3.shape)      
        # fq3 = fq3.permute(1, 2, 0).reshape(b, c, h, w)
           
        # # v4 = self.downsample(v4)
        # fq4 = self.fusion4(v4, txt)     
        # # print(fq4.shape)   
        # fq4 = fq4.permute(1, 2, 0).reshape(b, c, h, w)
        
        # fq5 = self.fusion5(v5, txt)
        # # print(fq5.shape)
        # fq5 = fq5.permute(1, 2, 0).reshape(b, c, h, w)

        # # print(fq3.shape)
        # # print(fq4.shape)
        # # print(fq5.shape)
        # # fusion 4: b, 512, 26, 26 / b, 512, 26, 26 / b, 512, 26, 26
        # # query
        # # print(fq3.shape, fq4.shape, fq5.shape)
        # fq = torch.cat([fq3, fq4, fq5], dim=1)
        # # print(fq.shape)
        # fq = self.aggr(fq)
        # fq1 = self.coordconv(fq)

        # fq = fq1 + fq




        fq = (v3 + v4 + v5 + txt).permute(1, 2, 0).reshape(b, c, h, w)
        # b, 512, 26, 26
        return fq


# v1版本，输入为img和word,实测训练一个epoch之后准确率很低
# class CMFM(nn.Module):
#     def __init__(self,
#                  in_channels=[512, 1024, 1024], out_channels=[256, 512, 1024]):
#         super(CMFM, self).__init__()
#         # aggregation
#         self.aggr = conv_layer(out_channels[1], out_channels[1], 1, 0)
#         self.coordconv = nn.Sequential(
#             CoordConv(out_channels[1], out_channels[1], 3, 1),
#             conv_layer(out_channels[1], out_channels[1], 3, 1))
#         self.conv = nn.Conv2d(3 * 128, 512, 1)
#         self.mmfa = MMFA()
        

#     def forward(self, imgs, state):
#         # print(state.shape)
#         # v3, v4, v5: 512, 52, 52 / 1024, 26, 26 / 512, 13, 13
#         v3, v4, v5 = imgs
#         # print(v3.shape)
#         # print(state.shape)

#         b, c, h, w = v3.shape
#         fq3 = self.mmfa(v3, state).permute(0, 2, 1).reshape(b, -1, h, w)
   
#         b, c, h, w = v4.shape
#         fq4 = self.mmfa(v4, state).permute(0, 2, 1).reshape(b, -1, h, w)

#         b, c, h, w = v5.shape
#         fq5 = self.mmfa(v5, state).permute(0, 2, 1).reshape(b, -1, h, w)

#         fq = torch.cat([fq3, fq4, fq5], dim=1)
#         # print(fq.shape)
#         fq = self.conv(fq)
#         # print(fq.shape)
#         fq = self.aggr(fq)
#         fq1 = self.coordconv(fq)

#         fq = fq1 + fq

#         return fq


# v2版本，输入为img和state，保留了原始的输入特征
# class CMFM(nn.Module):
#     def __init__(self,
#                  in_channels=[512, 1024, 1024],
#                  out_channels=[256, 512, 1024],
#                  stride = [2, 1, 2], # [1, 1, 1] for vit
#                  d_model = 512, nhead = 8):
#         super(CMFM, self).__init__()

#         # aggregation
#         self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
#         self.coordconv = nn.Sequential(
#             CoordConv(out_channels[1], out_channels[1], 3, 1),
#             conv_layer(out_channels[1], out_channels[1], 3, 1))
#         self.mmfa = MMFA()

#     def forward(self, imgs, state):
#         # v3, v4, v5: 512, 52, 52 / 1024, 26, 26 / 512, 13, 13
#         v3, v4, v5 = imgs

#         fq3 = self.mmfa(v3, state)
#         fq4 = self.mmfa(v4, state)
#         fq5 = self.mmfa(v5, state)

#         # print(fq3.shape, fq4.shape, fq5.shape)
#         fq = torch.cat([fq3, fq4, fq5], dim=1)
#         # print(fq.shape)
#         fq = self.aggr(fq)
#         fq1 = self.coordconv(fq)

#         fq = fq1 + fq
#         # b, 512, 26, 26
#         return fq


# V3版本，原始的前缀卷积改成了增强模块
class CMFM(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024],
                 stride = [2, 1, 2], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(CMFM, self).__init__()
        self.fusion3 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion4 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion5 = CrossAttn(d_model=d_model, nhead=nhead)     
        # self.txt_proj = nn.Linear(in_channels[2], out_channels[1])   
        self.txt_proj = nn.Linear(512, out_channels[1])
        # self.f3_proj = conv_layer(in_channels[0], out_channels[1], stride[0], 0, stride[0])
        # self.f4_proj = conv_layer(in_channels[1], out_channels[1], stride[1], 0, stride[1])
        # self.f5_proj = deconv_layer(in_channels[2], out_channels[1], stride[2], 0, stride[2])
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        
        # self.mmfe = MMFE(in_channels[2], out_channels[1])
        self.mmfe = RF(in_channels[2], out_channels[1])
        

    def forward(self, imgs, state):
        # v3, v4, v5: 512, 52, 52 / 1024, 26, 26 / 512, 13, 13
        v3, v4, v5 = imgs
              
        # print(v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)
        
        # 特征增强：
        v3 = self.mmfe(v3)
        v4 = self.mmfe(v4)
        v5 = self.mmfe(v5)

        # v3 = self.f3_proj(v3)
        # v4 = self.f4_proj(v4)
        # v5 = self.f5_proj(v5)

        txt = self.txt_proj(txt)
        # print(v3.shape, v4.shape, v5.shape)
        # fusion v3 
        b, c, h, w = v3.shape
        v3 = v3.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c
        # fusion v4 
        b, c, h, w = v4.shape
        v4 = v4.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c  
        # fusion v5 
        b, c, h, w = v5.shape
        v5 = v5.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c       

        # print(v3.shape, txt.shape)
        fq3 = self.fusion3(v3, txt)  
        # print(fq3.shape)      
        fq3 = fq3.permute(1, 2, 0).reshape(b, c, h, w)
           
        # v4 = self.downsample(v4)
        fq4 = self.fusion4(v4, txt)     
        # print(fq4.shape)   
        fq4 = fq4.permute(1, 2, 0).reshape(b, c, h, w)
        
        fq5 = self.fusion5(v5, txt)
        # print(fq5.shape)
        fq5 = fq5.permute(1, 2, 0).reshape(b, c, h, w)
        # fusion 4: b, 512, 26, 26 / b, 512, 26, 26 / b, 512, 26, 26
        # query
        # print(fq3.shape, fq4.shape, fq5.shape)
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        # print(fq.shape)
        fq = self.aggr(fq)
        fq1 = self.coordconv(fq)

        fq = fq1 + fq
        # b, 512, 26, 26
        return fq

class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class Decoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




if __name__ == '__main__':
    model = CMFM()
    # 创建三个形状为(16,512,16,16)的张量
    x = [torch.randn(16,512,16,16) for _ in range(3)]
    txt = torch.randn(16, 512)
    out = model(x, txt)