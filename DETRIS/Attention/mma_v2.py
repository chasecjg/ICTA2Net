
import torch
import torch.nn as nn
from typing import List, Optional
from torch import Tensor

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


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

class Neck(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024],
                 stride = [2, 1, 2], # [1, 1, 1] for vit
                 d_model = 512, nhead = 8):
        super(Neck, self).__init__()
        self.fusion3 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion4 = CrossAttn(d_model=d_model, nhead=nhead)
        self.fusion5 = CrossAttn(d_model=d_model, nhead=nhead)     
        # self.txt_proj = nn.Linear(in_channels[2], out_channels[1])   
        self.txt_proj = nn.Linear(512, out_channels[1])
        self.f3_proj = conv_layer(in_channels[0], out_channels[1], stride[0], 0, stride[0])
        self.f4_proj = conv_layer(in_channels[1], out_channels[1], stride[1], 0, stride[1])
        self.f5_proj = deconv_layer(in_channels[2], out_channels[1], stride[2], 0, stride[2])
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))
        

        

    def forward(self, imgs, state):
        # v3, v4, v5: 512, 52, 52 / 1024, 26, 26 / 512, 13, 13
        v3, v4, v5 = imgs
        # print(v3.shape, v4.shape, v5.shape)
        txt = state.unsqueeze(-1).permute(2, 0, 1)
        v3 = self.f3_proj(v3)
        v4 = self.f4_proj(v4)
        v5 = self.f5_proj(v5)
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

# torch.Size([16, 768, 16, 16])
# torch.Size([16, 512])

#   fpn_in: [768, 768, 768] #dinov2 backbone
#   fpn_out: [256, 512, 1024]

# fq3:torch.Size([16, 512, 16, 16]) fq4:torch.Size([16, 512, 16, 16]) fq5:torch.Size([16, 512, 16, 16])
class MMA(nn.Module):
    def __init__(self, channels=768, emd_dim=77, d_model = 512, nhead = 8):
        super(MMA, self).__init__()

        self.img_proj = conv_layer(768, 512, 1, 0, 1)
        self.att = CrossAttn(d_model=d_model, nhead=nhead)
    def forward(self, img, state):
        txt = state.unsqueeze(-1).permute(2, 0, 1)
        img = self.img_proj(img)
        
        b, c, h, w = img.shape
        img = img.reshape(b, c, -1).permute(2, 0, 1) # b, c, h, w -> b, c, hw -> hw, b, c

        attn_output = self.att(img, txt)
        attn_output = attn_output.permute(1, 2, 0).reshape(b, c, h, w)

        return attn_output


class MMFA(nn.Module):
    def __init__(self, channels=768):
        super(MMFA, self).__init__()

        self.msca = MMA()
           
        self.vis_conv = nn.Sequential(
            nn.Conv2d(channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.vis_conv = BasicConv2d(channels, 512, kernel_size=1, stride=1, padding=0, relu=True)
        self.conv2 =  BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1, relu=True)
    def forward(self, img, state):

        wei = self.msca(img,state)
        
        img = self.vis_conv(img)
        state = state.unsqueeze(-1).unsqueeze(-1)
        xo = img * wei + state* (1 - wei)
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
    txt = torch.randn(16, 512)
    mdoel = MMFA()
    out = mdoel(vis, txt)
    # print(out[0].shape)
    # print(out[1].shape)
    print(out.shape)