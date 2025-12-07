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
import torch.nn.functional as F
import torch.distributed as dist
from Multi_modal_AWB.DETRIS.model.score import ScoreNet
from Multi_modal_AWB.network.models import ColorTemperatureNet
from Multi_modal_AWB.network.ctn import CCT
from model.clip import build_model
from .layers_mamba import Neck, Decoder, Projector, CMFM
from .fusion_mamba import Fusion
from .dinov2.models.vision_transformer import vit_base,vit_large
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d

from Multi_modal_AWB.DETRIS.model.Affinity_mamba import Affinity
from Multi_modal_AWB.DETRIS.model.text_denoise import TextDenoisingModule

from transformers import AutoModel
from timm.data.transforms_factory import create_transform

from timm.models import create_model
from Multi_modal_AWB.utils import option
opt = option.init()

from Multi_modal_AWB.MambaVision.mambavision.models.mamba_vision_my import mamba_vision_T, mamba_vision_B, mamba_vision_L
import torch
class DETRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Text Encoder

        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.txt_backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size, cfg.txtual_adapter_layer,cfg.txt_adapter_dim).float()
        self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
        # self.fusion = Fusion(
        #          d_img=[768, 768, 768],
        #          d_txt=512,
        #          d_model=64,
        #          nhead=8,
        #          num_stages=3,
        #          strides=[1, 1, 1],
        #          num_layers=12,
        #          shared_weights=False,
        #          visual_layers=12,
        #          output_visual_layers=[4, 8],
        #          dim = 768,
        #          visual_adapter_dim = 128
        #         )
       # Fix Backbone
        for param_name, param in self.txt_backbone.named_parameters():
            if 'adapter' not in param_name : 
                param.requires_grad = False       
        # print(self.txt_backbone)

        
        # # Multi-Modal Decoder
        # self.neck = Neck(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        fpn_in = [1024, 1024, 1024] #dinov2 backbone
        fpn_out = [256, 512, 1024]
        self.neck = Neck(in_channels=fpn_in, out_channels=fpn_out, stride=cfg.stride)
        # self.neck = CMFM(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)

        # # Projector
        # self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        # color temperature model
        self.cct = ColorTemperatureNet(start_channels=6)
        # self.cct = CCT()
        self.score = ScoreNet()
        self.text_attention = TextDenoisingModule(512, 512)

        self.contrative = Affinity(1024, 512)

        # mamaba
        model_path = "/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/MambaVision/mambavision_base_21k.pth.tar"
        self.mamba = mamba_vision_B(pretrained=True, model_path = model_path)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, img1, img2, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # print(word.shape)
        # padding mask used in decoder
        # print(type(word))
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        feat = torch.cat((img1, img2), dim=1)
        cct = self.cct(feat)


        # img1's frature 
        avg_pool, vis1, word1, state1= self.fusion(img1, word, self.txt_backbone, self.mamba)
        
        proj_vis1, proj_txt1 = self.contrative(avg_pool, state1)
        state1, word1 = self.text_attention(state1, word1)
        # print(word1.shape)
        # word: b, 77, 512
        # state: b, 512
        # vis[0]: b, 768, 16, 16
        # print(vis1[0].shape)
        # print(state1.shape)
        fq1 = self.neck(vis1, state1)
        # fq1 = self.neck(vis1, word1)
        
        b, c, h, w = fq1.size()
        fq1 = self.decoder(fq1, word1, pad_mask)
        fq1 = fq1.reshape(b, c, h, w)

        # img2's feature
        avg_pool, vis2, word2, state2 = self.fusion(img2, word, self.txt_backbone, self.mamba)
        proj_vis2, proj_txt2 = self.contrative(avg_pool, state2)
        state2, word2 = self.text_attention(state2, word2)
        # return vis, word, state
        # # b, 512, 26, 26 (C4)
        fq2 = self.neck(vis2, state2)
        # fq2 = self.neck(vis2, word2)
        b, c, h, w = fq2.size()
        fq2 = self.decoder(fq2, word2, pad_mask)
        fq2 = fq2.reshape(b, c, h, w)

        score1, score2  = self.score(cct, fq1, fq2)

        
        

        return score1, score2, proj_vis1,proj_txt1, proj_vis2, proj_txt2


if __name__ == '__main__':
    model = DETRIS(opt)
    img1 = torch.randn(3, 3, 224, 224)
    img2 = torch.randn(3, 3, 224, 224)
    txt = torch.randn(3, 77, 512)
    res = model(img1, img2, txt)
    print(res[0])
