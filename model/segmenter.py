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
from .layers import Neck, Decoder, Projector, CMFM
from .fusion import Fusion
from .dinov2.models.vision_transformer import vit_base,vit_large
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d

from Multi_modal_AWB.DETRIS.model.Affinity import Affinity
from Multi_modal_AWB.DETRIS.model.text_denoise import TextDenoisingModule




class DETRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Text Encoder

        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.txt_backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size, cfg.txtual_adapter_layer,cfg.txt_adapter_dim).float()
        self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
    
       # Fix Backbone
        for param_name, param in self.txt_backbone.named_parameters():
            if 'adapter' not in param_name : 
                param.requires_grad = False       
        # print(self.txt_backbone)
   

        state_dict = torch.load(cfg.dino_pretrain) 
        if cfg.dino_name=='dino-base':
            self.dinov2 = vit_base(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        else:
            self.dinov2=vit_large(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        self.dinov2.load_state_dict(state_dict, strict=False)

        for param_name, param in self.dinov2.named_parameters():
            if 'adapter' not in param_name:
                param.requires_grad = False
        # CFM
        self.neck = Neck(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # CTN
        self.cct = ColorTemperatureNet(start_channels=6)
        # Score
        self.score = ScoreNet()
        # TDM
        self.text_attention = TextDenoisingModule(512, 512)
        
        # Contrative Features
        self.contrative = Affinity(768, 512)

        self.txt_memory = {}
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

        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        feat = torch.cat((img1, img2), dim=1)
        cct = self.cct(feat)


        # img1's frature 
        vis1, word1, state1= self.fusion(img1, word, self.txt_backbone, self.dinov2)
        state1, word1 = self.text_attention(state1, word1)
        proj_vis1, proj_txt1 = self.contrative(vis1[-1], state1)
        
        # print(word1.shape)
        # word: b, 77, 512
        # state: b, 512
        # vis[0]: b, 768, 16, 16

        fq1 = self.neck(vis1, state1)
        
        b, c, h, w = fq1.size()
        fq1 = self.decoder(fq1, word1, pad_mask)
        fq1 = fq1.reshape(b, c, h, w)

        # img2's feature
        vis2, word2, state2 = self.fusion(img2, word, self.txt_backbone, self.dinov2)
        state2, word2 = self.text_attention(state2, word2)
        proj_vis2, proj_txt2 = self.contrative(vis2[-1], state2)
        

        fq2 = self.neck(vis2, state2)

        b, c, h, w = fq2.size()
        fq2 = self.decoder(fq2, word2, pad_mask)
        fq2 = fq2.reshape(b, c, h, w)

        score1, score2  = self.score(cct, fq1, fq2)
        # print(cct.shape, fq1.shape, fq2.shape)
        # return score1, score2, proj_vis1,proj_txt1, proj_vis2, proj_txt2, fq1, fq2, cct
        return score1, score2, proj_vis1,proj_txt1, proj_vis2, proj_txt2

