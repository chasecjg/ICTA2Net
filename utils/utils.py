import os
import requests

# import torch
import torch
import torch.nn as nn
from torch.autograd import Variable


import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class RankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankingLoss, self).__init__()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, score1, score2, label):
        # label is 1 if score1 should be ranked higher than score2, -1 otherwise
        return self.margin_ranking_loss(score1, score2, label)


class RankNetLoss_v1(nn.Module):
    def __init__(self):
        super(RankNetLoss_v1, self).__init__()

    def forward(self, pred_scores_1, pred_scores_2, labels):
        """
        calculate RankNet loss
        :param pred_scores_1: predicted scores for the first image, shape (batch_size,)
        :param pred_scores_2: predicted scores for the second image, shape (batch_size,)
        :param labels: true labels, 1 indicates the first image is better, -1 indicates the second image is better, shape (batch_size,)
        :return: RankNet loss value
        """
        score_diffs = pred_scores_1 - pred_scores_2
        loss = torch.log(1 + torch.exp(-labels * score_diffs)).mean()
        return loss
    

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, pred_score_1, pred_score_2, labels):
        """
        calculate RankNet loss
        :param pred_score_1: predicted scores for the first sample, shape (batch_size,)
        :param pred_score_2: predicted scores for the second sample, shape (batch_size,)
        :param labels: true labels, 1 indicates the first sample is better, -1 indicates the second sample is better, shape (batch_size,)
        :return: RankNet loss value
        """
        score_difference = pred_score_1 - pred_score_2
        # calculate probabilities
        p_ij = 1 / (1 + torch.exp(-score_difference))
        # clip probabilities to avoid log(0)
        p_ij = torch.clamp(p_ij, min=1e-7, max=1 - 1e-7)
        # map labels
        mapped_labels = 0.5 * (1 + labels)
        # calculate loss
        loss = -torch.mean(mapped_labels * torch.log(p_ij) + (1 - mapped_labels) * torch.log(1 - p_ij))
        return loss
    
    
class ContrastiveLoss(nn.Module):
    """Computes the contrastive loss

    Args:
        - k: the number of transformations per batch
        - temperature: temp to scale before exponential

    Shape:
        - Input: the raw, feature scores.
                tensor of size :math:`(k x minibatch, F)`, with F the number of features
                expects first axis to be ordered by transformations first (i.e., the
                first "minibatch" elements is for first transformations)
        - Output: scalar
    """

    def __init__(self, k: int, temp: float, abs: bool, reduce: str) -> None:
        super().__init__()
        self.k = k
        self.temp = temp
        self.abs = abs
        self.reduce = reduce
        #         self.iter = 0

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        n_samples = len(out)
        assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)

        #         if (self.iter % 100) == 0:
        #             print(sim)
        #          self.iter += 1

        sim = torch.exp(sim * self.temp)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(self.k):
            start, end = i * (n_samples // self.k), (i + 1) * (n_samples // self.k)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if self.reduce == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss






class RankLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        y_true: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        device, dtype = y_pred[0].device, y_pred[0].dtype

        target = torch.ones_like(y_true[0]).to(device).to(dtype)
        # target = target.unsqueeze(-1)
        # Set indices where y_true1 < y_true2 to -1
        target[y_true[0] < y_true[1]] = -1.0

        return F.margin_ranking_loss(
            y_pred[0],
            y_pred[1],
            target,
            margin=self.margin
        )


class RegRankLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.reg_loss = nn.MSELoss(reduction="mean")
        self.rank_loss = RankLoss(margin)

    def forward(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        y_true: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        y_true = (y_true[0].view(-1, 1), y_true[1].view(-1, 1))


        loss_reg = (
            self.reg_loss(y_pred[0], y_true[0]) +
            self.reg_loss(y_pred[1], y_true[1])
        ) / 2.0

        loss_rank = self.rank_loss(y_pred, y_true)
        loss = loss_reg + loss_rank
        return loss, loss_reg, loss_rank
    
    




class ContrastiveLoss(nn.Module):
    """
    calculates the contrastive loss between visual and textual embeddings
    """
    def __init__(self, temperature=0.07):
        """
        Initializes the ContrastiveLoss module.
        
        parameters:
            temperature: temperature scaling parameter, controls the sharpness of the similarity distribution
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, vis_emb, text_emb):
        """
        Forward pass for contrastive loss calculation.
        
        parameters:
            vis_emb: [B, D] - projected visual features
            text_emb: [B, D] - projected textual features
            
        returns:
            Symmetric contrastive loss value
        """
        # Feature normalization
        vis_emb = F.normalize(vis_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # calculate similarity logits
        logits = torch.mm(vis_emb, text_emb.t()) / self.temperature
        
        # auto-generate labels
        batch_size = vis_emb.size(0)
        labels = torch.arange(batch_size, device=vis_emb.device)
        
        # calculate cross-entropy loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2




class GroupContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, debug: bool = False):
        super().__init__()
        self.temperature = temperature
        self.debug = debug
        
    def _validate_mask(self, mask: torch.Tensor, group_ids: torch.Tensor) -> bool:
        """verify mask correctness"""
        gids = group_ids.cpu().numpy()
        mask = mask.cpu().numpy()
        n_samples = len(gids)
        
        # check diagonal elements
        if np.any(np.diag(mask) != 0):
            print("error: diagonal elements of mask should be zero")
            return False
        
        # check same-group and different-group pairs
        for i in range(n_samples):
            for j in range(n_samples):
                if gids[i] == gids[j] and i != j:
                    if mask[i,j] != 1:
                        print(f"error: same-group sample ({i},{j}) not marked as positive")
                        return False
                elif mask[i,j] == 1:
                    print(f"error: different-group sample ({i},{j}) incorrectly marked as positive")
                    return False
        return True
    
    def visualize_mask(self, mask: torch.Tensor, group_ids: torch.Tensor, save_path: str = None):
        """Enhanced mask visualization"""
        mask_np = mask.cpu().numpy()
        gids = group_ids.cpu().numpy()
        unique_gids = np.unique(gids)
        
        plt.figure(figsize=(12, 10))
        
        # Draw group boundaries
        for gid in unique_gids:
            indices = np.where(gids == gid)[0]
            if len(indices) > 0:
                color = plt.cm.tab20(gid % 20)
                start, end = indices[0]-0.5, indices[-1]+0.5
                for pos in [start, end]:
                    plt.axvline(pos, color=color, linestyle='--', alpha=0.7)
                    plt.axhline(pos, color=color, linestyle='--', alpha=0.7)
        
        # Draw mask heatmap
        im = plt.imshow(mask_np, cmap='Blues', vmin=0, vmax=1)
        
        # mask values and group ids
        for i in range(len(gids)):
            for j in range(len(gids)):
                if mask_np[i,j] == 1:
                    plt.text(j, i, f"{gids[i]}", ha='center', va='center', color='red')
                elif i == j:
                    plt.text(j, i, "X", ha='center', va='center', color='black')
        
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Positive Sample Mask\nGroup IDs: {gids}")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def forward(self, 
               vis_feats: Tuple[torch.Tensor, torch.Tensor], 
               txt_feats: Tuple[torch.Tensor, torch.Tensor], 
               group_ids: torch.Tensor) -> torch.Tensor:
        """
       group-wise contrastive loss calculation
        
        parameters:
            vis_feats: (proj_vis1, proj_vis2) image features [B,D]
            txt_feats: (proj_txt1, proj_txt2) text features [B,D]
            group_ids: [B] group-id
            
        returns:
            value of contrastive loss
        """
        # merge features and group ids
        all_vis = torch.cat(vis_feats, dim=0)  # [2B, D]
        all_txt = torch.cat(txt_feats, dim=0)  # [2B, D]
        expanded_gids = group_ids.repeat(2)    # [2B]
        
        # normalize features
        all_vis = F.normalize(all_vis, dim=1)
        all_txt = F.normalize(all_txt, dim=1)
        
        # compute similarity matrix
        sim_matrix = torch.mm(all_vis, all_txt.t()) / self.temperature  # [2B, 2B]
        
        # construct positive sample mask
        pos_mask = torch.eq(
            expanded_gids.unsqueeze(1), 
            expanded_gids.unsqueeze(0)
        ).float()
        pos_mask.fill_diagonal_(0)  # exclude self-comparison
        
        # verify mask correctness
        # if not self._validate_mask(pos_mask, expanded_gids):
        #     raise ValueError("Positive sample mask validation failed!")
        
        # debug visualization
        if self.debug:
            print("="*50)
            print(f"Group IDs (expanded): {expanded_gids.cpu().numpy()}")
            print(f"Similarity Matrix:\n{sim_matrix.detach().cpu().numpy().round(2)}")
            print(f"Positive Mask:\n{pos_mask.cpu().numpy()}")
            self.visualize_mask(
                pos_mask, 
                expanded_gids,
                save_path="/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/utils/vis"
            )
        
        # calculate loss
        exp_sim = torch.exp(sim_matrix)
        pos = (exp_sim * pos_mask).sum(dim=1)  # same of positive similarities
        neg = exp_sim.sum(dim=1) - pos - exp_sim.diag()  # exclude positives and self-similarity
        
        loss = -torch.log(pos / (pos + neg + 1e-8)).mean()
        return loss

if __name__ == '__main__':
    # initialize
    contrast_loss = GroupContrastiveLoss(
        temperature=0.05,
        debug=True  # enable debug mode
    )
    proj_vis1 = torch.randn(8, 512)
    proj_vis2 = torch.randn(8, 512)
    proj_txt1 = torch.randn(8, 512)
    proj_txt2 = torch.randn(8, 512)
    batch_group_ids = torch.tensor([0, 0, 1, 1, 3, 5, 6, 6])
    
    loss = contrast_loss(
        vis_feats=(proj_vis1, proj_vis2),
        txt_feats=(proj_txt1, proj_txt2),
        group_ids=batch_group_ids
    )