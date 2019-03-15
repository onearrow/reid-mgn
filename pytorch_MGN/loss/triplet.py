#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

class TripletSemihardLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, device, margin=0, size_average=True):
        super(TripletSemihardLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.device = device

    def forward(self, input, target):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input)

        pos_mask = same_id
        neg_mask = 1 - same_id

        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor - 1e6 * (1 - mask)
            _max, _idx = torch.max(input_tensor, dim=axis, keepdim=keepdims)
            return _max, _idx

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1e6 * (1 - mask)
            _min, _idx = torch.min(input_tensor, dim=axis, keepdim=keepdims)
            return _min, _idx

        # output[i, j] = || feature[i, :] - feature[j, :] ||_2
        dist_squared = torch.sum(input ** 2, dim=1, keepdim=True) + \
                       torch.sum(input.t() ** 2, dim=0, keepdim=True) - \
                       2.0 * torch.matmul(input, input.t())
        dist = dist_squared.clamp(min=1e-16).sqrt()

        pos_max, pos_idx = _mask_max(dist, pos_mask, axis=-1)
        neg_min, neg_idx = _mask_min(dist, neg_mask, axis=-1)

        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        y = torch.ones(same_id.size()[0]).to(self.device)
        return F.margin_ranking_loss(neg_min.float(),
                                     pos_max.float(),
                                     y,
                                     self.margin,
                                     self.size_average)

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) # 计算两个向量之间的相似度,当两个向量之间的距离大于margin,则loss为正,小于时为0。在一个mini-batch中,loss(x,y)=max(0,−y∗(x1−x2)+margin)。计算输入x1,x2(2个1D张量)与y的损失。
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0) # 获取batch_size
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n) # 每个数平方后,进行加和(通过keepdim保持2维),再扩展成nxn维
        dist = dist + dist.t() # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(1, -2, inputs, inputs.t()) # 然后减去2倍的第i个特征*第j个特征,从而通过完全平方式得到(a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability # 然后开方
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t()) # 这里dist[i][j],=1代表i和j的label相同,=0代表i和j的label不相同
        dist_ap, dist_an = [], []
        # mini-batch中,每个输入分别作为anchor,找到最远的positive和最近的negative,计算图片的embedding的欧式距离得到
        # 但embedding是不断在优化更新的,因此上述组合关系也是在不断变化,每次更新都重新选择将影响效率,所以选择每个mini-batch选择一次
        # 在一个mini-batch中,根据当时的embedding,选择一次三元组,在这些三元组上计算triplet-loss,再对embedding进行更新,不断重复,直到收敛或训练到指定迭代次数
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0)) # 在i与所有有相同label的j的距离中找一个最大的,同id但最不像的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) # 在i与所有不同label的j的距离找一个最小的,不同id但最像的
        dist_ap = torch.cat(dist_ap) # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an) # 声明一个与dist_an相同shape的全1的tensor,y为1或-1,取1时,x1>x2
        loss = self.ranking_loss(dist_an, dist_ap, y) # y的作用更像是一个标志位
        if self.mutual:
            return loss, dist
        return loss
