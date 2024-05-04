import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from Pointnet2.pointnet2 import pointnet2_utils
from chamfer_distance import chamfer_distance
from auction_match import auction_match
from utils1.utils import knn_point

class CustomLoss(nn.Module):
    def __init__(self, alpha=1., nn_size=5, radius=.07, h=.03, eps=1e-12):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps

    def get_emd_loss(self, pred, target, pcd_radius):
        idx, _ = auction_match(pred, target)
        matched_out = pointnet2_utils.gather_operation(target.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
        dist2 = (pred - matched_out).pow(2)
        dist2 = dist2.view(dist2.shape[0], -1)
        dist2 = torch.mean(dist2, dim=1, keepdim=True)
        dist2 = dist2 / pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, pred, target, pcd_radius):
        cost_for, cost_bac = chamfer_distance(target, pred)
        cost = .8 * cost_for + .2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred, target, pcd_radius):
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        idx = idx[:, :, 1:].to(torch.int32)
        idx = idx.contiguous()
        pred = pred.transpose(1, 2).contiguous()
        grouped_points = pointnet2_utils.grouping_operation(pred, idx)
        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(-dist2 / self.h ** 2)
        uniform_loss = torch.mean((self.radius-dist)*weight)
        return uniform_loss

    def forward(self, pred, target, pcd_radius):
        return self.get_emd_loss(pred, target, pcd_radius)*100, self.alpha*self.get_repulsion_loss(pred, target, pcd_radius)


def get_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)