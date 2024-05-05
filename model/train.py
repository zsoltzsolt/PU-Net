import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from Pointnet2.pointnet2 import pointnet2_utils
from chamfer_distance import chamfer_distance
from auction_match import auction_match
from utils1.utils import knn_point
from utils1.data import load_patch_data
from punet import PUNet
from tqdm import tqdm

def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)

class PUNET_Datset(torch.utils.data.Dataset):
    def __init__(self, h5_filename='h5_data/Patches_noHole_and_collected.h5', skip_rate=1, points=1024,
                    random_input=True, norm=True, split='train', is_training=True):
        super(PUNET_Datset, self).__init__()
        self.npoints = points
        self.random_input = random_input
        self.norm = norm
        self.is_training = is_training

        input_, ground_truth, data_radius, object_name = load_patch_data(h5_filename, skip_rate, points, random_input, norm)
        self.input_ = input_
        self.ground_truth = ground_truth
        self.data_radius = data_radius
        self.object_name = object_name

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, index):
        input_data = self.input_[index]
        gt_data = self.ground_truth[index]
        radius_data = self.data_radius[index]

        return input_data[:1024], gt_data, radius_data
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
        return self.get_emd_loss(pred, target, pcd_radius) * 100, \
               self.alpha * self.get_repulsion_loss(pred, target, pcd_radius)


def get_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

if __name__ == '__main__':
    train_dataset = PUNET_Datset(split='train')
    print("Dataset size: ", len(train_dataset))

    if len(train_dataset) == 0:
        print("Train dataset is empty!")
        exit()

    train_loader = DataLoader(dataset=train_dataset, batch_size=2)

    if len(train_loader) == 0:
        print("Train loader is empty!")
        exit()

    model = PUNet().cuda()

    optimizer = get_optimizer(model)
    loss_fn = CustomLoss().cuda()
    print("We are going to train")
    model.train()

    for epoch in range(100):
        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        print("Epoch:", epoch)

        try:
            for batch in tqdm(train_loader):

                optimizer.zero_grad()
                input_data, gt_data, radius_data = batch


                input_data = input_data.float().cuda()
                gt_data = gt_data.float().cuda()
                gt_data = gt_data[..., :3].contiguous()
                radius_data = radius_data.float().cuda()

                preds = model(input_data)
                loss1, rep_loss = loss_fn(preds, gt_data, radius_data)
                loss = loss1 + rep_loss

                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                emd_loss_list.append(loss1.item())
                rep_loss_list.append(rep_loss.item())


        except Exception as e:
            print("Error occurred:", e)
            exit()

        print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, repulsion loss {:.4f}, lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))

