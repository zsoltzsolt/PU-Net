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
from rich.progress import track
import argparse

class PUNET_Datset(torch.utils.data.Dataset):
    """
        Dataset class for loading and processing point cloud data for PUNET.

        Parameters:
        h5_filename (str): Path to the HDF5 file containing the dataset. Default is 'h5_data/Patches_noHole_and_collected.h5'.
        skip_rate (int): Rate at which to skip data samples when loading the dataset. Default is 1.
        points (int): Number of points to sample from each point cloud. Default is 1024.
        random_input (bool): Whether to randomize the input points. Default is True.
        norm (bool): Whether to normalize the input points. Default is True.
        split (str): Dataset split to load ('train', 'test', etc.). Default is 'train'.
        is_training (bool): Whether the dataset is being used for training. Default is True.

        Attributes:
        input_ (numpy.ndarray): Array of input point clouds.
        ground_truth (numpy.ndarray): Array of ground truth point clouds.
        data_radius (numpy.ndarray): Array of radii for each point cloud.
        object_name (list): List of object names corresponding to each point cloud.
    """
    def __init__(self, h5_filename='h5_data/Patches_noHole_and_collected.h5', skip_rate=1, points=1024,
                 random_input=True, norm=True, split='train', is_training=True):
        super(PUNET_Datset, self).__init__()
        self.npoints = points
        self.random_input = random_input
        self.norm = norm
        self.is_training = is_training

        input_, ground_truth, data_radius, object_name = load_patch_data(h5_filename, skip_rate, points, random_input,
                                                                         norm)

        self.input_ = input_
        self.ground_truth = ground_truth
        self.data_radius = data_radius
        self.object_name = object_name

        with open('./data/{}_list.txt'.format(split), 'r') as f:
            split_choice = [int(x) for x in f]
        self.ground_truth = self.ground_truth[split_choice, ...]
        self.input_ = self.input_[split_choice, ...]

    def __len__(self):
        """
              Returns the number of samples in the dataset.

              Returns:
              int: Number of samples in the dataset.
        """
        return len(self.input_)

    def __getitem__(self, index):
        """
                Returns a tuple of input data, ground truth data, and radius data for the given index.

                Parameters:
                index (int): Index of the data sample to retrieve.

                Returns:
                tuple: A tuple containing:
                    - input_data (numpy.ndarray): Input point cloud data (first 1024 points).
                    - gt_data (numpy.ndarray): Ground truth point cloud data.
                    - radius_data (float): Radius of the point cloud.
        """
        input_data = self.input_[index]
        gt_data = self.ground_truth[index]
        radius_data = self.data_radius[index]

        return input_data[:1024], gt_data, radius_data


class CustomLoss(nn.Module):
    """
        Custom loss module that combines Earth Mover's Distance (EMD) loss, Chamfer Distance (CD) loss,
        and a repulsion loss for point cloud data.

        Parameters:
        alpha (float): Weighting factor for the repulsion loss. Default is 1.0.
        nn_size (int): Number of nearest neighbors to consider for the repulsion loss. Default is 5.
        radius (float): Radius parameter used in the repulsion loss. Default is 0.07.
        h (float): Bandwidth parameter used in the repulsion loss. Default is 0.03.
        eps (float): Small value to prevent division by zero in the repulsion loss. Default is 1e-12.
    """
    def __init__(self, alpha=1., nn_size=5, radius=.07, h=.03, eps=1e-12):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps

    def get_emd_loss(self, pred, target, pcd_radius):
        """
                Computes the Earth Mover's Distance (EMD) loss between the predicted and target point clouds.

                Parameters:
                pred (torch.Tensor): Predicted point cloud.
                target (torch.Tensor): Ground truth point cloud.
                pcd_radius (torch.Tensor): Radius of the point cloud.

                Returns:
                torch.Tensor: Computed EMD loss.
        """
        idx, _ = auction_match(pred, target)
        matched_out = pointnet2_utils.gather_operation(target.transpose(1, 2).contiguous(), idx).transpose(1,
                                                                                                           2).contiguous()
        dist2 = (pred - matched_out).pow(2)
        dist2 = dist2.view(dist2.shape[0], -1)
        dist2 = torch.mean(dist2, dim=1, keepdim=True)
        dist2 = dist2 / pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, pred, target, pcd_radius):
        """
                Computes the Chamfer Distance (CD) loss between the predicted and target point clouds.

                Parameters:
                pred (torch.Tensor): Predicted point cloud.
                target (torch.Tensor): Ground truth point cloud.
                pcd_radius (torch.Tensor): Radius of the point cloud.

                Returns:
                torch.Tensor: Computed CD loss.
        """
        cost_for, cost_bac = chamfer_distance(target, pred)
        cost = .8 * cost_for + .2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred, target, pcd_radius):
        """
               Computes the repulsion loss for the predicted point cloud to encourage uniform distribution.

               Parameters:
               pred (torch.Tensor): Predicted point cloud.
               target (torch.Tensor): Ground truth point cloud (unused in this method).
               pcd_radius (torch.Tensor): Radius of the point cloud (unused in this method).

               Returns:
               torch.Tensor: Computed repulsion loss.
        """
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        idx = idx[:, :, 1:].to(torch.int32)
        idx = idx.contiguous()
        pred = pred.transpose(1, 2).contiguous()
        grouped_points = pointnet2_utils.grouping_operation(pred, idx)
        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).to(torch.device('cuda')))
        dist = torch.sqrt(dist2)
        weight = torch.exp(-dist2 / self.h ** 2)
        uniform_loss = torch.mean((self.radius - dist) * weight)
        return uniform_loss

    def forward(self, pred, target, pcd_radius):
        """
                Computes the joint loss, including the scaled EMD loss and weighted repulsion loss.

                Parameters:
                pred (torch.Tensor): Predicted point cloud.
                target (torch.Tensor): Ground truth point cloud.
                pcd_radius (torch.Tensor): Radius of the point cloud.

                Returns:
                tuple: A tuple containing the EMD loss scaled by 100 and the weighted repulsion loss.
        """
        return self.get_emd_loss(pred, target, pcd_radius) * 100, self.alpha * self.get_repulsion_loss(pred, target, pcd_radius)


def get_optimizer(model: nn.Module):
    """
        Create and return an AdamW optimizer for the given model.

        Parameters:
        model (torch.nn.Module): The model whose parameters will be optimized.

        Returns:
        torch.optim.AdamW: The AdamW optimizer initialized with the model's parameters,
                           learning rate of 1e-3, and weight decay of 1e-5.
        """
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train(epochs):
    # If a GPU is available we will use it, else the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the train dataset (samples with the specified indexes in train_list.txt file)
    train_dataset = PUNET_Datset(points=1024, split='train')
    print("Train Dataset size: ", len(train_dataset))
    # Create a dataloader with our training dataset having a batch size of 4 samples
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    # Initialize the model and move it to the available device (GPU or CPU)
    model = PUNet().to(device)
    # We use AdamW optimizer
    optimizer = get_optimizer(model)
    # We use a custom loss to determine reconstruction and repulsion loss
    loss_fn = CustomLoss(alpha=0.5).to(device)
    # Set the model to training mode
    model.train()
    # Load a point cloud from xyz file and consider only the xyz coordinates
    data = np.loadtxt('./model/uploads/cow.xyz')[:, :3]

    for epoch in range(epochs):
        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        try:
            model.train()
            for batch in track(train_loader):
                optimizer.zero_grad()
                input_data, gt_data, radius_data = batch

                input_data = input_data.float().to(device)
                gt_data = gt_data.float().to(device)
                gt_data = gt_data[..., :3].contiguous()
                radius_data = radius_data.float().to(device)

                predictions = model(input_data)
                loss1, rep_loss = loss_fn(predictions, gt_data, radius_data)
                loss = loss1 + rep_loss

                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                emd_loss_list.append(loss1.item())
                rep_loss_list.append(rep_loss.item())

        except Exception as e:
            print("Error occurred:", e)
            exit()

        output = model(torch.tensor([data, data], dtype=torch.float32).to(device))
        output_list = output[0].cpu().detach().numpy().tolist()

        print(f"Epoch {epoch} loss: {np.mean(loss_list)}, emd loss: {np.mean(emd_loss_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PU-Net model.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    args = parser.parse_args()
    train(args.epochs)