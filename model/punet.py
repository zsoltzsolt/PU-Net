import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
import pointnet2.pytorch_utils as pt_utils

class PUNet(nn.Module):
    def __init__(self, num_point=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
        super(PUNet, self).__init__()

        self.num_point = num_point
        self.use_normal = use_normal
        self.use_bn = use_bn
        self.use_res = use_res
        self.up_ratio = up_ratio
        self.num_points = [num_point//(2**i) for i in range(4)]
        self.mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512],
        ]
        self.radius = [.05, .1, .2, .3]
        self.nsamples = [32, 32, 32, 32]

        #   Hierarchical feature learning
        self.SA_modules = nn.ModuleList()
        for i in range(len(self.num_points)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.num_points[i],
                    radius=self.radius[i],
                    nsample=self.nsamples[i],
                    mlp=self.mlps[i],
                    use_xyz=True,
                    use_res=self.use_res,
                    bn=self.use_bn,
                ))

        # Multi-level feature aggregation
        self.FP_modules = nn.ModuleList()
        for i in range(len(self.num_points)-1):
            self.FP_modules.append(
                PointnetFPModule(
                    mlp=[self.mlps[i+1][-1], 64],
                    bn=self.use_bn,
                )
            )

        # Feature expansion
        in_ch = len(self.num_points)*64+3
        self.FC_modules = nn.ModuleList()
        for i in range(up_ratio):
            self.FC_modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=self.use_bn,
                )
            )

        # Coordinate Reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=self.use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)
        )

    def forward(self, points, npoint=None):
        return None



if __name__ == '__main__':
    net = PUNet(num_point=1024, use_normal=False, use_bn=False, use_res=False)
    print(net.num_points)