import torch
import torch.nn as nn
from Pointnet2.pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
import Pointnet2.pointnet2.pytorch_utils as pt_utils


class PUNet(nn.Module):
    def __init__(self, num_point=1024, upscale_ratio=4, use_normal=False, use_batch_norm=False, use_res=False):
        super().__init__()

        self.npoint = num_point
        self.use_normal = use_normal
        self.up_ratio = upscale_ratio

        self.npoints = [num_point//(2**i) for i in range(4)]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32] * 4

        in_ch = 0 if not use_normal else 3
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k],
                    use_xyz=True,
                    use_res=use_res,
                    bn=use_batch_norm))
            in_ch = mlps[k][-1]

        self.FP_Modules = nn.ModuleList()
        for i in range(len(self.npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[i + 1][-1], 64],
                    bn=use_batch_norm))

        in_ch = len(self.npoints) * 64 + 3  # 4 layers + input xyz
        self.FC_Modules = nn.ModuleList()
        for _ in range(upscale_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_batch_norm))

        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_batch_norm),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False))

    def forward(self, points, npoint=None):
        if npoint is None:
            num_points = [None] * len(self.npoints)
        else:
            num_points = []
            for i in range(len(self.npoints)):
                num_points.append(npoint // 2 ** i)

        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() if self.use_normal else None

        l_xyz, l_feats = [xyz], [feats]
        for i in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[i](l_xyz[i], l_feats[i], npoint=num_points[i])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        up_feats = []
        for i in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[i](xyz, l_xyz[i + 2], None, l_feats[i + 2])
            up_feats.append(upk_feats)

        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1)

        r_feats = []
        for i in range(len(self.FC_Modules)):
            feat_i = self.FC_Modules[i](feats)
            r_feats.append(feat_i)
        r_feats = torch.cat(r_feats, dim=2)

        output = self.pcd_layer(r_feats)
        return output.squeeze(-1).transpose(1, 2).contiguous()


if __name__ == '__main__':
    model = PUNet(upscale_ratio=8, use_normal=True).cuda()
    points = torch.randn([7, 1024, 6]).float().cuda()
    while True:
        output = model(points)
    print(output.shape)