import torch
import torch.nn as nn

class PUNet(nn.Module):
    def __init__(self, num_point=1024, up_ratio=4, use_normal=False, use_bn=False, use_res=False):
        super(PUNet, self).__init__()

        self.num_point = num_point
        self.use_normal = use_normal
        self.use_bn = use_bn
        self.use_res = use_res
        self.num_points = [num_point//(2**i) for i in range(4)]
        self.mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512],
        ]
        self.radius = [.05, .1, .2, .3]
        self.nsamples = [32, 32, 32, 32]


if __name__ == '__main__':
    net = PUNet(num_point=1024, use_normal=False, use_bn=False, use_res=False)
    print(net.num_points)