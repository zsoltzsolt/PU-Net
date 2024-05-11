from open3d import *
import pickle
import numpy as np
import torch
def main():
    with open('saved_models/E9_85.pkl', 'rb') as f:
        x = pickle.load(f)

    data = np.loadtxt('cow.xyz')[:, :3]
    output = x(torch.tensor([data,data,data,data], dtype=torch.float32, device='cuda'))
    output_list = output[0].cpu().detach().numpy().tolist()
    with open('cow_out.xyz', 'w') as f:
        for i in output_list:
            for j in i:
                f.write(str(j)+' ')
            f.write('\n')
    cloud = io.read_point_cloud("cow_out.xyz")
    visualization.draw_geometries([cloud])

if __name__ == "__main__":
    main()