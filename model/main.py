import pickle
import torch

import numpy as np

if __name__ == '__main__':
    with open('saved_models/E5_1.pkl', 'rb') as f:
        x = pickle.load(f)

    data = np.loadtxt('cow.xyz')[:,:3]
    output = x(torch.tensor([data,data,data,data], dtype=torch.float32, device='cuda'))
    output_list = output[0].cpu().detach().numpy().tolist()
    with open('cow6.xyz', 'w') as f:
        for i in output_list:
            for j in i:
                f.write(str(j)+' ')
            f.write('\n')
    print(output_list)