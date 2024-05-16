from train import PUNET_Datset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from train import CustomLoss
from rich.progress import track

def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    test_joint_loss_list = []
    test_emd_loss_list = []
    test_repulsion_loss_list = []
    with torch.no_grad():
        for batch in track(data_loader):
            input_data, gt_data, radius_data = batch

            input_data = input_data.float().to(device)
            gt_data = gt_data.float().to(device)
            gt_data = gt_data[..., :3].contiguous()
            radius_data = radius_data.float().to(device)

            predictions = model(input_data)
            emd_loss, rep_loss = loss_fn(predictions, gt_data, radius_data)
            joint_loss = emd_loss + rep_loss

            test_joint_loss_list.append(joint_loss.item())
            test_emd_loss_list.append(emd_loss.item())
            test_repulsion_loss_list.append(rep_loss.item())

    return np.mean(test_joint_loss_list), np.mean(test_emd_loss_list), np.mean(test_repulsion_loss_list)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = PUNET_Datset(points=1024, split='test')

    print("Test Dataset size: ", len(test_dataset))

    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    loss_fn = CustomLoss(alpha=0.5).to(device)

    with open('saved_models/E9_85.pkl', 'rb') as f:
        model = pickle.load(f)

    test_joint_loss, test_emd_loss, test_repulsion_loss = evaluate_model(model, test_loader, loss_fn, device)

    print(f"Test loss: {test_joint_loss}, EMD loss: {test_emd_loss}, Repulsion loss: {test_repulsion_loss}")
