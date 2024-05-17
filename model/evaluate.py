from train import PUNET_Datset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from train import CustomLoss
from rich.progress import track

def evaluate_model(model, data_loader, loss_fn, device):
    """
        Evaluate the performance of a given model on a test dataset using EMD_loss, repulsion_loss adn joint loss

        Parameters:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the test dataset.
        loss_fn (function): Loss function that computes the EMD and repulsion losses.
        device (torch.device): Device to run the evaluation on

        Returns:
        tuple: A tuple containing the mean joint loss, mean EMD loss, and mean repulsion loss.
            - mean_joint_loss (float): The average joint loss over the test dataset.
            - mean_emd_loss (float): The average Earth Mover's Distance (EMD) loss over the test dataset.
            - mean_repulsion_loss (float): The average repulsion loss over the test dataset.
    """
    model.eval() # Set the model in inference mode
    test_joint_loss_list = []
    test_emd_loss_list = []
    test_repulsion_loss_list = []
    with torch.no_grad():
        for batch in track(data_loader): # Calculate the loss for every batch and save in a list
            input_data, gt_data, radius_data = batch

            input_data = input_data.float().to(device)
            gt_data = gt_data.float().to(device)
            gt_data = gt_data[..., :3].contiguous()
            radius_data = radius_data.float().to(device)
            # Making predictions on our model
            predictions = model(input_data)
            # Calculate EMD loss and repulsion loss
            emd_loss, rep_loss = loss_fn(predictions, gt_data, radius_data)
            # Calculate joint_loss
            joint_loss = emd_loss + rep_loss

            # Save the losses for each batch
            test_joint_loss_list.append(joint_loss.item())
            test_emd_loss_list.append(emd_loss.item())
            test_repulsion_loss_list.append(rep_loss.item())
    # Return the mean over each loss
    return np.mean(test_joint_loss_list), np.mean(test_emd_loss_list), np.mean(test_repulsion_loss_list)


if __name__ == '__main__':
    # If GPU is available we will use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load from our dataset only the samples specified by their index in test_list.txt file
    test_dataset = PUNET_Datset(points=1024, split='test')

    print("Test Dataset size: ", len(test_dataset))
    # Build a DataLoader using our dataset haveing batch size of 4 samples
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)
    # Use our custom loss function that determines reconstruction loss and also repulsion loss
    loss_fn = CustomLoss(alpha=0.5).to(device)
    # Use our saved model from the last training to evaluate its performance on test dataset
    with open('saved_models/E9_85.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    # Use evaluate_model function to determine mean loss over the test dataset
    test_joint_loss, test_emd_loss, test_repulsion_loss = evaluate_model(model, test_loader, loss_fn, device)

    print(f"Test loss: {test_joint_loss}, EMD loss: {test_emd_loss}, Repulsion loss: {test_repulsion_loss}")
