import sys

sys.path.insert(1, 'siamese_fcn')
sys.path.insert(1, 'pytorch_datasets')
sys.path.insert(1, 'evaluation')
sys.path.insert(1, 'results')
sys.path.insert(1, 'visualization')
sys.path.insert(1, 'preprocessing')



import os
import torch
from matplotlib import pyplot as plt
import numpy as np
from data_augmentation import RandomFlip, RandomRot
from losses.focal_loss import FocalLoss
import torchvision.transforms as tr
from torch import nn
from torch.utils.data import DataLoader
from cscd_dataset_loader import CSCD_Dataset
from data_examination import examine_subset
from levir_dataset_loader import LEVIR_Dataset
from metrics import evaluate_net_predictions, get_ground_truth, get_predictions
from plots import create_loss_accuracy_figures
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from tables import create_tables, load_metrics, store_mean_difference_per_epoch
from train_test import train
from unet import Unet 
from late_siam_net import SiamLateFusion
import argparse
from aggregate_training_results import plot_loss
from statistical_tests import aggregate_distribution_histograms, perform_statistical_tests





def get_args():
    '''
    All of the arguments to run a single pipeline of the experiment. Here is given a short description of each parameter.
    
    experiment_name -- How your experiment is called. Also the name of the directory and model weights which will be saved. 
    epochs -- The number of epochs the neural network is trained for. 
    fp_modifier -- False positive modifier. Higher value means higher punishment for False Positives. Used when the loss function is negative log
                   likelihood (nll).
    batch_size -- The batch size of the tensors run throughout the neural network/s during training. 
    dir -- The directory from which the dataset is sampled from. 
    dataset_name -- The name of the dataset for the if-statement choosing the PyTorch dataset. The implemented classes are 'LEVIR-CD' and 'CSCD'.
    loss -- The loss function optimized during training. Current possibilities include 'focal_loss' and 'nll'. The focal loss is implemented in its corresponding 
            directory (see folder structure or code below), while negative log likelihood is using its traditional PyTorch implementation. 
    restore_prev -- Boolean parameter. In case a given pipeline has already been performed, it restores the weights of the models, such that only evaluation
                    and plot generation are performed. 
    generate_plots -- Boolean parameter. Specifies if all plots should be regenerated. 
    data_augmentation -- Boolean parameter. Specifies if data augmentations should be performed during training (Rotations, Translations, etc.)
    patch_side -- Specifies the side M of an M x M patch. To increase the amount of data we train our models on, we cut out input images into patches. 
                  Discussed in the paper on what patch sides were experimented with.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type = str, default = "Default")
    parser.add_argument("--epochs", type = int, default=1)
    parser.add_argument("--fp_modifier", type = int, default=10)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--dir", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--dataset_name", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--loss", type=str, default="nll")
    parser.add_argument("--restore_prev", type = bool, default = False)
    parser.add_argument("--generate_plots", type = bool, default = False)
    parser.add_argument("--data_augmentation", type = bool, default = False)
    parser.add_argument("--patch_side", type = int, default = 96)

    return parser.parse_args()

FUSIONS = ["Early", "Middle-Conc", "Middle-Diff", "Late"] # Constant hardcoded names of the four fusion architectures tested 


def run_experiment(experiment_name, dataset_name, datasets, dataset_loaders, criterion, epochs, restore_prev=False, generate_plots=False):
    '''
    A full experiment, running all fusion architectures on a dataset, specified in location and preprocessing via the given arguments to the initial script. 
    
    The four current fusion architectures are: 
        - Early - Concatenates prior to processing via U-Net. 
        - Middle-Conc. - Concatenates within the U-Net network structure in a Siamese configuration, and via skip connections via two encoders.
        - Middle-Diff. - Very similar to Middle-Conc., but instead of concatenating in the encoder, it takes the image differences. 
        - Late - Processes both images separetely via U-Net in a Siamese configuration, and concatenates them after.
        
        The first three have been proposed in the following paper (INSERT LINK) and have inspired a large part of this research. The final fusion architecture
        has been first proposed in (INSERT LINK) by the author and had further experiments done upon it as well.
    
    If specified, all four models get trained for the specified number of epochs (with early stopping). The model is then, depending on the arguments
    either reloaded (along with all of its recorded training metrics), or trained for the specified number of epochs. 
    
    Training, validation and evaluation data are then subsequently recorded in tables and output as tables. 
    
    If the generate_plots parameter is on, then accordingly the data (either from the training metrics or newly evaluated one) is used to generate plots.
    These plots consist of histograms, cdfs and KDEs (kernel density estimates) of distributions. All of them are also accordingly aggregated for all fusion 
    architectures and put pairwise against each other when needed (for example cdfs when we compare first and second order stochastic dominance). The examine_subset() function
    shows a sample visualization of the current model. 
    
    All of these plots are stored within an 'experiment_results' directory under the name of the experiment name directory. 
    
    '''
    net, net_name = None, None 
    EXPERIMENT_PATH = os.path.join('experiment_results', experiment_name)
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    train_set, val_set, test_set = datasets
    
    train_set_loader, val_set_loader, test_set_loader = dataset_loaders
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    aggregate_categorical = []
    
    n = 4
    arr = plt.cm.viridis(np.linspace(0, 1, n))
    COLORS = {"Early":  arr[0], "Middle-Conc": arr[1], "Middle-Diff": arr[2], "Late": arr[3]}
    
    predictions_dict = {}
    
    for fusion in FUSIONS:
        print('Training ', fusion, ':')
        if fusion == "Early":
            net, net_name = Unet(6, 2), f'Early-{experiment_name}'
        elif fusion == "Middle-Conc": 
            net, net_name = SiamUnet_conc(3, 2), f'Middle-Conc-{experiment_name}'
        elif fusion == "Middle-Diff":
            net, net_name = SiamUnet_diff(3, 2), f'Middle-Diff-{experiment_name}'
        elif fusion == "Late": 
            net, net_name = SiamLateFusion(3, 2), f'Late-{experiment_name}'

        net.to(DEVICE)

        model_path = os.path.join(EXPERIMENT_PATH, net_name)

        if os.path.exists(f'{model_path}.pth') and restore_prev == True:
            print('Restored weights!')
            state_dict = torch.load(f'{model_path}.pth', weights_only=True)
            net.load_state_dict(state_dict)
            
            training_metrics, validation_metrics, test_metrics = load_metrics(model_path)
            

        else: 
            os.makedirs(model_path, exist_ok=True)
            training_metrics, validation_metrics = train(net, train_set, train_set_loader, val_set, val_set_loader, criterion, DEVICE, n_epochs= epochs, save=True, save_dir = f'{model_path}.pth', skip_val = False, early_stopping = True)
            

        
        test_metrics = evaluate_net_predictions(net, criterion, test_set_loader)
        create_tables(training_metrics, validation_metrics, test_metrics, os.path.join(model_path, 'tables'))
        if generate_plots:
            create_loss_accuracy_figures(training_metrics, validation_metrics, test_metrics, net_name, os.path.join(model_path, 'figures'))
            visualization_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
            examine_subset(net, net_name, 10, DEVICE,  visualization_loader, save_path=os.path.join(model_path, 'figures'))
        
        predictions = get_predictions(net, test_set)
        predictions_dict[fusion] = predictions
          
        
    if generate_plots:
        plot_loss(experiment_name, FUSIONS, COLORS)
        store_mean_difference_per_epoch(aggregate_categorical, EXPERIMENT_PATH)
        perform_statistical_tests(dataset_name, predictions_dict, EXPERIMENT_PATH, p_val=0.05)
        ground_truth = get_ground_truth(test_set)
        aggregate_distribution_histograms(dataset_name, ground_truth, predictions_dict, COLORS, EXPERIMENT_PATH)
    
    
def get_dataset(dataset_name, dirname, mode, FP_MODIFIER, PATCH_SIDE=96, transform=None):
    '''Dataset getter function to give out a specified PyTorch dataset. Made like this for 
    extension purposes.
    
    There exist some depricated datasets part of this getter function, appropriately
    found in the 'deprecated' directory of the project. 
    '''
    if dataset_name == 'LEVIR':
        return LEVIR_Dataset(dirname, mode, FP_MODIFIER, patch_side=PATCH_SIDE, transform=transform)
    elif dataset_name == 'CSCD':
        return CSCD_Dataset(dirname, mode, FP_MODIFIER, patch_side=PATCH_SIDE, transform=transform)
    else:
        raise ValueError("Unknown dataset name")
    
def get_loss(name, weights):
    '''Dataset getter function to give out a specified PyTorch loss. Made like this for 
    extension purposes.
    
    As this is a pixel-level classification task
    '''
    criterion = None 
    
    if name == "nll":
        criterion = nn.NLLLoss(weight=weights) # Negative log likelihood designated to punish false positives more.
    elif name == "focal_loss":
        criterion = FocalLoss(gamma=2, alpha=0.25) # Flexible parameters, based on statements from the original Focal Loss paper.
    else: 
        criterion = nn.NLLLoss(weight=weights) # Default. Left like this for extension purposes. 
        
    return criterion
    
if __name__ == "__main__":
    ''''
    Argument preprocessing to run the experiment from the argument parser.
    '''
    ARGS = get_args()
    EXPERIMENT_NAME = ARGS.experiment_name
    DATASET_NAME = ARGS.dataset_name
    DIRECTORY = ARGS.dir
    BATCH_SIZE = ARGS.batch_size 
    FP_MODIFIER = ARGS.fp_modifier
    RESTORE_PREV = ARGS.restore_prev
    N_EPOCHS = ARGS.epochs
    GENERATE_PLOTS = ARGS.generate_plots
    LOSS = ARGS.loss
    DATA_AUGMENTATION = ARGS.data_augmentation
    PATCH_SIDE = ARGS.patch_side
    
    torch.manual_seed(42) #fixed seed for reproducibility
    
    if DATA_AUGMENTATION:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None
    
    train_dataset = get_dataset(DATASET_NAME, DIRECTORY, "train", FP_MODIFIER, transform=data_transform, PATCH_SIDE = PATCH_SIDE)
    val_dataset = get_dataset(DATASET_NAME, DIRECTORY, "val", FP_MODIFIER, transform=data_transform, PATCH_SIDE = PATCH_SIDE)
    test_dataset = get_dataset(DATASET_NAME, DIRECTORY, "test", FP_MODIFIER, transform=data_transform, PATCH_SIDE = PATCH_SIDE)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.FloatTensor(train_dataset.weights).to(device)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)
    
    criterion = get_loss(LOSS, weights)

    run_experiment(EXPERIMENT_NAME, DATASET_NAME, [train_dataset, val_dataset, test_dataset], 
                   [train_loader, val_loader, test_loader], criterion, N_EPOCHS,
                   restore_prev=RESTORE_PREV, generate_plots=GENERATE_PLOTS)

    




     




