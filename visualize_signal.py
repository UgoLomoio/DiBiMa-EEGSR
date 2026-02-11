import torch
from torch import nn
import gc
from models import *
import os 
from metrics import *
import mne
from torch.cuda import empty_cache
import sys
from utils import set_seed, add_zero_channels
from train import * 

gc.collect()
empty_cache()
mne.set_log_level('ERROR') 

project_path = os.getcwd()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = True # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")

dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
n_patients = dict_n_patients["mmi"]  # Number of patients in the dataset   

imgs_path = os.path.join(project_path, 'imgs')
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

def plot_signal(signal_lr, signal_hr, dataset_name, fs, ch_idx=0, multiplier=8):

    if signal_lr.shape[-1] != signal_hr.shape[-1]:
        signal_lr = nn.functional.interpolate(signal_lr, scale_factor=multiplier, mode='linear', align_corners=False)
    elif signal_lr.shape[1] != signal_hr.shape[1]:
        signal_lr = add_zero_channels(signal_lr, signal_hr.shape[1], dataset_name=dataset_name, multiplier=multiplier)
    
    i = random.randint(0, signal_lr.shape[0]-1)
    if signal_lr.ndim == 3:
        signal_lr = signal_lr[i, :, :]  # Take the i-th sample in the batch
    if signal_hr.ndim == 3:
        signal_hr = signal_hr[i, :, :]  # Take the i-th sample in the batch


    signal_lr_ch = signal_lr[ch_idx, :]
    signal_hr_ch = signal_hr[ch_idx, :]
    print(f"Signal LR Channel Shape: {signal_lr_ch.shape}, Signal HR Channel Shape: {signal_hr_ch.shape}")

    time = np.arange(signal_hr_ch.shape[-1]) / fs  # Assuming a sampling rate of 160 Hz

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4))
    plt.plot(time, signal_lr_ch, label='Low Resolution')
    plt.plot(time, signal_hr_ch, label='High Resolution')
    plt.title(f"EEG Signal {dataset_name.upper()} - Channel {ch_idx}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(imgs_path, f'signal_{dataset_name}_ch{ch_idx}.png'))

def visualize(dataset_name, target_channels=64):

    gc.collect()
    empty_cache()

    multiplier = 8

    # Create train-test split
    patients = list(range(1, n_patients + 1))
    test_size = 0.2
    train_patients, val_patients, test_patients = train_test_val_split_patients(patients, test_size=test_size, random_state=seed)
    data_folder = data_path + os.sep + dataset_name

    dataloader_test = None  # Initialize dataloader_test
    #dataloader_train = None  # Initialize dataloader_train

    print("\n=== Training and Evaluating Models ===")
    for sr_type in sr_types:
        print(f"\n--- SR Type: {sr_type} ---")           
        # Prepare dataloaders
        if split_windows_first:
            dataset = EEGDataset(subject_ids=train_patients+val_patients+test_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)
            _, _, dataloader_test = prepare_dataloaders_windows(
                dataset_name, dataset, seconds=seconds, batch_size=batch_size, return_test=True
            )
            del dataset
        else:
            _, dataloader_test = prepare_dataloaders_windows(
                dataset_name, dataset, seconds=seconds, batch_size=batch_size, return_test=True
            )
        
        input_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f'x{multiplier}'])  # Input channels based on SR type
                
        dataloader_test.dataset.multiplier = multiplier  # Set multiplier attribute for later use
        dataloader_test.dataset.target_channels = target_channels  # Set target_channels attribute for later use
        dataloader_test.dataset.sr_type = sr_type  # Set sr_type attribute for later use
        dataloader_test.dataset.seconds = seconds  # Set seconds attribute for later use
        dataloader_test.dataset.dataset_name = dataset_name  # Set dataset_name attribute for later use
        dataloader_test.dataset.num_channels = input_channels  # Set num_channels attribute for later use

        #plot example signals
        print("\nPlotting Example Signals...")
        #outlier sample in the dataloader 
        signal_lr = next(iter(dataloader_test))[0]
        signal_hr = next(iter(dataloader_test))[1]
        fs = 160 if dataset_name == "mmi" else 200
        #channel with max or min point-value 
        ch_idx = torch.argmax(torch.abs(signal_hr)) % signal_hr.shape[1]
        print(f"Selected Channel Index for Visualization: {ch_idx}")
        print(f"Signal Shapes - LR: {signal_lr.shape}, HR: {signal_hr.shape}")
        plot_signal(signal_lr, signal_hr, dataset_name, fs=fs, ch_idx=ch_idx, multiplier=multiplier) 

def main():
     
    set_seed(seed)
    dataset_names = ["mmi", "seed"]
    for dataset_name in dataset_names:
        print(f"\n\n########## Running Ablation Study for Dataset: {dataset_name} ##########")
        if dataset_name == "mmi":
            target_channels = 64
        else:
            target_channels = 62
        visualize(dataset_name, target_channels=target_channels)

if __name__ == '__main__':

    sys.exit(main())
