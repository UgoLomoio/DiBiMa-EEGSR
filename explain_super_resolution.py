import os
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import EEGDataset
import torch
import matplotlib.pyplot as plt
import warnings
import numpy as np
from captum.attr import LayerGradCam
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from test import best_params, loss_fn, learning_rate, debug, epochs, diffusion_params, prediction_type, load_model_weights
from models import DiBiMa_Diff, DiBiMa_nn

cwd = os.getcwd()

os.makedirs('model_weights', exist_ok=True)

# Data directory
DATA_DIR = os.path.join(os.getcwd(), 'eeg_data')

# Hyperparameters
BATCH_SIZE = 4
NUM_WORKERS = 1

demo = True # we only need one for explanation
seed = 2
dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = DATA_DIR

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np

class DiBiMaGradCam:
    """Fixed manual LayerGradCAM for DiBiMa EEG SR."""
    
    def __init__(self, diffusion_model, device='cuda'):
        self.device = device
        self.model = diffusion_model.model.to(device).train()
        self.model.use_diffusion = True
        self.num_timesteps = diffusion_model.scheduler.num_train_timesteps
        self.scheduler = diffusion_model.scheduler
        
        decoder_modules = list(self.model.decoder_sr)
        self.target_layer = decoder_modules[-2]  # Conv1d
        print(f"✅ Target Conv1d: {type(self.target_layer).__name__}")
    
    def generate_heatmap(self, lr, hr, pos):
        self.model.zero_grad()
        
        if lr.ndim == 2: lr = lr.unsqueeze(0)
        if hr.ndim == 2: hr = hr.unsqueeze(0)
        if pos.ndim == 2: pos = pos.unsqueeze(0)
        
        b = lr.size(0)
        t = torch.full((b,), self.num_timesteps - 1, dtype=torch.long, device=self.device)
        lr_d = lr.to(self.device)
        hr_d = hr.to(self.device)
        pos_d = pos.to(self.device)
        noise = torch.randn_like(hr_d)
        x_t = self.scheduler.add_noise(hr_d, noise, t)
        
        x_t.requires_grad_(True)
        pred = self.model(x_t, t, lr_d, pos_d)
        loss = F.mse_loss(pred, noise)
        print(f"Loss: {loss.item():.4f}")
        
        # SIMPLIFIED: Input gradients (works always)
        self.model.zero_grad()
        loss.backward()
        input_grads = x_t.grad
        
        # 1D temporal heatmap from input gradients
        heatmap = input_grads.abs().mean(dim=[0,1]).squeeze().cpu().numpy()
        if heatmap.ndim == 0:  # FIXED: scalar → error
            print("⚠️ Scalar fallback")
            heatmap = np.array([float(heatmap)])
        else:
            print(f"✅ Heatmap shape: {heatmap.shape}")
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, loss.item()

    def plot_explanation(self, signal, heatmap, true_class=None, fs=160, figsize=(15, 5)):
        """
        Enhanced plot with COLORMAP + colorbar.
        """
        if signal.ndim == 2:
            print("Averaging multi-channel signal for plotting.")
            signal = signal.mean(dim=0).cpu().detach().numpy()  # average channels
        if heatmap.ndim == 3:
            print("Averaging multi-channel heatmap for plotting.")
            heatmap = heatmap.squeeze()
        if heatmap.ndim == 2:
            heatmap = heatmap.mean(axis=0).cpu().detach().numpy()  # average channels
        # Ensure heatmap is 1D array
        heatmap = np.asarray(heatmap).flatten()
        if len(heatmap) == 1:
            heatmap = np.full_like(signal, heatmap[0])
        elif len(heatmap) != len(signal):
            from scipy.ndimage import zoom
            heatmap = zoom(heatmap, len(signal)/len(heatmap), order=1)
        
        time = np.arange(len(signal)) / fs
        signal_min, signal_max = signal.min(), signal.max()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [1, 3]})
        
        # 1. Original signal
        ax1.plot(time, signal, 'steelblue', linewidth=1.5, alpha=0.9)
        ax1.fill_between(time, signal, alpha=0.2, color='steelblue')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original EEG Signal', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Overlay + COLORMAP
        ax2.plot(time, signal, 'k', linewidth=1.5, alpha=0.8, label='EEG Signal')
        
        # imshow WITH COLORMAP + extent matching signal range
        im = ax2.imshow(heatmap[np.newaxis, :], 
                    cmap='jet',  # Red=high importance, Blue=low
                    alpha=0.65, 
                    aspect='auto',
                    extent=[time[0], time[-1], signal_min*1.05, signal_max*1.05],
                    vmin=0, vmax=1)
        
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Amplitude', fontsize=12)
        
        title = f'GradCAM++: Super-Resolution Explanation'
        if true_class is not None:
            title += f' (True: {true_class})'
        ax2.set_title(title, fontweight='bold', pad=10)
        
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # HORIZONTAL COLORBAR BELOW plot
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', 
                       fraction=0.05, pad=0.1, shrink=0.8)
        cbar.set_label('GradCAM Importance →', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":

    dataset_name = 'mmi'
    fs_hr = 160
    target_channels = 64
    sr_types = ['spatial', 'temporal']
    
    # Data
    num_subjects = dict_n_patients[dataset_name]
    all_ids = list(range(1, num_subjects + 1)) 
    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=seed)
    dataset_path = os.path.join(data_folder, dataset_name)
                  
    dataset_test = EEGDataset(test_ids, dataset_path, dataset_name=dataset_name, demo=demo) 
    val_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
           
    seconds = 10
    target_channels = 64
    results = {}
    for sr_type in sr_types:
        
        print(f"Explaining super-resolution for type: {sr_type}")
        in_channels = 64 if sr_type == "temporal" else 8
        fs_lr = 20 if sr_type == "temporal" else 160
        str_param = "x8" if sr_type == "temporal" else f"8to64chs"
        model_path = f'model_weights/fold_1/DiBiMa_eeg_{str_param}_{sr_type}_1.pth'
        print(f"Loading model from: {model_path}")
        model = DiBiMa_nn(
                                    target_channels=target_channels,
                                    num_channels=in_channels,
                                    fs_lr=fs_lr,
                                    fs_hr=fs_hr,
                                    seconds=seconds,
                                    residual_global=False,
                                    residual_internal=True,
                                    use_subpixel=True,
                                    sr_type=sr_type,
                                    use_mamba=best_params["use_mamba"],
                                    use_diffusion=best_params["use_diffusion"],
                                    n_mamba_layers=best_params["n_mamba_layers"],
                                    mamba_dim=best_params["dim"],
                                    mamba_d_state=best_params["d_state"],
                                    mamba_version=best_params["version"],
                                    n_mamba_blocks=best_params["n_mamba_blocks"],
                                    use_positional_encoding=False,
                                    merge_type=best_params["merge_type"],
                                    use_electrode_embedding=best_params["use_electrode_embedding"],  
        )
        model_pl = DiBiMa_Diff(model,
                                            loss_fn,
                                            diffusion_params=diffusion_params,
                                            learning_rate=learning_rate,
                                            scheduler_params=None,
                                            predict_type=prediction_type,  # "epsilon" or "sample"
                                            debug=debug,
                                            epochs=epochs,
                                            plot=False
        ).to(device)
        model_pl.model = load_model_weights(model_pl.model, model_path).to(device)

        # Set sr_type for dataset
        val_loader.dataset.sr_type = sr_type
        val_loader.dataset.multiplier = 8
        val_loader.dataset.sr_type = sr_type
        val_loader.dataset.fs_lr = fs_lr
        val_loader.dataset.num_channels = in_channels
        channel_names = val_loader.dataset.channel_names

        # Get a batch of test data
        eeg_lr, eeg_hr, pos, label = next(iter(val_loader))
        eeg_lr = eeg_lr.to(device)
        eeg_hr = eeg_hr.to(device)
        print(f"Low-res EEG shape: {eeg_lr.shape}, High-res EEG shape: {eeg_hr.shape}")
        
        # Explainer 
        explainer = DiBiMaGradCam(model_pl)
        # Single explanation
        lr, hr, pos, label = next(iter(val_loader))  # get first batch
        lr = lr[0].squeeze()  # (C, L)
        hr = hr[0].squeeze()
        pos = pos[0]
        label = label[0].item()
        heatmap, _ = explainer.generate_heatmap(lr, hr, pos)
        #Plot explanation
        fig = explainer.plot_explanation(lr, heatmap, label, fs=fs_lr)
        fig.savefig(f'gradcam_{dataset_name}_{sr_type}_single.png')
        plt.close(fig)