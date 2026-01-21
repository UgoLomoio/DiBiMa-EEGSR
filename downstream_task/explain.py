import os
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils_class import EEGDatasetClassification, load_model_weights
from downstream_task.resnet50 import ResNet50, ResNetPL
import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerGradCam
import warnings
warnings.filterwarnings('ignore')


current_dir = os.getcwd()
parent_dir = os.path.join(current_dir, os.pardir)  # or os.path.dirname(current_dir)
parent_abs = os.path.abspath(parent_dir)
print(f"Adding to sys.path: {parent_abs}")
sys.path.insert(0, parent_abs)
from test import best_params, loss_fn, learning_rate, debug, epochs, diffusion_params, prediction_type
from models import DiBiMa_Diff, DiBiMa_nn

cwd = os.getcwd()

os.makedirs('model_weights', exist_ok=True)

# Data directory
DATA_DIR = os.path.join(parent_dir, 'eeg_data')

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 1    # Adjust based on your system
MULTIPLIER = 8
LR = 1e-4
MAX_EPOCHS = 50

demo = True # we only need one for explanation
seed = 2
dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = DATA_DIR

if demo:
    MAX_EPOCHS = 2
    BATCH_SIZE = 4

class ResNet1DGradCam:
    """
    Robust GradCAM for 1D ResNet EEG using Captum (no string layer issues).
    """
    def __init__(self, model, device='cuda'):
        
        self.device = device
        self.model = model.to(device).eval()
        
        # Target: LAST Bottleneck conv3 (layer 17: before final pool)
        modules = list(self.model.features)
        #for i, m in enumerate(modules):
        #    print(f"{i}: {m}")
        self.target_layer = modules[17].layer[6]  # Bottleneck(2048,512,2048,False)[-1].conv3
        print(f"✅ Target: {self.target_layer}")
        
        self.layer_gc = LayerGradCam(self.model, self.target_layer)

    def generate_heatmap(self, inputs, target_class=None):
        """LayerGradCAM."""
        self.model.zero_grad()
        
        inputs = inputs.clone().requires_grad_(True).to(self.device)
        
        # Forward WITHOUT no_grad (Captum needs it)
        logits = self.model(inputs)  # full tuple!
        pred_class = logits.argmax(dim=1).item()
        
        if target_class is None:
            target_class = pred_class
        
        print(f"Explaining class {target_class}, logits: {logits[0,target_class]:.3f}")
        
        # GradCAM - NO with torch.no_grad() or enable_grad()
        attributions = self.layer_gc.attribute(
            inputs, 
            target=target_class
        )
        
        print(f"Attributions shape: {attributions.shape}, norm: {attributions.abs().mean():.2e}")
        
        # Proper 1D reshape: (1,C,L) → (L,)
        if attributions.dim() == 3:  # (B=1,C,L)
            heatmap = attributions[0].mean(0).relu().cpu().detach().numpy()
        else:
            heatmap = attributions.mean(-1).relu().cpu().detach().numpy()
        
        # Normalize
        heatmap = heatmap.flatten()
        if heatmap.max() > 1e-6:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            print("⚠️ Zero heatmap - model not discriminative")
            heatmap = np.zeros_like(heatmap)
        
        return heatmap, pred_class

    def plot_explanation(self, signal, heatmap, pred_class, true_class=None, fs=160, figsize=(15, 5)):
        """
        Enhanced plot with COLORMAP + colorbar.
        """
        if signal.ndim == 2:
            signal = signal.mean(dim=0).cpu().detach().numpy()  # average channels
        if heatmap.ndim == 3:
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
        
        title = f'GradCAM++: Predicted Class {pred_class}'
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

def main():

    for dataset_name in ["mmi"]:#not using 'seed' in classification
        target_channels = 64 if dataset_name == "mmi" else 62
        fs_hr = 160 if dataset_name == "mmi" else 200
       
        # Data
        num_subjects = dict_n_patients[dataset_name]
        all_ids = list(range(1, num_subjects + 1)) 
        train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=seed)
                
        model_sr = None

        dataset_path = os.path.join(data_folder, dataset_name)
                  
        dataset_test = EEGDatasetClassification(test_ids, dataset_path, dataset_name=dataset_name, model_sr=model_sr, demo=demo) 
        val_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
               
        seconds = 10
        results = {}

        for sr_type in ["temporal", "spatial"]:
            results[sr_type] = {}
            for input_type in ["hr", "lr", "sr"]:
                results[sr_type][input_type] = {}
                if input_type == "lr":
                    in_channels = target_channels if sr_type == "temporal" else int(target_channels//MULTIPLIER)
                    fs_lr = int(fs_hr//MULTIPLIER) if sr_type == "temporal" else fs_hr
                elif input_type == "hr":
                    in_channels = target_channels
                else:
                    #Model
                    in_channels = target_channels if sr_type == "temporal" else int(target_channels//MULTIPLIER)
                    fs_lr = int(fs_hr//MULTIPLIER) if sr_type == "temporal" else fs_hr
                    str_param = "x8" if sr_type == "temporal" else f"{in_channels}to{target_channels}chs"
                    model_path = f'{parent_abs}/model_weights/fold_1/DiBiMa_eeg_{str_param}_{sr_type}_1.pth'
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
                    model_sr = load_model_weights(model_pl, model_path).to(device)
                    val_loader.dataset.model_sr = model_sr

                val_loader.dataset.input_type = input_type
                val_loader.dataset.sr_type = sr_type
                val_loader.dataset.num_channels = in_channels
 
                NUM_CLASSES = dataset_test.labels.unique().shape[0] #+ 1
                

                model = ResNet50(in_channels=in_channels, classes=NUM_CLASSES).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                weights = torch.load(os.path.join(parent_abs, 'downstream_task', 'model_weights', f'class_weights_{dataset_name}.pt'), weights_only=False).to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=weights)
                lit_model = ResNetPL(model, optimizer, criterion)

                # Load best model
                lit_model.load_state_dict(torch.load(os.path.join('model_weights', f'best_model_{dataset_name}_{input_type}_{sr_type}.pth')))
                lit_model.eval() 

                # Explainer 
                explainer = ResNet1DGradCam(lit_model.model)
                # Single explanation
                signal, label = next(iter(val_loader))  # get first batch
                signal = signal[0].squeeze()  # (C, L)
                label = label[0].item()
                heatmap, pred = explainer.generate_heatmap(signal.unsqueeze(0), target_class=label)
                if sr_type == "spatial":
                    fs = fs_hr
                else:
                    fs = fs_lr if input_type != "hr" else fs_hr
                #Plot explanation
                fig = explainer.plot_explanation(signal, heatmap, pred, label, fs=fs)
                fig.savefig(f'gradcam_{dataset_name}_{input_type}_{sr_type}_single.png')
                plt.show()
                #plt.close(fig)
                break  # only one for explanation

if __name__ == "__main__":
    main()
    
