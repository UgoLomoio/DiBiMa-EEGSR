
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
import mne
from mne.datasets import eegbci
from scipy import signal
import shutil
import matplotlib.pyplot as plt
from umap import UMAP
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

unmask_channels = {
    8: [0, 7, 14, 21, 28, 35, 42, 49],  # Core 10-20: Fp1,Fp2,Fz,F3,F4,C3,C4,Pz

    16: [0, 2, 4, 6, 14, 16, 18, 20, 22, 25, 27, 42, 43, 56, 58, 61],  # Your 16ch (frontal/central/parietal)
    
    32: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
         32, 34, 36, 38, 40, 42, 43, 44, 46, 48, 50, 52, 54, 56, 58, 61]  # Even 10-10 coverage
}

map_runs = {
    1: 'eyes_open',
    2: 'eyes_closed',
    3: 'task1',
    4: 'task2',
    5: 'task3',
    6: 'task4',
    7: 'task1',
    8: 'task2',
    9: 'task3',
    10: 'task4',
    11: 'task1',   
    12: 'task2',
    13: 'task3',
    14: 'task4'
}
map_tasks = {
    'task1': 'open-close right or left fist',
    'task3': 'open-close both fists or both feet',
    'task2': 'imagine right or left fist',
    'task4': 'imagine both fists or both feet'
}

map_annotations = {
    "T0": "rest",
    "T1": "right or left fist",
    "T2": "both fists or both feet"
}

def get_lr_data(eeg_64, num_channels=8):
    
    """Extract low-res from 64ch EEG using unmask_channels"""
    indices = unmask_channels[num_channels]
    return eeg_64[indices, :]

# -------------------------------
# EEGDataset for Super-Resolution EEG
# -------------------------------
class EEGDataset(Dataset):
    """
    Dataset EEG preprocessato per Super-Resolution EEG (EEG BCI dataset).
    Genera segmenti LR (160/sr_factor Hz) e HR (160 Hz) sincronizzati.
    """
    def __init__(self, subject_ids, runs, project_path, add_noise=True, fs_hr=160, fs_lr=16, seconds=10, verbose=False, demo=False, num_channels=64):
        self.data_lr = []
        self.data_hr = []
        self.add_noise = add_noise
        self.project_path = project_path
        self.fs_hr = fs_hr
        self.fs_lr = fs_lr
        self.sr_factor = int(fs_hr // fs_lr)
        self.verbose = verbose
        self.demo = demo
        self.seconds = seconds
        self.lr_window_length = fs_lr * seconds
        self.hr_window_length = fs_hr * seconds
        self.num_channels = num_channels    
        self.raw_data = []
        self.labels = []
        self.scaler_lr = MinMaxScaler(feature_range=(0, 1))
        self.scaler_hr = MinMaxScaler(feature_range=(0, 1))

        for i, subject in enumerate(subject_ids):
            print(f"Processing subject {subject} ({i+1}/{len(subject_ids)})", end='\r')
            if self.demo and i >= 2:
                print("\nDemo mode: stopping after 2 subjects.")
                break
            for run in runs:
                #print(f"  Run {run}", end='\r')
                # Local path for EDF file
                local_path = os.path.join(project_path, f'S{subject:03d}R{run:02d}.edf')

                # Download the file if it doesn't exist
                if not os.path.exists(local_path):
                    print(f"Downloading S{subject:03d}R{run:02d}.edf...")
                    try:
                        eegbci.load_data(subject, [run], path=os.path.dirname(project_path), update_path=True)
                        downloaded = os.path.join(
                            os.path.dirname(project_path),
                            'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0',
                            f'S{subject:03d}', f'S{subject:03d}R{run:02d}.edf'
                        )
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        shutil.move(downloaded, local_path)
                        print(f"Saved to: {local_path}")
                    except Exception as e:
                        print(f"Download error: {e}")
                        continue

                # Load and preprocess the raw data
                try:
                    raw = mne.io.read_raw_edf(local_path, preload=True, verbose=self.verbose)
                    self.raw_data.append(raw)
                    raw = self._preprocess_raw(raw)
                    data = raw.get_data()
                    sfreq = raw.info['sfreq']

                    # ================================
                    # 🔹 HR (160 Hz)
                    # ================================
                    if self.fs_hr == self.fs_lr:
                        data_hr = data
                    else:
                        sfreq_hr = self.fs_hr  # 160 Hz
                        num_samples_hr = int(data.shape[1] * (sfreq_hr / sfreq))
                        data_hr = signal.resample(data, num_samples_hr, axis=1)

                    # ================================
                    # 🔹 LR (16 Hz)
                    # ================================
                    if self.fs_lr == self.fs_hr:
                        data_lr = data_hr
                    else:
                        sfreq_lr = self.fs_lr  # 16 Hz
                        num_samples_lr = int(data_hr.shape[1] * (sfreq_lr / sfreq_hr))
                        data_lr = signal.resample(data_hr, num_samples_lr, axis=1)
                        
                    # ================================
                    # 🔹 Segmentazione
                    # ================================
                    for start in range(0, data_lr.shape[1] - self.lr_window_length, self.lr_window_length):
                        lr_seg = data_lr[:, start:start + self.lr_window_length]

                        hr_seg = data_hr[:, (start*self.sr_factor):(start * self.sr_factor + self.hr_window_length)]

                        if self.add_noise:
                            print("Adding noise to LR segment")
                            lr_seg += np.random.normal(0, 0.2, lr_seg.shape)

                        self.data_lr.append(lr_seg.astype(np.float32))
                        self.data_hr.append(hr_seg.astype(np.float32))
                        self.labels.append(run)  # Example label: run number
                                        
                except Exception as e:
                    print(f"⚠️ Exception during processing {local_path}: {e}")
                    continue

        self.data_lr = np.array(self.data_lr, dtype=np.float32)
        self.data_hr = np.array(self.data_hr, dtype=np.float32)
        
        self.data_lr = self.data_lr*1e6  # scale to microvolts 
        self.data_hr = self.data_hr*1e6  # scale to microvolts
        
        self.data_lr = self._zscore_normalization(torch.tensor(self.data_lr)).numpy()
        self.data_hr = self._zscore_normalization(torch.tensor(self.data_hr)).numpy()
        
        print(self.data_lr.min(), self.data_lr.max())
        print(self.data_hr.min(), self.data_hr.max())

        #self.data_lr = self._normalize_data(self.data_lr.reshape(-1, self.data_lr.shape[2]), self.scaler_lr).reshape(self.data_lr.shape)
        #self.data_hr = self._normalize_data(self.data_hr.reshape(-1, self.data_hr.shape[2]), self.scaler_hr).reshape(self.data_hr.shape)
        print(f"✅ Number of segments created: {len(self.data_lr)}")

    # -------------------------------
    # Preprocessing MNE
    # -------------------------------
    def _preprocess_raw(self, raw):
        raw.pick_types(eeg=True, eog=False, stim=False)
        raw.notch_filter(50.0, fir_design='firwin', verbose=self.verbose)
        raw.filter(0.5, 40.0, fir_design='firwin', verbose=self.verbose)
        raw.set_eeg_reference('average')
        return raw

    # -------------------------------
    # Z-score normalization
    # -------------------------------
    def _zscore_normalization(self, data):

        # Z-score per channel (common for EEG DL)
        mean = data.mean(dim=-1, keepdim=True)  # [B,C,1]
        std = data.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-6)  # Avoid div0
        normalized = (data - mean) / std  # [-3,3] typical
        return normalized
    
    def _normalize_data(self, data, scaler):
        data_reshaped = data.T  # Shape: (num_samples, num_channels)
        data_normalized = scaler.fit_transform(data_reshaped)
        return data_normalized.T  # Shape: (num_channels, num_samples)

    # -------------------------------
    # Dataset interface
    # -------------------------------
    def __len__(self):
        return len(self.data_lr)

    def __getitem__(self, idx):
        
        lr_data_64 = self.data_lr[idx]  
        hr_data = self.data_hr[idx]
        if self.num_channels != 64:
            lr_data = get_lr_data(lr_data_64, num_channels=self.num_channels)
        else:   
            lr_data = lr_data_64
        return lr_data, hr_data

def add_zero_channels(input_tensor, target_channels=64):
    """Add zero channels to input_tensor to match target_channels."""

    if input_tensor.ndim == 2:
        batch_size = None
        nchs = input_tensor.size(0)
        lenght = input_tensor.size(1)
    else:
        batch_size = input_tensor.size(0)
        nchs = input_tensor.size(1)
        lenght = input_tensor.size(2)

    if batch_size is None:
        input_target = torch.zeros((target_channels, lenght), device=input_tensor.device)
    else:
        input_target = torch.zeros((batch_size, target_channels, lenght), device=input_tensor.device)

    channels_to_use = unmask_channels[nchs]
    for i, ch in enumerate(channels_to_use):
        if batch_size is None:
            input_target[ch, :] = input_tensor[i, :]
        else:
            input_target[:, ch, :] = input_tensor[:, i, :]
    return input_target

def plot_umap_latent_space(model, eeg, labels, save_path=None, map_labels=None):

    # Get the latent space representations
    latent_vectors = []
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        eeg = torch.tensor(eeg, dtype=torch.float32).to(device)
        latent_vector = model(eeg, return_latent=True)[1]
        latent_vector = latent_vector.view(latent_vector.size(0), -1)
        print(latent_vector.shape)
        latent_vectors.append(latent_vector.cpu().numpy())
    latent_vectors = np.vstack(latent_vectors)

    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed).fit(latent_vectors)

    plt.figure(figsize=(10, 8))
    u_labels = np.unique(labels)
    for ul in u_labels:
        latent = [latent for i, latent in enumerate(latent_vectors) if labels[i] == ul]
        latent = np.array(latent)
        embedding = reducer.transform(latent)
        if map_labels:
            l = map_labels[ul]
        else:
            l = ul
        plt.scatter(embedding[:, 0], embedding[:, 1], label=l, alpha=0.5)
    
    plt.title('UMAP Projection of Latent Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def explain_super_resolution(model, eeg_lr, eeg_hr, channel_names, save_path=None):
    """
    Use saliency maps to explain which parts of the low-resolution EEG input
    are most important for the super-resolution task.
    """
    model.eval()
    eeg_lr.requires_grad_()

    # Forward pass
    output = model(eeg_lr)

    # Compute loss (MSE between output and high-res target)
    criterion = torch.nn.MSELoss()
    loss = criterion(output, eeg_hr)
    
    # Backward pass to compute gradients
    loss.backward()

    # Get saliency map (absolute value of gradients)
    saliency = eeg_lr.grad.data.abs().squeeze().cpu().numpy()

    # Plot saliency maps for each channel
    num_channels = eeg_lr.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels))
    for i in range(num_channels):
        axes[i].plot(saliency[i], color='red')
        axes[i].set_title(f'Saliency Map - Channel: {channel_names[i]}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Saliency')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def tensor2raw(eeg_tensor, info):
    """
    Convert a PyTorch tensor back to an MNE Raw object.
    """
    eeg_data = eeg_tensor.cpu().numpy()
    info = mne.create_info(ch_names=info['ch_names'], sfreq=info['sfreq'], ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    return raw

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """
    Combined loss:
      recon_loss = mse_loss
                   + lambda_smooth * smoothness_loss
                   + lambda_l2 * l2_reg

    mse_loss         = MSE(pred, target)
    smoothness_loss  = mean |pred[:, t] - pred[:, t-1]|
    l2_reg           = sum ||param||_2^2 over model parameters
    """

    def __init__(self, lambda_smooth: float = 0.01, lambda_l2: float = 1e-5):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_l2 = lambda_l2

    def forward(self, pred, target, model: nn.Module):
        """
        Args:
            pred:      predicted patches (B, ...)
            target:        target patches   (B, ...)
            model:          model whose parameters will be L2-regularized
        """
        # Reconstruction MSE
        mse_loss = F.mse_loss(pred, target)

        # Temporal smoothness on predicted recovered signal
        smoothness_loss = torch.mean(
            torch.abs(pred[:, 1:] - pred[:, :-1])
        )

        # L2 regularization over all parameters
        l2_reg = sum(torch.sum(p ** 2) for p in model.parameters())

        recon_loss = mse_loss + self.lambda_smooth * smoothness_loss + self.lambda_l2 * l2_reg
        return recon_loss#, {"mse": mse_loss.detach(), "smoothness": smoothness_loss.detach(), "l2": l2_reg.detach(), "total": recon_loss.detach()}

def random_matplotlib_color():
    """Generate single random RGB color (0-1 range)."""
    return tuple(random.random() for _ in range(3))

def generate_colors(n_colors: int = 1, method: str = 'hsv_uniform'):
    """
    Generate visually distinct colors for 64-channel signals (ECG/PPG).
    
    Args:
        n_colors: Number of colors (default 1)
        method: 'hsv_uniform' (recommended), 'random', or 'tab20'
    
    Returns:
        List of (R,G,B) tuples, matplotlib-ready
    """
    if method == 'tab20':
        # Matplotlib's built-in qualitative colormap (20 colors, repeats)
        cmap = plt.cm.tab20(np.linspace(0, 1, n_colors))
        return [tuple(color[:3]) for color in cmap]
    
    elif method == 'random':
        # Pure random (may have clashes)
        return [random_matplotlib_color() for _ in range(n_colors)]
    
    else:  # 'hsv_uniform' - best perceptual separation
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            sat = np.clip(0.7 + 0.2 * np.sin(i * np.pi / 8), 0.6, 1.0)
            val = np.clip(0.9 + 0.05 * np.sin(i * np.pi / 4), 0.85, 1.0)
            color = plt.cm.hsv(hue)[:3]
            color = tuple(np.array(color) * np.array([1, sat, val]))
            colors.append(color)
        return colors