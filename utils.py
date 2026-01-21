
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
import mne
from mne.datasets import eegbci
from scipy import signal
import shutil
import matplotlib.pyplot as plt
from umap import UMAP
import random
import gc
import torch
import scipy

# Set random seeds for reproducibility
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

unmask_channels = {
    8: [0, 7, 14, 21, 28, 35, 42, 49],  # Core 10-20: Fp1,Fp2,Fz,F3,F4,C3,C4,Pz

    16: [0, 2, 4, 6, 14, 16, 18, 20, 22, 25, 27, 42, 43, 56, 58, 61],  # Your 16ch (frontal/central/parietal)
    
    32: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
         32, 34, 36, 38, 40, 42, 43, 44, 46, 48, 50, 52, 54, 56, 58, 61]  # Even 10-10 coverage
}

map_runs_dataset = {
    "mmi": range(1, 15),
    "seed": None  # All files in folder
}

map_tasks = {
    'task1': 'open-close right or left fist',
    'task3': 'open-close both fists or both feet',
    'task2': 'imagine right or left fist',
    'task4': 'imagine both fists or both feet'
}

map_runs_mmi = {
    1: 'eyes_open',
    2: 'eyes_closed',
    3: map_tasks['task1'],
    4: map_tasks['task2'],
    5: map_tasks['task3'],
    6: map_tasks['task4'],
    7: map_tasks['task1'],
    8: map_tasks['task2'],
    9: map_tasks['task3'],
    10: map_tasks['task4'],
    11: map_tasks['task1'],   
    12: map_tasks['task2'],
    13: map_tasks['task3'],
    14: map_tasks['task4']
}

map_labels_mmi = {
                'eyes_open': 0,
                'eyes_closed': 1,
                'open-close right or left fist': 2,
                'open-close both fists or both feet': 3,
                'imagine right or left fist': 4,
                'imagine both fists or both feet': 5
}
map_labels_mmi_rev = {v: k for k, v in map_labels_mmi.items()}


map_annotations = {
    "T0": "rest",
    "T1": "right or left fist",
    "T2": "both fists or both feet"
}

def get_lr_data_temporal(eeg_hr, factor=2):
    """Downsample high-res EEG temporally by factor."""
    if eeg_hr.ndim == 2:
        _, num_samples = eeg_hr.shape
    elif eeg_hr.ndim == 3:
        _, _, num_samples = eeg_hr.shape
    downsampled_length = num_samples // factor
    eeg_lr = signal.resample(eeg_hr, downsampled_length, axis=-1)
    return eeg_lr

def get_lr_data_spatial(eeg_64, num_channels=8):
    
    """Extract low-res from 64ch EEG using unmask_channels"""
    indices = unmask_channels[num_channels]
    if eeg_64.ndim == 2:
        return eeg_64[indices, :]
    elif eeg_64.ndim == 3:
        return eeg_64[:, indices, :]
# -------------------------------
# Preprocessing MNE
# -------------------------------
def _preprocess_raw(raw, verbose=False):
    raw.pick_types(eeg=True, eog=False, stim=False)
    raw.notch_filter(50.0, fir_design='firwin', verbose=verbose)
    raw.filter(0.5, 40.0, fir_design='firwin', verbose=verbose)
    raw.set_eeg_reference('average')
    return raw

# Extract valid EEG positions (C, 3), nan-mask if needed
def get_electrode_positions(raw, channel_order=None):
    picks = mne.pick_types(raw.info, eeg=True)
    locs = np.array([raw.info['chs'][p]['loc'][:3] for p in picks])  # (C_eeg, 3)
    if np.isnan(locs).any():
        raise ValueError("NaNs persist after montage; check channel names.")
    if channel_order:  # Reorder to model input (e.g., 64 chs)
        reorder_idx = [channel_order.index(ch) for ch in raw.ch_names if ch in channel_order]
        locs = locs[reorder_idx]
    locs = torch.tensor(locs, dtype=torch.float32)  # (C_eeg, 3)
    return locs

def set_montage(signal, dataset_name, sr_type, label, pos, channel_names, fs):
    
    # Find and drop T9/T10 from BOTH signal and channel_names (mmi-specific)
    # if dataset_name == "mmi":
    #    drop_idx = [i for i, ch in enumerate(channel_names) if ch in ['T9', 'T10']]
    #    signal = np.delete(signal, drop_idx, axis=0)  # Remove rows: 64→62
    #    channel_names = [ch for ch in channel_names if ch not in ['T9', 'T10']]  # 64→62
        
    #    print(f"Dropped {len(drop_idx)} channels, signal now {signal.shape}, names {len(channel_names)}")  # Debug
    
    info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg')
    
    if sr_type == 'spatial':
        if "LR" in label:
            input_channels = signal.shape[0]
            selected_channels = unmask_channels[input_channels]
            if pos.ndim == 2:
                pos_subset = pos[selected_channels, :].cpu().detach().numpy()
            else:
                pos_subset = pos[0, selected_channels, :].cpu().detach().numpy()
            montage = mne.channels.make_dig_montage(
                ch_pos={ch_name: pos_subset[i] for i, ch_name in enumerate(channel_names)},
                coord_frame='head'
            )
        else:
            if dataset_name == 'mmi':
                montage = mne.channels.make_standard_montage('standard_1020')
            else:
                montage = mne.channels.read_custom_montage('./eeg_data/seed/channel_62_pos.locs')
    else:  # temporal or None
        if dataset_name == 'mmi':
            montage = mne.channels.make_standard_montage('standard_1020')
        else:
            montage = mne.channels.read_custom_montage('./eeg_data/seed/channel_62_pos.locs')
    
    info.set_montage(montage)  # Now matches perfectly
    raw = mne.io.RawArray(signal, info)
    
    return raw

def download_mmi_data(subject_ids, runs, project_path, demo = False, verbose=False, is_classification=False):

    datas = []
    labels = []
    positions = []
    channel_names = None

    for i, subject in enumerate(subject_ids):
        #print(f"Processing subject {subject}")
        print(f"⬇️  Downloading/Reading data for subject: {i+1}/{len(subject_ids)}", end='\r')
        if demo:
            if i == 1:
                break  # For demo, process only first subject
        for run in runs:
            if is_classification and run not in [1, 2]:
                continue  # Skip non-classification runs

            local_path = os.path.join(project_path, f'S{subject:03d}R{run:02d}.edf')
            #print(f"Checking local path: {local_path}")
            if not os.path.exists(local_path):
                print(f"Downloading S{subject:03d}R{run:02d}.edf...")
                try:
                    eegbci.load_data(subject, [run], path=project_path, update_path=True, force_update=False)
                    downloaded = os.path.join(
                        os.path.dirname(project_path), "mmi",
                        'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0',
                        f'S{subject:03d}', f'S{subject:03d}R{run:02d}.edf'
                    )
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    shutil.move(downloaded, local_path)
                    print(f"Saved to: {local_path}")
                except Exception as e:
                    print(f"Download error: {e}")
                    continue
            try:
                raw = mne.io.read_raw_edf(local_path, preload=True, verbose=verbose)
                mne.datasets.eegbci.standardize(raw) 
                signal = raw.get_data()
                if channel_names is None:
                    channel_names = raw.ch_names
                raw = set_montage(signal, "mmi", sr_type=None, label="HR", pos=None, channel_names=channel_names, fs=160)

                raw = _preprocess_raw(raw, verbose=verbose)
                data = raw.get_data()
                data = data*1e6  # scale to microvolts
                data = data.astype(np.float32)
                datas.append(data)
                label = map_runs_mmi[run]
                labels.append(label)
                position = get_electrode_positions(raw, channel_order=None)
                positions.append(position)

            except Exception as e:
                print(f"⚠️ Exception during processing {local_path}: {e}")
                continue

    # Clean up downloaded files
    path_to_remove = os.path.join(
        os.path.dirname(project_path), "mmi",
        'MNE-eegbci-data', 'files', 'eegmmidb'
    )
    if os.path.exists(path_to_remove):
        shutil.rmtree(path_to_remove)

    labels = [map_labels_mmi[lbl] for lbl in labels]
    labels = torch.tensor(np.array(labels), dtype=torch.int32)
    positions = np.array(positions)
    positions = torch.tensor(positions, dtype=torch.float32)
    return datas, labels, positions, channel_names

def load_mat(filepath):
    mat = scipy.io.loadmat(filepath)
    return mat

def load_seed_channel_positions(filepath):
    channels = []
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                ch_name = parts[-1]  # Or parts[-1] if label last
                theta_deg = float(parts[1])  # Azimuth, negative=left
                phi_frac = float(parts[2])   # Elevation fraction (~0-0.6)
                phi_rad = np.radians(phi_frac * 90)  # Adjust scale to ~90° max
                theta_rad = np.radians(theta_deg)
                x = np.sin(phi_rad) * np.cos(theta_rad)
                y = np.sin(phi_rad) * np.sin(theta_rad)
                z = np.cos(phi_rad)
                channels.append(ch_name)
                positions.append([x, y, z])
    return channels, np.array(positions)

def load_seed_data(subject_ids, project_path, demo=False, verbose=False):

    seed_datapath = os.path.join(project_path, "Preprocessed_EEG")
    files = os.listdir(seed_datapath)
    if len(files) == 0:
        raise ValueError("No SEED data found in the specified path.")

    datas = []
    labels = []
    positions = []

    channels, position = load_seed_channel_positions(os.path.join(project_path, "channel_62_pos.locs"))
    i = 0
    for file in files:
        for subject in subject_ids:
            print(f"Processing subject {i+1}/{len(subject_ids)}", end='\r')
            if demo:
                if i == 1:
                    break  # For demo, process only first subject
            if file.startswith(f'{int(subject)}_'):
                filepath = os.path.join(seed_datapath, file)
                try:
                    data_hr = load_mat(filepath)
                    sfreq = 200  # Original SEED sampling rate
                    eeg_keys = [key for key in data_hr.keys() if "eeg" in key.lower()]
                    if len(eeg_keys) == 0:
                        print(f"No EEG data found in {filepath}.")
                        continue
                    else:
                        for key in eeg_keys:
                            
                            data = data_hr[key]  # Shape: (channels, samples)
                            #raw = mne.io.RawArray(data, mne.create_info(ch_names=channels, sfreq=sfreq, ch_types='eeg'))
                            raw = set_montage(data, "seed", sr_type=None, label="HR", pos=None, channel_names=channels, fs=sfreq)
                            
                            raw = _preprocess_raw(raw, verbose=verbose)

                            data = raw.get_data()
                            data = data*1e6  # scale to microvolts
                            data = data.astype(np.float32)
                            datas.append(data)
                            labels.append(int(subject))
                            positions.append(torch.tensor(position, dtype=torch.float32))

                    i += 1
                except Exception as e:
                    print(f"⚠️ Exception during processing {filepath}: {e}")

    positions = np.array(positions)
    positions = torch.tensor(positions, dtype=torch.float32)
    labels = np.array(labels, dtype=np.int32)
    labels = torch.tensor(labels)
    return datas, labels, positions, channels

def download_eegbci_data(subject_ids, runs, project_path, demo=False, dataset_name="mmi", is_classification=False, verbose=False):
    """
    Scarica i dati EEG BCI per i soggetti e le sessioni specificate.
    Salva i file EDF localmente in project_path.
    """
    if dataset_name.lower() not in ["mmi", "seed"]:
        raise ValueError("Dataset not supported. Use 'mmi' or 'seed'.")
    
    if dataset_name.lower() == "mmi":
        return download_mmi_data(subject_ids, runs, project_path, demo=demo, verbose=verbose, is_classification=is_classification)
    else:
        return load_seed_data(subject_ids, project_path, demo=demo, verbose=verbose)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------
# EEGDataset for Super-Resolution EEG
# -------------------------------
class EEGDataset(Dataset):
    """
    Dataset EEG preprocessato per Super-Resolution EEG (EEG BCI dataset).
    Genera segmenti LR (160/sr_factor Hz) e HR (160 Hz) sincronizzati.
    """
    def __init__(self, subject_ids, data_folder, dataset_name = "mmi", sr_type="temporal", seconds=10, verbose=False, demo=False, num_channels=64, multiplier=2):
        
        self.project_path = data_folder
        self.verbose = verbose
        self.demo = demo
        self.seconds = seconds
        self.num_channels = num_channels    
        self.dataset_name = dataset_name
        self.runs = map_runs_dataset[self.dataset_name]
        self.sr_type = sr_type  # 'temporal' or 'spatial'
        self.multiplier = multiplier  # Downsampling factor for temporal or spatial SR
        self.fs_hr = 160  if self.dataset_name == "mmi" else 200
        self.hr_window_length = self.fs_hr * self.seconds  

        self.scaler = StandardScaler()  #MinMaxScaler(feature_range=(0, 1))

        self.datas, self.labels, self.positions, self.channel_names = download_eegbci_data(
            subject_ids, runs=self.runs, project_path=self.project_path,
            demo=self.demo,
            dataset_name=self.dataset_name, verbose=self.verbose, is_classification=False
        )   

        self.ref_position = self.positions[0]  # Assuming all raws have same channel positions

        self.datas_hr, self.positions = self._split_windows(self.hr_window_length)
        print(f"\n✅ Number of segments created: {len(self.datas_hr)}")
        
        self.datas_hr = self._zscore_normalization(torch.tensor(self.datas_hr)).numpy()
        print(f"\nData z-score normalization complete: {self.datas_hr.shape}")
        self.datas_hr = self._normalize_data()
        print(f"\nData normalization complete: {self.datas_hr.shape}")

    def _zscore_normalization(self, data):
        # Z-score per channel (common for EEG DL)
        mean = data.mean(dim=-1, keepdim=True)  # [B,C,1]
        std = data.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-5)  # Avoid div0
        normalized = (data - mean) / std  # [-3,3] typical
        return normalized
    
    def _normalize_data(self):
        data_normalized = []
        for data in self.datas_hr:
            data_reshaped = data.T  # Shape: (num_samples, num_channels)
            data_norm = self.scaler.fit_transform(data_reshaped)
            data_normalized.append(torch.tensor(data_norm.T))  # Shape: (num_channels, num_samples)
        data_normalized = torch.stack(data_normalized)
        return data_normalized    
    
    def _downsample(self, data, factor):
        _, num_samples = data.shape
        downsampled_length = num_samples // factor
        downsampled_data = signal.resample(data, downsampled_length, axis=1)
        return downsampled_data        
    
    def _split_windows(self, window_length, stride=None):  # stride=window_length for non-overlap
        if stride is None:
            stride = window_length  # Non-overlapping by default    
        datas_hr = []
        positions = []
        labels = []
        for i, data in enumerate(self.datas):  # self.datas = list of (C, T_i) arrays
            print(f"Splitting data {i+1}/{len(self.datas)} (len={data.shape[1]})", end='\r')
            position = self.positions[i]  # (C, 3)     
            T = data.shape[1]
            num_windows = (T - window_length) // stride + 1  # Floor div for valid windows
            for j in range(num_windows):
                start = j * stride
                end = start + window_length
                window = data[:, start:end].astype(np.float32)  # (C, window_length)
                datas_hr.append(window)
                positions.append(position)  # Same pos for all windows from this trial
                labels.append(self.labels[i])
        datas_hr = torch.tensor(np.array(datas_hr), dtype=torch.float32)  # (N_windows_total, C, W)
        positions = torch.stack(positions)  # (N_windows_total, C, 3)
        self.labels = torch.tensor(np.array(labels), dtype=torch.int32)
        return datas_hr, positions
        
    def __len__(self):
        return len(self.datas_hr)

    def __getitem__(self, idx):
        hr_data = self.datas_hr[idx]
        pos = self.positions[idx]
        label = self.labels[idx]
        if self.sr_type == "temporal":
            lr_data = self._downsample(hr_data, self.multiplier)
        else:  # spatial
            if isinstance(hr_data, torch.Tensor):
                lr_data = hr_data.clone()
            else:
                lr_data = hr_data.copy() # numpy array
            lr_data = get_lr_data_spatial(lr_data, num_channels=self.num_channels)
        return lr_data, hr_data, pos, label

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

def plot_umap_latent_space(model, dataloader, save_path=None, map_labels=None, seed=42):
    """
    Plots UMAP projection of model latent space colored by labels.
    
    Args:
        model: PyTorch model with return_latent=True support
        dataloader: yields (eeg_lr, eeg_hr, pos, label)
        save_path: Optional save path for figure
        map_labels: Optional dict {int: str} for label mapping
        seed: Random seed for reproducibility
    """
    # Collect all latent vectors and labels
    latent_vectors = []
    labels = []
    
    model.eval()
    device = next(model.parameters()).device
    
    print("Extracting latent representations...")
    for i, (eeg_lr, eeg_hr, pos, label) in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}", end='\r')
        
        eeg_lr = eeg_lr.to(device)
        eeg_hr = eeg_hr.to(device) 
        pos = pos.to(device)
        label = label.to(device)  # In case needed
        
        with torch.no_grad():
            if model.__class__.__name__ == "DiBiMa_Diff":
                batch_size = eeg_lr.size(0)
                t = torch.full((batch_size,), model.scheduler.num_train_timesteps - 1, 
                             device=device, dtype=torch.long)
                noise_hr = torch.randn_like(eeg_hr).to(device)
                x_t_hr = model.scheduler.add_noise(eeg_hr, noise_hr, t)
                latent = model(x_t_hr, t, lr=eeg_lr, pos=pos, return_latent=True)[-1]
            else:
                latent = model(eeg_lr, return_latent=True)[-1]
            
            # Flatten to (B, D)
            latent = latent.reshape(latent.size(0), -1).cpu().numpy()
            latent_vectors.append(latent)
            
            # Collect labels per sample
            for l in label.flatten():
                labels.append(map_labels[l.item()] if map_labels else l.item())
        
        # Memory management
        del eeg_lr, eeg_hr, pos, label, latent
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\nAggregating data...")
    latent_vectors = np.vstack(latent_vectors)  # (N_total, D)
    labels = np.array(labels)
    
    print(f"UMAP on {latent_vectors.shape[0]} samples, {latent_vectors.shape[1]} dims")
    
    # Fit UMAP ONCE on full dataset
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', 
                   random_state=seed, n_jobs=-1)
    embedding = reducer.fit_transform(latent_vectors)
    
    # Plot by unique labels
    plt.figure(figsize=(12, 8))
    u_labels = np.unique(labels)
    
    print("Generating scatter plot...")
    for ul in u_labels:
        mask = labels == ul
        if np.sum(mask) > 0:
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       label=str(ul), alpha=0.6, s=20)
            print(f"  {ul}: {np.sum(mask)} points")
    
    plt.title('UMAP Latent Space Projection', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()
    print("UMAP visualization complete!")


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