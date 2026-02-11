
import math
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
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
from sklearn.model_selection import train_test_split
from mne.preprocessing import find_bad_channels_lof, find_bad_channels_maxwell

# Set random seeds for reproducibility
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

#case 1 of https://ieeexplore.ieee.org/document/9796118

seed_channels = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
mmi_channels = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz"]


map_seed_channels = {seed_channels[i]: i for i in range(len(seed_channels))}
map_mmi_channels = {mmi_channels[i]: i for i in range(len(mmi_channels))}

case1_seed = {
    'x2': ['AF3', 'AF4', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'CB1', 'CB2', 'PO3', 'PO4', 'PO5', 'PO7', 'PO6', 'PO8', 'POz', 'O1', 'Oz', 'O2'], 
    'x4': ['Fp1', 'Fp2', 'F5', 'Fz', 'F6', 'C3', 'Cz', 'C4', 'T7', 'T8', 'P5', 'Pz', 'P6', 'O1', 'O2'],
    'x8': ['AF3', 'AF4', 'FC5', 'FC6', 'CP5', 'CP6', 'PO5', 'PO6']
}
case1_mmi = {
    "x2": ["Fpz", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "Oz", "Iz"], 
    "x4": ["Fp1", "Fp2", "F5", "Fz", "F6", "T7", "C3", "Cz", "C4", "T8", "P5", "Pz", "P6", "O1", "Oz", "O2"],
    "x8": ["AF3", "AF4", "FC5", "FC6", "CP5", "CP6", "O1", "O2"]
}

case2_mmi = {
    "x2": ["Fp1", "Fp2", "F5", "F1", "F2", "F6", "FC5", "FC1", "FC2", "FC6", "T9", "C5", "C1", "C2", "C6", "T10", "CP5", "CP1", "CP2", "CP6", "P5", "P1", "P2", "P6", "O1", "O2"],   
    "x4": ["Fpz", "AF3", "AF4", "FC5", "FC1", "FC2", "FC6", "T9", "CP5", "CP1", "CP2", "CP6", "T10", "PO3", "PO4", "Oz"],
    "x8": ["Fp1", "Fp2", "Fz", "T7", "T8", "Pz", "O1", "O2"]
}
case2_seed = {
    "x2": ["Fpz", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "Oz"], 
    "x4": ["Fpz", "AF3", "AF4", "FC5", "FC1", "FC2", "FC6", "CP5", "CP1", "CP2", "CP6", "CB1", "PO3", "PO4", "CB2", "Oz"],
    "x8": ["Fp1", "Fp2", "Fz", "T7", "T8", "Pz", "O1", "O2"]
}

unmask_channels = {
                    "mmi":{
                            "x2": [map_mmi_channels[i] for i in case1_mmi['x2']],

                            "x4": [map_mmi_channels[i] for i in case1_mmi['x4']],  # Your 16ch (frontal/central/parietal) #[0, 2, 4, 6, 14, 16, 18, 20, 22, 25, 27, 42, 43, 56, 58, 61]
                            
                            "x8": [map_mmi_channels[i] for i in case1_mmi['x8']] # Even 10-10 coverage
                    },
                    "seed":{
                            "x2": [map_seed_channels[i] for i in case1_seed['x2']],

                            "x4": [map_seed_channels[i] for i in case1_seed['x4']],  # Frontal/central/parietal

                            "x8": [map_seed_channels[i] for i in case1_seed['x8']]  # Even 10-10 coverage
                            }
                    }

#reverse ch_name:i to i:ch_name
unmask_channels = {
    dataset: {
        key: sorted(value) for key, value in channels.items()
    } for dataset, channels in unmask_channels.items()
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

map_labels_seed = {
    -1: 0,  # negative
    0: 1,   # neutral
    1: 2    # positive
}

labels_subject_mapping = {
    1: 1,
    2: 0,
    3: -1,
    4: -1,
    5: 0,
    6: 1,
    7: -1,
    8: 0,
    9: 1,
    10: 1,
    11: 0,
    12: -1,
    13: 0,
    14: 1,
    15: -1
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

def get_lr_data_spatial(eeg_64, dataset_name, sr_ratio):
    
    """Extract low-res from 64ch EEG using unmask_channels"""
    indices = unmask_channels[dataset_name][f"x{sr_ratio}"]
    if eeg_64.ndim == 2:
        return eeg_64[indices, :]
    elif eeg_64.ndim == 3:
        return eeg_64[:, indices, :]
# -------------------------------
# Preprocessing MNE
# -------------------------------
def _preprocess_raw(raw, dataset_name, verbose=False):
    fs_cut = 40 if dataset_name == "mmi" else 50
    raw.pick_types(eeg=True, eog=False, stim=False)
    
    if dataset_name == "seed":
        # SKIP dev_head_t - not needed for EEG preprocessing [web:42]
        
        # Bad channel detection & cleaning (your function)
        raw = detect_and_clean_seed_trial(raw)
        if raw is None:
            return None
        
        # SEED filters
        raw.notch_filter(50.0, fir_design='firwin', verbose=verbose)
        raw.filter(1.0, fs_cut, fir_design='firwin', verbose=verbose)
        
        # Scale (confirm units with raw.plot() first)
        data = raw.get_data()
        #data = data * 1e-4
        data = (data - data.mean(1, keepdims=True)) / (data.std(1, keepdims=True) + 1e-8)
        raw._data = data
        
    else:
        raw.notch_filter(50.0, fir_design='firwin', verbose=verbose)
        raw.filter(0.5, fs_cut, fir_design='firwin', verbose=verbose)
    
    return raw


# Extract valid EEG positions (C, 3), nan-mask if needed
def get_electrode_positions(raw, channel_order=None):
    """
    Extract electrode positions from raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage set
    channel_order : list, optional
        Desired channel order for reordering
        
    Returns
    -------
    locs : torch.Tensor
        Electrode positions (C_eeg, 3)
    """
    picks = mne.pick_types(raw.info, eeg=True)
    locs = np.array([raw.info['chs'][p]['loc'][:3] for p in picks])  # (C_eeg, 3)
    
    # Check for NaNs and report which channels are problematic
    if np.isnan(locs).any():
        nan_channels = []
        for i, p in enumerate(picks):
            if np.isnan(locs[i]).any():
                ch_name = raw.info['chs'][p]['ch_name']
                nan_channels.append(ch_name)
        
        # Get available montage channels for debugging
        montage = raw.get_montage()
        if montage is not None:
            available_channels = list(montage.get_positions()['ch_pos'].keys())
            print(f"⚠️ Available montage channels: {available_channels[:10]}...")  # Show first 10
        
        raise ValueError(
            f"NaNs persist after montage; check channel names. "
            f"Problematic channels: {nan_channels}. "
            f"Raw channels: {[raw.info['chs'][p]['ch_name'] for p in picks]}"
        )
    
    if channel_order:  # Reorder to model input (e.g., 64 chs)
        reorder_idx = [channel_order.index(ch) for ch in raw.ch_names if ch in channel_order]
        locs = locs[reorder_idx]
    
    locs = torch.tensor(locs, dtype=torch.float32)  # (C_eeg, 3)
    return locs

def load_seed_channel_positions(filepath):

    channels = []
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                ch_name = parts[-1] # Or parts[-1] if label last
                theta_deg = float(parts[1]) # Azimuth, negative=left
                phi_frac = float(parts[2]) # Elevation fraction (~0-0.6)
                phi_rad = np.radians(phi_frac * 90) # Adjust scale to ~90° max
                theta_rad = np.radians(theta_deg)
                x = np.sin(phi_rad) * np.cos(theta_rad)
                y = np.sin(phi_rad) * np.sin(theta_rad)
                z = np.cos(phi_rad)
                channels.append(ch_name)
                positions.append([x, y, z])
    return channels, np.array(positions)

def set_montage(signal, dataset_name, pos=None, channel_names=None, fs=160):
    """
    Set montage for raw EEG data.
    
    For MMI: pos=None, uses standard_1020
    For SEED: pos=array of positions from .pos file
    """
    
    #print(channel_names)

    # Clean channel names
    cleaned_names = []
    for ch in channel_names:
        clean = ch.strip().rstrip('.')
        cleaned_names.append(clean)
    
    # Create MNE RawArray
    info = mne.create_info(ch_names=cleaned_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(signal, info)
    
    if dataset_name == "mmi":
        # MMI dataset: Use standard montage
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            #montage_channels = list(montage.get_positions()['ch_pos'].keys())
            #print(f"Montage channels available: {montage_channels[:10]}...")  # Show first 10
            #print(f"Raw channels: {cleaned_names}")
            raw.set_montage(montage, on_missing='warn')
        except Exception as e:
            print(f"⚠️ Error setting standard montage: {e}")
    else:
        # SEED dataset: Use custom positions from .pos file
        try:            
            ch_pos = {channel_names[i]: pos[i] for i in range(len(channel_names))}
            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos,
                coord_frame='head'
            )
            raw.set_montage(montage, on_missing='warn')
        except Exception as e:
            print(f"⚠️ Error setting custom montage: {e}")
            raise
    
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
                #print(raw.ch_names)
                if channel_names is None:
                    channel_names = raw.ch_names
                else:
                    if channel_names != raw.ch_names:
                        raise ValueError("Inconsistent channel names across recordings.")
                
                #we don't have the positions for mmi, we wait to set the montage and then extract them 
                raw = set_montage(signal, "mmi", pos=None, channel_names=channel_names)
                if raw is None:
                    print(f"⚠️ set_montage returned None for {local_path}")
                    continue
                
                raw = _preprocess_raw(raw, dataset_name="mmi", verbose=verbose)
                if raw is None:
                    print(f"⚠️ _preprocess_raw returned None for {local_path}")
                    continue

                position = get_electrode_positions(raw, channel_order=None)
                if position is None:
                    print(f"⚠️ No valid electrode positions for {local_path}")
                    continue

                positions.append(position)
                data = raw.get_data()
                data = data*1e3  # Convert to µV
                data = data.astype(np.float32)
                datas.append(data)
                label = map_runs_mmi[run]
                labels.append(map_labels_mmi[label])

            except Exception as e:
                print(f"⚠️ Exception during processing {local_path}: {e}")
                import traceback
                traceback.print_exc()  # This will show the exact line causing the error
                continue

    # Clean up downloaded files
    path_to_remove = os.path.join(
        os.path.dirname(project_path), "mmi",
        'MNE-eegbci-data', 'files', 'eegmmidb'
    )
    if os.path.exists(path_to_remove):
        shutil.rmtree(path_to_remove)
        
    labels = torch.tensor(np.array(labels), dtype=torch.int32)
    positions = np.array(positions)
    positions = torch.tensor(positions, dtype=torch.float32)
    return datas, labels, positions, channel_names

def load_mat(filepath):
    mat = scipy.io.loadmat(filepath)
    return mat

def load_seed_data(subject_ids, project_path, demo=False, verbose=False):

    seed_datapath = os.path.join(project_path, "Preprocessed_EEG")
    files = os.listdir(seed_datapath)
    if len(files) == 0:
        raise ValueError("No SEED data found in the specified path.")

    datas = []
    labels = []
    positions = []

    channels, position = load_seed_channel_positions(os.path.join(project_path, "channel_62_pos.locs"))
    #print(position)
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

                            raw = set_montage(data, "seed", pos=position, channel_names=channels, fs=sfreq)   
                            raw = _preprocess_raw(raw, dataset_name="seed", verbose=verbose)
                            if raw is None:
                                continue
                            data = raw.get_data()
                            data = data.astype(np.float32)
                            datas.append(data)
                            label = map_labels_seed[labels_subject_mapping[subject]]
                            labels.append(label)
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

class EEGWindowsDataset(Dataset):
    """
    Minimal dataset for pre-processed, pre-windowed EEG data.
    Assumes windows are already normalized and ready to use.
    Only generates LR on-the-fly during __getitem__.
    """
    def __init__(self, windows, labels, positions, sr_type="temporal", dataset_name="mmi",
                 target_channels=64, multiplier=2, channel_names=None, fs_hr=160):
        """
        Args:
            windows: Tensor/array (N, C, T) - preprocessed HR windows
            labels: Tensor/array (N,) - class labels
            positions: Tensor/array (N, C, 3) or (C, 3) - electrode positions
            sr_type: 'temporal' or 'spatial'
            num_channels: int - for spatial SR (number of LR channels)
            multiplier: int - temporal downsampling factor
            channel_names: list - optional channel names
        """

        self.datas_hr = torch.tensor(windows, dtype=torch.float32) if not isinstance(windows, torch.Tensor) else windows.float()
        self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long()
        
        # Handle positions
        if isinstance(positions, torch.Tensor):
            self.positions = positions.float()
        else:
            self.positions = torch.tensor(positions, dtype=torch.float32)
        
        # Broadcast if single position (C, 3) -> (N, C, 3)
        if self.positions.dim() == 2:
            self.positions = self.positions.unsqueeze(0).expand(len(self.datas_hr), -1, -1)
        
        self.sr_type = sr_type
        self.target_channels = target_channels
        self.multiplier = multiplier
        self.channel_names = channel_names
        self.dataset_name = dataset_name
        self.num_classes = len(torch.unique(self.labels))
        self.ref_position = self.positions[0]  
        self.fs_hr = fs_hr  
        self.fs_lr = int(fs_hr//self.multiplier) if sr_type == "temporal" else fs_hr
        self.num_channels = math.ceil(self.target_channels / self.multiplier) if sr_type == "spatial" else self.datas_hr.shape[1]
        print(f"✅ EEGWindowsDataset: {len(self.datas_hr)} windows, shape {self.datas_hr.shape}")

    def _downsample(self, data, factor):
        """Temporal downsampling via signal.resample"""
        downsampled_length = data.shape[1] // factor
        return signal.resample(data, downsampled_length, axis=1)

    def __len__(self):
        return len(self.datas_hr)

    def __getitem__(self, idx):
        hr_data = self.datas_hr[idx]  # (C, T)
        pos = self.positions[idx]      # (C, 3)
        label = self.labels[idx]       # scalar
        
        # Generate LR on-the-fly
        if self.sr_type == "temporal":
            lr_data = self._downsample(hr_data.numpy(), self.multiplier)
            lr_data = torch.tensor(lr_data, dtype=torch.float32)
        else:  # spatial
            lr_data = get_lr_data_spatial(hr_data.clone(), dataset_name=self.dataset_name, sr_ratio=self.multiplier)
        
        return lr_data, hr_data, pos, label

# -------------------------------
# EEGDataset for Super-Resolution EEG
# -------------------------------
class EEGDataset(Dataset):
    """
    Dataset EEG preprocessato per Super-Resolution EEG (EEG BCI dataset).
    Genera segmenti LR (160/sr_factor Hz) e HR (160 Hz) sincronizzati.
    """
    def __init__(self, subject_ids, data_folder, dataset_name = "mmi", sr_type="temporal", seconds=10, verbose=False, demo=False, num_channels=64, multiplier=2, normalize=False, is_classification=False):
        
        self.project_path = data_folder
        self.verbose = verbose
        self.normalize = normalize
        self.demo = demo
        self.seconds = seconds
        self.num_channels = num_channels    
        self.dataset_name = dataset_name
        self.runs = map_runs_dataset[self.dataset_name]
        self.sr_type = sr_type  # 'temporal' or 'spatial'
        self.multiplier = multiplier  # Downsampling factor for temporal or spatial SR
        self.fs_hr = 160  if self.dataset_name == "mmi" else 200
        self.hr_window_length = self.fs_hr * self.seconds  
        self.is_classification = is_classification 

        self.scaler = StandardScaler()  #MinMaxScaler(feature_range=(0, 1))

        self.datas, self.labels, self.positions, self.channel_names = download_eegbci_data(
            subject_ids, runs=self.runs, project_path=self.project_path,
            demo=self.demo, dataset_name=self.dataset_name, verbose=self.verbose, is_classification=self.is_classification
        )   
        
        self.positions = self.positions.cpu()
        self.labels = self.labels.cpu()

        self.num_classes = 6 if self.dataset_name == "mmi" else 3  #len(torch.unique(self.labels))
        
        self.ref_position = self.positions[0]  # Assuming all raws have same channel positions

        self.datas_hr, self.positions = self._split_windows(self.hr_window_length)
        print(f"\n✅ Number of segments created: {len(self.datas_hr)}")
        
        #self.datas_hr = self._zscore_normalization(torch.tensor(self.datas_hr)).numpy()
        #print(f"\nData z-score normalization complete: {self.datas_hr.shape}")
        
        if self.normalize:
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
        # self.datas_hr shape: (N_windows, C, T)
        N, C, T = self.datas_hr.shape
        # Flatten to (N*C, T) - treat each channel independently
        data_reshaped = self.datas_hr.reshape(N * C, T)  # (N*C, T)
        # Fit scaler per channel
        data_norm = self.scaler.fit_transform(data_reshaped)
        # Reshape back to (N, C, T)
        data_normalized = torch.tensor(data_norm, dtype=torch.float32).reshape(N, C, T)
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
        self.labels = torch.tensor(np.array(labels), dtype=torch.int32).cpu()
        return datas_hr, positions
        
    def __len__(self):
        return len(self.datas_hr)

    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
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
            lr_data = get_lr_data_spatial(lr_data, dataset_name=self.dataset_name, sr_ratio=self.multiplier)
        lr_data = torch.tensor(lr_data, dtype=torch.float32)       
        return lr_data.to(device), hr_data.to(device), pos.to(device), label.to(device)

def train_test_val_split_patients(patients, test_size=0.2, val_size=0.1, random_state=42):

    train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=random_state)
    relative_val_size = val_size / (1 - test_size)
    train_patients, val_patients = train_test_split(train_patients, test_size=relative_val_size, random_state=random_state)
    
    return train_patients, val_patients, test_patients

def train_test_val_split(windows, labels, positions, test_size=0.2, val_size=0.1, random_state=42):

    train_data, test_data, train_labels, test_labels, train_positions, test_positions = train_test_split(
        windows, labels, positions, test_size=test_size, random_state=random_state, stratify=labels
    )
    relative_val_size = val_size / (1 - test_size)
    train_data, val_data, train_labels, val_labels, train_positions, val_positions = train_test_split(
        train_data, train_labels, train_positions, test_size=relative_val_size, random_state=random_state, stratify=train_labels
    )
    
    return (train_data, train_labels, train_positions), (val_data, val_labels, val_positions), (test_data, test_labels, test_positions)

def plot_mean_timeseries(timeseries, save_path=None):
        
    fig = plt.figure(figsize=(12, 4))
    for key, value in timeseries.items():
        if value.ndim == 3:
            value = value[0]  # Take the first sample in the batch
        print(f"Plotting {key} with shape {value.shape}, min {value.min()}, max {value.max()}")
        mean_signal = np.mean(value, axis=0)  # Mean across channels
        plt.plot(mean_signal, label=key)
    plt.title(f'Mean Timeseries Across Channels')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    plt.close(fig)

def clear_directory(path, ignore=[]):
    """Remove all files in the specified directory."""
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename == '.gitignore' or filename in ignore:
                continue  # Skip .gitignore files and ignored files/folders
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def add_zero_channels(input_tensor, target_channels=64, dataset_name="mmi", multiplier=2):
    """Add zero channels to input_tensor to match target_channels."""

    if input_tensor.ndim == 2:
        batch_size = None
        nchs = input_tensor.size(0)
        length = input_tensor.size(1)
    else:
        batch_size = input_tensor.size(0)
        nchs = input_tensor.size(1)
        length = input_tensor.size(2)

    if batch_size is None:
        input_target = torch.zeros((target_channels, length), device=input_tensor.device)
    else:
        input_target = torch.zeros((batch_size, target_channels, length), device=input_tensor.device)

    channels_to_use = unmask_channels[dataset_name][f"x{multiplier}"]
    for i, ch in enumerate(channels_to_use):
        if batch_size is None:
            input_target[ch, :] = input_tensor[i, :]
        else:
            input_target[:, ch, :] = input_tensor[:, i, :]
    return input_target

def compute_latents(model, dataloader, device, split = "train", map_labels=None):

    """
    Computes latent vectors from model for all data in dataloader.
    
    Args:
        model: PyTorch model with return_latent=True support
        dataloader: yields (eeg_lr, eeg_hr, pos, label)
        device: torch device
    """

    latent_vectors = [] 
    labels = []

    print("Computing latents for training set", end='\r')
    for i, (eeg_lr, eeg_hr, pos, label) in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}", end='\r')
        
        eeg_lr = eeg_lr.to(device)
        eeg_hr = eeg_hr.to(device) 
        pos = pos.to(device)
        label = label.to(device)  # In case needed
        
        with torch.no_grad():
            if model.__class__.__name__ == "DiBiMa_Diff":
                batch_size = eeg_lr.size(0)
                t = torch.full((batch_size,), model.train_scheduler.num_train_timesteps - 1, 
                             device=device, dtype=torch.long)
                x_t_hr = torch.randn_like(eeg_hr).to(device)
                latent = model(x_t_hr, t, lr=eeg_lr, pos=pos, label=label, return_latent=True)[-1]
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

    return np.vstack(latent_vectors), np.array(labels)

def plot_umap_latent_space(model, dataloader_train, dataloader_test, save_path=None, map_labels=None, seed=42):
    """
    Plots UMAP projection of model latent space colored by labels.
    
    Args:
        model: PyTorch model with return_latent=True support
        dataloader_train: yields (eeg_lr, eeg_hr, pos, label)
        dataloader_test: yields (eeg_lr, eeg_hr, pos, label)
        save_path: Optional save path for figure
        map_labels: Optional dict {int: str} for label mapping
        seed: Random seed for reproducibility
    """
    # Collect all latent vectors and labels
    latent_vectors_train = []
    latent_vectors_test = []
    labels_train = []
    labels_test = []
    
    model.eval()
    device = next(model.parameters()).device
    
    print("Computing latents for training set...")
    latent_vectors_train, labels_train = compute_latents(model, dataloader_train, device, split="train", map_labels=map_labels)
    print("\nComputing latents for test set...")
    latent_vectors_test, labels_test = compute_latents(model, dataloader_test, device, split="test", map_labels=map_labels)
   
    print(f"UMAP on {latent_vectors_train.shape[0]} samples, {latent_vectors_train.shape[1]} dims")
    
    # Fit UMAP ONCE on full dataset
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', 
                   random_state=seed, n_jobs=-1)
    embedding_train = reducer.fit_transform(latent_vectors_train)
    embedding_test = reducer.transform(latent_vectors_test)

    # Plot by unique labels
    plt.figure(figsize=(12, 8))
    u_labels = np.unique(labels_test)
    
    print("Generating scatter plot...")
    for ul in u_labels:
        mask = labels_test == ul
        if np.sum(mask) > 0:
            plt.scatter(embedding_test[mask, 0], embedding_test[mask, 1], 
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
    def __init__(
        self, 
        lambda_spectral: float = 0.1,
        lambda_l2: float = 1e-6,  # Reduced
        freq_bands: dict = None
    ):
        super().__init__()
        self.lambda_spectral = lambda_spectral
        self.lambda_l2 = lambda_l2
        
        # EEG frequency bands (Hz)
        self.freq_bands = freq_bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    def spectral_loss(self, pred, target, sample_rate=250):
        """Preserve spectral power in EEG bands"""
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Power spectral density
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2
        
        return F.mse_loss(pred_psd, target_psd)
    
    def forward(self, pred, target, model: nn.Module):
        # Base reconstruction
        mse_loss = F.mse_loss(pred, target)
        
        # Spectral preservation
        spectral_loss = self.spectral_loss(pred, target)
        
        # L2 only on convolutional/linear weights, exclude biases/norms
        l2_reg = sum(
            torch.sum(p ** 2) 
            for name, p in model.named_parameters() 
            if 'weight' in name and len(p.shape) >= 2
        )
        
        total_loss = (
            mse_loss 
            + self.lambda_spectral * spectral_loss
            + self.lambda_l2 * l2_reg
        )
        
        return total_loss

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
    
class EEGSuperResolutionLoss(nn.Module):
    def __init__(self, 
                 lambda_grad=1.0, 
                 lambda_corr=0.5,
                 lambda_freq=0.3,
                 use_freq_loss=True):
        """
        Composite loss for EEG temporal super-resolution
        
        Args:
            lambda_grad: Weight for gradient loss (controls sharp transitions)
            lambda_corr: Weight for Pearson correlation loss (temporal coherence)
            lambda_freq: Weight for frequency domain loss (spectral fidelity)
            use_freq_loss: Whether to include frequency domain loss
        """
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_corr = lambda_corr
        self.lambda_freq = lambda_freq
        self.use_freq_loss = use_freq_loss
        self.mse = nn.MSELoss()
        self.__name__ = "EEGSuperResolutionLoss"
        
    def gradient_loss(self, pred, target):
        """
        Gradient difference loss - captures rate of change
        Helps preserve sharp transitions
        """
        # Temporal gradient (first derivative along time axis)
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        
        return F.mse_loss(pred_grad, target_grad)
    
    def pearson_correlation_loss(self, pred, target):
        """
        Pearson correlation coefficient loss
        Preserves temporal waveform patterns
        """
        # Flatten spatial dimensions, keep batch and time
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        
        # Center the data
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        
        # Covariance
        cov = (pred_centered * target_centered).sum(dim=1)
        
        # Standard deviations
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
        
        # Pearson correlation coefficient
        pcc = cov / (pred_std * target_std + 1e-8)
        
        # Loss is 1 - PCC (want to maximize correlation)
        return (1 - pcc).mean()
    
    def frequency_domain_loss(self, pred, target):
        """
        FFT-based frequency domain loss
        Preserves spectral characteristics
        """
        # Apply FFT along time dimension
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Compute magnitude spectrum loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss = F.mse_loss(pred_mag, target_mag)
        
        # Compute phase loss (optional, helps with temporal alignment)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss
    
    def forward(self, pred, target):
        """
        Combined loss function
        
        Args:
            pred: Predicted HR EEG (B, C, T)
            target: Ground truth HR EEG (B, C, T)
        """
        # Base MSE loss
        loss_mse = self.mse(pred, target)
        
        # Gradient loss for sharp transitions
        loss_grad = self.gradient_loss(pred, target)
        
        # Pearson correlation for temporal coherence
        loss_corr = self.pearson_correlation_loss(pred, target)
        
        # Frequency domain loss
        if self.use_freq_loss:
            loss_freq = self.frequency_domain_loss(pred, target)
        else:
            loss_freq = 0.0
        
        # Combined loss
        total_loss = (loss_mse + 
                     self.lambda_grad * loss_grad + 
                     self.lambda_corr * loss_corr)
        
        if self.use_freq_loss:
            total_loss += self.lambda_freq * loss_freq
        
        # Return total and individual losses for monitoring
        return total_loss, {
            'mse': loss_mse.item(),
            'gradient': loss_grad.item(),
            'correlation': loss_corr.item(),
            'frequency': loss_freq.item() if self.use_freq_loss else 0.0,
            'total': total_loss.item()
        }
    

def detect_and_clean_seed_trial(raw, reject_threshold=0.16):
    """Version-safe bad channel detection for SEED"""
    
    # Basic stats without montage dependency
    data = raw.get_data()
    
    # Flat channels: variance near zero
    variances = data.var(axis=1)
    flat_idx = np.where(variances < 1e-12)[0]
    
    # Noisy outliers: z-score variance >5
    var_z = (variances - variances.mean()) / (variances.std() + 1e-8)
    noisy_idx = np.where(var_z > 5)[0]
    
    # Saturated: low first differences (repeats)
    diffs = np.abs(np.diff(data, axis=1)).mean(axis=1)
    sat_idx = np.where(diffs < 1e-10)[0]
    
    all_bads = list(set(list(flat_idx) + list(noisy_idx) + list(sat_idx)))
    raw.info['bads'] = [raw.ch_names[i] for i in all_bads]
    
    n_bad_frac = len(all_bads) / raw.info['nchan']
    #print(f"Trial bad fraction: {n_bad_frac:.1%} ({len(all_bads)}/{raw.info['nchan']})")
    
    if n_bad_frac > reject_threshold:
        return None
    
    # Interpolate (works without montage if no spatial ops needed)
    try:
        raw.interpolate_bads(reset_bads=False)
    except Exception as e:
        print(f"Interpolation failed (ok for DL): {e}")
        # Still return - bads marked for your model to ignore
    
    return raw

if __name__ == "__main__":
    
    for dataset_name in ["mmi", "seed"]:
        sr_type = "spatial"
        for sr_ratio in ["x2", "x4", "x8"]:
            channels = unmask_channels[dataset_name][sr_ratio]
            print(f"{dataset_name} {sr_type} {sr_ratio}: {len(channels)} channels -> {channels}")
