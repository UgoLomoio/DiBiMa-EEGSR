import os
import sys
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

current_dir = os.getcwd()
parent_dir = os.path.join(current_dir, os.pardir)  # or os.path.dirname(current_dir)
parent_abs = os.path.abspath(parent_dir)
print(f"Adding to sys.path: {parent_abs}")
sys.path.insert(0, parent_abs)
from utils import *

datapath = os.path.join(current_dir, 'data')

map_runs_dataset = {
    "mmi": range(1, 15),
    "seed": None  # All files in folder
}
dict_n_patients = {
    "mmi": 109,
    "seed": 15
}

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

class EEGDatasetClassification(Dataset):

    def __init__(self, subject_ids, data_folder, dataset_name = "mmi", input_type = "hr", model_sr = None, sr_type="temporal", seconds=10, verbose=False, demo=False, num_channels=64, multiplier=2):
        
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
        self.input_type = input_type  # 'hr', 'lr' or 'sr' 
        self.model_sr = model_sr  # For 'sr' input_type

        self.scaler = StandardScaler()  #MinMaxScaler(feature_range=(0, 1))

        self.datas, self.labels, self.positions, _ = download_eegbci_data(
            subject_ids, runs=self.runs, project_path=self.project_path,
            demo=self.demo,
            dataset_name=self.dataset_name, verbose=self.verbose, is_classification=True
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
        label = self.labels[idx]
        pos = self.ref_position  # (C, 3)

        batch_size = hr_data.size(0)
        
        if self.input_type == "lr" or self.input_type == "sr":
            if self.sr_type == "temporal":
                data = self._downsample(hr_data, self.multiplier)
            else:  # spatial
                if isinstance(hr_data, torch.Tensor):
                    data = hr_data.clone()
                else:
                    data = hr_data.copy() # numpy array
                data = get_lr_data_spatial(data, num_channels=self.num_channels)
            data = torch.tensor(data, dtype=torch.float32)

        elif self.input_type == "sr":
            if self.model_sr.__class__.__name__ == 'DiBiMa_Diff':
                t = torch.full((batch_size,), self.model_sr.scheduler.num_train_timesteps - 1, device=hr_data.device, dtype=torch.long) # (B,)
                # Diffuse HR
                noise_hr = torch.randn_like(hr_data)
                x_t_hr = self.model_sr.scheduler.add_noise(hr_data, noise_hr, t)
                # Model prediction (same signature as training)
                data = self.model_sr(x_t_hr, t, data, pos)  # (B, C_HR, L')
            else:
                data = self.model_sr(data)  # (B, C_HR, L')
            return data, label
        
        else:  # 'hr'
            data = hr_data

        return data.float(), label.long()
        
def load_model_weights(model, model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(model_path):
        if model.__class__.__name__ == 'DiBiMa_Diff':
            model.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        raise Exception(f"Model weights file not found at {model_path}.")
    return model

