import mne
import torch
import torch.nn as nn
import numpy as np
from utils import unmask_channels
from metrics import evaluate_model
from mne.channels.interpolation import _make_interpolation_matrix
import os 

cwd = os.getcwd()
sep = os.sep
datapath = cwd + sep + "eeg_data" + sep + "preprocessed"

class SSILayer(torch.nn.Module):
    def __init__(self, W_matrix):  # W: (8_in, 64_out)
        super().__init__()
        self.register_buffer('W', W_matrix)
        self.n_in = W_matrix.shape[0]
    
    def forward(self, eeg_sparse):
        # Input is (B, N_channels, T) -> convert to (B, T, N_channels) for einsum
        if eeg_sparse.shape[1] <= self.n_in:  # (B, 8ch, T)
            eeg_input = eeg_sparse.transpose(1, 2)  # (B, T, 8ch)
        else:  # Already (B, T, N_channels)
            eeg_input = eeg_sparse[..., :self.n_in]  # Take first N_in channels
        
        # Now einsum works: (B, T, 8) x (8, 64) -> (B, T, 64)
        sr_out = torch.einsum('btj,jk->btk', eeg_input, self.W)
        
        # Return original format: (B, 64ch, T)
        return sr_out.transpose(1, 2)
    
if __name__ == "__main__":
    
    fold = 1

    montage = mne.channels.make_standard_montage('standard_1005')
    target_ch_names = montage.ch_names[:64]
    positions = montage.get_positions()['ch_pos']

    SSI_MATRICES = {}
    for density, ch_idx in unmask_channels.items():
        # Get 3D positions
        sparse_pos = np.array([positions[montage.ch_names[i]] for i in ch_idx])
        target_pos = np.array([positions[ch] for ch in target_ch_names])
        
        # ✅ REAL spherical spline matrix (64_out, N_in)
        W = _make_interpolation_matrix(sparse_pos, target_pos)
        
        SSI_MATRICES[density] = torch.tensor(W.T, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')  # (N_in, 64_out)
        print(f"Density {density}: W shape = {SSI_MATRICES[density].shape}")

    # Instantiated layers for your pipeline
    ssi_8to64 = SSILayer(SSI_MATRICES[8])
    #ssi_16to64 = SSILayer(SSI_MATRICES[16]) 
    #ssi_32to64 = SSILayer(SSI_MATRICES[32])

    ssi = {8: ssi_8to64,
           #16: ssi_16to64,
           #32: ssi_32to64
           }
    
    to_df = []
    fold = 1
    for ch in [8]: #, 16, 32]:
        print(f"Evaluating SSI model for {ch}→64 channels...")
        test_datapath = datapath + sep + f"test_sr_spatial_{ch}to64chs_{fold}.pt"
        dataset = torch.load(test_datapath, weights_only=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        #print(next(iter(dataloader))[0].shape, next(iter(dataloader))[1].shape)
        model = ssi[ch]
        results, results_raw = evaluate_model(model, dataloader, num_channels=ch)
        print(f"Results for {ch}→64chs SSI:")
        to_append = {}
        to_append['method'] = f'SSI_{ch}to64'
        for key, value in results.items():
            to_append[key] = value
        to_df.append(to_append)
    
    import pandas as pd
    df = pd.DataFrame(to_df)
    df.to_csv(os.path.join(cwd, 'ssi_results.csv'))