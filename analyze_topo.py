import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from utils import EEGDataset
import os 
from sklearn.model_selection import train_test_split
from utils import get_lr_data_spatial, get_lr_data_temporal, unmask_channels

from models import DiBiMa_Diff, DiBiMa_nn
from test import *
from utils import set_montage
import mne 
from torch.nn.functional import l1_loss

def next_power_of_2(n):
    return 2 ** int(np.ceil(np.log2(n)))


def load_model(in_channels, target_channels, fs_lr, fs_hr, seconds, sr_type):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    str_param = f'{int(in_channels)}to{int(target_channels)}chs' if sr_type == 'spatial' else f'x{int(fs_hr/fs_lr)}'
    model_path = f'./model_weights/fold_1/DiBiMa_eeg_{str_param}_{sr_type}_1.pth'
    print(f"Loading model from {model_path}...")
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
        use_electrode_embedding=best_params["use_electrode_embedding"],  
        merge_type=best_params['merge_type']
    ).to(device)
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
    model_pl.model = load_model_weights(model, model_path).to(device)
    model_pl.eval()
    return model_pl

def compute_topomap_psd(signals, fs_hr, sr_type, target_channel_names, dataset_name, pos):

    bands = dict(δ=(1,4), θ=(4,8), α=(8,13), β=(13,30), γ=(30,45))
    row_labels = list(signals)
    col_labels = list(bands)
    nrows, ncols = len(row_labels), len(col_labels)

    # First pass: collect all psd_band_log for global vlim
    all_psd_values = []
    raw_dict = {}
    psd_dict = {}
    for i, label in enumerate(row_labels):
        print(f"Processing {label}... signal shape: {signals[label].shape}")
        fs = fs_hr
        if sr_type == 'spatial':
            if "LR" in label:
                input_channels = signals[label].shape[0]
                channel_names = list(target_channel_names[unmask_channels[input_channels]])
            else:
                channel_names = list(target_channel_names)
        else:  # temporal
            channel_names = list(target_channel_names)

        raw = set_montage(signals[label], dataset_name, sr_type, label, pos, channel_names, fs)
        raw_dict[label] = raw  # Store raw for later use
        psd_dict[label] = {}
        for j, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            n_fft = min(1024, next_power_of_2(raw.n_times))
            n_per_seg = min(256, n_fft // 4)
            n_overlap = n_per_seg // 2
            spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=80, n_fft=n_fft,
                                        n_per_seg=n_per_seg, n_overlap=n_overlap, verbose=False)
            psds, freqs = spectrum.get_data(return_freqs=True)
            if psds.ndim == 3:
                psds = psds[0]
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            if idx_band.any():
                psd_band = np.mean(psds[:, idx_band], axis=1)
            psd_band_log = 10 * np.log10(np.maximum(psd_band, 1e-12))
            all_psd_values.extend(psd_band_log)
            psd_dict[label][band_name] = psd_band_log  # Store for later use

    global_vmin, global_vmax = np.min(all_psd_values), np.max(all_psd_values)
    print(f"Global vlim: ({global_vmin:.2f}, {global_vmax:.2f}) dB/Hz")
    return raw_dict, psd_dict, global_vmin, global_vmax 

def plot_topomap(models, dataloader, index, fs_hr, sr_type, target_channel_names, dataset_name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, hr, pos, _ = list(iter(dataloader))[index]
    with torch.no_grad():

        hr = hr[0].to(device)
        pos = pos[0].to(device)

        if sr_type == 'spatial':
            lr_8 = get_lr_data_spatial(hr, num_channels=math.ceil(hr.shape[0]/8)).to(device)
            lr_4 = get_lr_data_spatial(hr, num_channels=math.ceil(hr.shape[0]/4)).to(device)
            lr_2 = get_lr_data_spatial(hr, num_channels=math.ceil(hr.shape[0]/2)).to(device)
        else:
            lr_8 = torch.tensor(get_lr_data_temporal(hr.cpu().detach().numpy(), factor=8)).to(device)
            lr_4 = torch.tensor(get_lr_data_temporal(hr.cpu().detach().numpy(), factor=4)).to(device)
            lr_2 = torch.tensor(get_lr_data_temporal(hr.cpu().detach().numpy(), factor=2)).to(device)

        print(f"Generating SR signals for topomap plotting...")
        print(f"HR shape: {hr.shape}, LR_/8 shape: {lr_8.shape}, LR_/4 shape: {lr_4.shape}, LR_/2 shape: {lr_2.shape}")
        sr_8 = models['x8'].predict(hr.unsqueeze(0), lr_8.unsqueeze(0), pos.unsqueeze(0))[0]
        sr_4 = models['x4'].predict(hr.unsqueeze(0), lr_4.unsqueeze(0), pos.unsqueeze(0))[0]
        sr_2 = models['x2'].predict(hr.unsqueeze(0), lr_2.unsqueeze(0), pos.unsqueeze(0))[0]
        
        if sr_type == 'temporal':
            lr_8 = signal.resample(lr_8.cpu().detach().numpy(), lr_8.shape[-1] * 8, axis=-1)
            lr_4 = signal.resample(lr_4.cpu().detach().numpy(), lr_4.shape[-1] * 4, axis=-1)
            lr_2 = signal.resample(lr_2.cpu().detach().numpy(), lr_2.shape[-1] * 2, axis=-1)
            lr_8 = torch.tensor(lr_8).to(device)
            lr_4 = torch.tensor(lr_4).to(device)
            lr_2 = torch.tensor(lr_2).to(device)

    signals = {
        'LR_/8': lr_8.cpu().detach().numpy(),
        'LR_/4': lr_4.cpu().detach().numpy(),
        'LR_/2': lr_2.cpu().detach().numpy(),
        'SR_x8': sr_8.cpu().detach().numpy(),
        'SR_x4': sr_4.cpu().detach().numpy(),
        'SR_x2': sr_2.cpu().detach().numpy(),
        'HR': hr.cpu().detach().numpy()
    }

    bands = dict(δ=(1,4), θ=(4,8), α=(8,13), β=(13,30), γ=(30,45))
    row_labels = list(signals)
    col_labels = list(bands)
    nrows, ncols = len(row_labels), len(col_labels)

    raw_dict, psd_dict, global_vmin, global_vmax = compute_topomap_psd(signals, fs_hr, sr_type, target_channel_names, dataset_name, pos)  

    # Second pass: plot with fixed vlim and capture first image
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
    first_im = None

    for i, label in enumerate(row_labels):
        raw = raw_dict[label]  # Reuse stored raw
        for j, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            ax = axes[i, j]
            # Reuse stored psd_band_log
            psd_band_log = psd_dict[label][band_name]
            
            im, _ = mne.viz.plot_topomap(psd_band_log, raw.info, axes=ax, cmap='hot',
                                        show=False, vlim=(global_vmin, global_vmax))
            if first_im is None:
                first_im = im  # Capture first mappable
                
            if j == 0:
                ax.text(-1, 0.5, label, transform=ax.transAxes, rotation=0,
                            va='center', fontsize=15, weight='bold')
            if i == 0:
                ax.text(0.5, 1, band_name, transform=ax.transAxes,
                            ha='center', fontsize=15, weight='bold')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(first_im, cax=cbar_ax, label='Power (dB/Hz)')
    plt.suptitle('EEG Band PSD Topomaps by Input Resolution', fontsize=16, weight='bold', y=2.00)
    plt.tight_layout()
    plt.savefig(f'imgs/topomap_{sr_type}_index{index}_{dataset_name}.png', dpi=300)
    plt.show()

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    data_folder = './eeg_data'  # Path to EEG data
    dataset_name = 'mmi'  # or 'seed'
    batch_size = 32
    demo = False  
    num_channels = 64 if dataset_name == 'mmi' else 62
    seed = 2 
    seconds = 10

    print("Downloading test data...")
    patients = list(range(1, 110)) if dataset_name == 'mmi' else None
    test_size = 0.2
    _, test_patients = train_test_split(patients, test_size=test_size, random_state=seed)
    data_path = os.path.join(data_folder, dataset_name)
    dataset_test = EEGDataset(subject_ids=test_patients, data_folder=data_path, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=num_channels)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    print("Test datasets loaded successfully.")    

    print("Computing frequency domain MAE on test set...")       
    target_channel_names = np.array(dataset_test.channel_names)

    sr_types = ['spatial', 'temporal']
    sr_signals = {}
    hr_signals = {}
    models = {}
    for sr_type in sr_types:
        models[sr_type] = {}
        hr_signals[sr_type] = []
        sr_signals[sr_type] = {"x8": [], "x4": [], "x2": []}
        if sr_type == 'temporal':
            models_path = {
                'x8': './model_weights/fold_1/DiBiMa_eeg_x8_temporal_1.pth',
                'x4': './model_weights/fold_1/DiBiMa_eeg_x4_temporal_1.pth',
                'x2': './model_weights/fold_1/DiBiMa_eeg_x2_temporal_1.pth'
            }
            in_channels = target_channels = num_channels
            fs_hr = 160 if dataset_name == 'mmi' else 200
            for model_path in models_path.values():
                fs_lr = fs_hr // int(model_path.split('_')[-3][1:])
                name = model_path.split('_')[-3]  # e.g., 'x8'
                models[sr_type][name] = load_model(in_channels, target_channels, fs_lr, fs_hr, seconds, sr_type)

            with torch.no_grad():
                for _, hr, pos, _ in iter(dataloader_test):
                    hr = hr.to(device)
                    pos = pos.to(device)
                    
                    lr_8 = torch.tensor(get_lr_data_temporal(hr.cpu().detach().numpy(), factor=8)).to(device)
                    lr_4 = torch.tensor(get_lr_data_temporal(hr.cpu().detach().numpy(), factor=4)).to(device)
                    lr_2 = torch.tensor(get_lr_data_temporal(hr.cpu().detach().numpy(), factor=2)).to(device)

                    sr_8 = models[sr_type]['x8'].predict(hr, lr_8, pos)
                    sr_4 = models[sr_type]['x4'].predict(hr, lr_4, pos)
                    sr_2 = models[sr_type]['x2'].predict(hr, lr_2, pos)
                    
                    lr_8 = signal.resample(lr_8.cpu().detach().numpy(), lr_8.shape[-1] * 8, axis=-1)
                    lr_4 = signal.resample(lr_4.cpu().detach().numpy(), lr_4.shape[-1] * 4, axis=-1)
                    lr_2 = signal.resample(lr_2.cpu().detach().numpy(), lr_2.shape[-1] * 2, axis=-1)
                    lr_8 = torch.tensor(lr_8)
                    lr_4 = torch.tensor(lr_4)
                    lr_2 = torch.tensor(lr_2)
                    hr_signals[sr_type].append(hr.cpu())    
                    sr_signals[sr_type]['x8'].append(sr_8.cpu())
                    sr_signals[sr_type]['x4'].append(sr_4.cpu())
                    sr_signals[sr_type]['x2'].append(sr_2.cpu())
        else:
            target_channels = 64 if dataset_name == 'mmi' else 62
            models_path = {
                'x8': f'./model_weights/fold_1/DiBiMa_eeg_{math.ceil(target_channels/8)}to{target_channels}chs_spatial_1.pth',
                'x4': f'./model_weights/fold_1/DiBiMa_eeg_{math.ceil(target_channels/4)}to{target_channels}chs_spatial_1.pth',
                'x2': f'./model_weights/fold_1/DiBiMa_eeg_{math.ceil(target_channels/2)}to{target_channels}chs_spatial_1.pth'
            }
            fs_lr = fs_hr = 160 if dataset_name == 'mmi' else 200
            for model_path in models_path.values():
                in_channels = int(model_path.split('_')[4].split('to')[0])  # e.g., '8' from '8to64chs'
                target_channels = num_channels
                model = load_model(in_channels, target_channels, fs_lr, fs_hr, seconds, sr_type)
                name = "x" + str(math.ceil(num_channels / in_channels))
                models[sr_type][name] = model
            
            with torch.no_grad():   
                for _, hr, pos, _ in iter(dataloader_test):
                    hr = hr.to(device)
                    pos = pos.to(device)
                    
                    lr_8 = get_lr_data_spatial(hr, num_channels=math.ceil(target_channels/8)).to(device)
                    lr_4 = get_lr_data_spatial(hr, num_channels=math.ceil(target_channels/4)).to(device)
                    lr_2 = get_lr_data_spatial(hr, num_channels=math.ceil(target_channels/2)).to(device)
                    sr_8 = models[sr_type]['x8'].predict(hr, lr_8, pos)
                    sr_4 = models[sr_type]['x4'].predict(hr, lr_4, pos)
                    sr_2 = models[sr_type]['x2'].predict(hr, lr_2, pos)
                    hr_signals[sr_type].append(hr.cpu())
                    sr_signals[sr_type]['x8'].append(sr_8.cpu())
                    sr_signals[sr_type]['x4'].append(sr_4.cpu())
                    sr_signals[sr_type]['x2'].append(sr_2.cpu())
                    lr_2 = lr_2.cpu()
                    lr_4 = lr_4.cpu()
                    lr_8 = lr_8.cpu()

    print("Calculating MAE in frequency bands...")
    maes = {}
    for sr_type in sr_types:
        print(f"Processing {sr_type} SR...")
        maes[sr_type] = {}
        hr_all = torch.cat([torch.tensor(arr) for arr in hr_signals[sr_type]], dim=0)
        for scale in ['x8', 'x4', 'x2']:
            print(f"  Scale {scale}...")
            sr_all = torch.cat([torch.tensor(arr) for arr in sr_signals[sr_type][scale]], dim=0)
            maes[sr_type][scale] = {}

            n_fft = min(256, hr_all.shape[-1] // 4)
            psd_hr = mne.time_frequency.psd_array_welch(hr_all.cpu().numpy(), fs_hr, n_fft=n_fft,
                                                    n_per_seg=128, n_overlap=64, n_jobs=-1,
                                                    fmin=0, fmax=fs_hr/2, verbose=False)[0]
            psd_sr = mne.time_frequency.psd_array_welch(sr_all.cpu().numpy(), fs_hr, n_fft=n_fft,
                                                    n_per_seg=128, n_overlap=64, n_jobs=-1,
                                                    fmin=0, fmax=fs_hr/2, verbose=False)[0]
            bands = dict(δ=(1,4), θ=(4,8), α=(8,13), β=(13,30), γ=(30,45))
            freqs = mne.time_frequency.psd_array_welch(hr_all[:1], fs_hr, n_fft=n_fft,
                                                    n_per_seg=128, n_overlap=64, n_jobs=-1,
                                                    fmin=0, fmax=fs_hr/2, verbose=False)[1]  # Shared freqs

            for band_name, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                if np.sum(mask) == 0: continue # Skip if no frequencies in this band
                psd_hr_band = np.mean(psd_hr[:, :, mask], axis=2)  # [n_ch]
                psd_sr_band = np.mean(psd_sr[:, :, mask], axis=2)
                psd_hr_band_log = 10 * np.log10(np.maximum(psd_hr_band, 1e-12))
                psd_sr_band_log = 10 * np.log10(np.maximum(psd_sr_band, 1e-12))
                mae = l1_loss(torch.tensor(psd_sr_band_log), torch.tensor(psd_hr_band_log), reduction='mean').item()
                maes[sr_type][scale][band_name] = mae
            maes[sr_type][scale]['Overall'] = torch.mean(torch.tensor(list(maes[sr_type][scale].values()))).item()

    print("MAE Results (dB/Hz):")
    for sr_type in sr_types:
        print(f"\n{sr_type.upper()} SR:")
        for scale in ['x8', 'x4', 'x2']:
            print(f"  Scale {scale}:")
            for band_name, mae in maes[sr_type][scale].items():
                print(f"    {band_name}: {mae:.4f} dB/Hz")
    df = pd.DataFrame.from_dict({(i,j): maes[i][j] 
                           for i in maes.keys()
                           for j in maes[i].keys()}, orient='index')
    df.index.names = ['SR Type', 'Scale']
    df.to_csv(f'mae_topo_{dataset_name}.csv')
    print("\nMAE Summary Table (dB/Hz):")
    print(df)

    print("Plotting topographic maps for a sample from the test set...")
    for sr_type in sr_types:
        print(f"Plotting for {sr_type} SR...")
        index = torch.argmin(torch.tensor([maes[sr_type][scale]['Overall'] for scale in ['x8']]))  # index with minimum MAE or any other criteria
        plot_topomap(models[sr_type], dataloader_test, index=index, fs_hr=160 if dataset_name == 'mmi' else 200,
                     sr_type=sr_type, target_channel_names=target_channel_names,
                     dataset_name=dataset_name)

if __name__ == "__main__":
    main()