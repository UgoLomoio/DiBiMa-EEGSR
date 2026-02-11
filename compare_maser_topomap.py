import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from utils import EEGDataset, get_lr_data_spatial, unmask_channels, add_zero_channels
import os 
from sklearn.model_selection import train_test_split
from utils import set_montage
import mne 
import sys
from models import DiBiMa_Diff, DiBiMa_nn, DiBiMa
from test import load_model_weights,train_scheduler, val_scheduler, loss_fn, learning_rate, prediction_type, debug, epochs, models_path
    
models_path = models_path + os.sep + "fold_1"

def next_power_of_2(n):
    return 2 ** int(np.ceil(np.log2(n)))

def load_maser_model(config_path, model_path):
    """Load a pre-trained MASER model from the specified path."""
    cwd = os.getcwd()
    sep = os.sep
    maser_project_path = cwd + sep + "maser"
    sys.path.append(maser_project_path)
    from inference import load_model
    model, args = load_model(config_path, model_path)
    return model, args


def load_dibima_model(device, dataset_name='mmi', multiplier=8):
    """Load a pre-trained DiBiMa diffusion model."""
    sr_type = "spatial"
    target_channels = 64 if dataset_name == 'mmi' else 62
    input_channels = len(unmask_channels[dataset_name][f'x{multiplier}'])
    seconds = 2
    
    dibima_params = {
        "use_electrode_embedding": False,
        "use_label": False,
        "use_lr_conditioning": True,
        "use_mamba": True,
        "use_diffusion": True,
        "n_mamba_layers": 2,
        "mamba_dim": 128,
        "mamba_d_state": 16,
        "mamba_version": 1,
        "n_mamba_blocks": 3,
        "internal_residual": True,
        "merge_type": "add"
    }
    
    model_nn = DiBiMa_nn(
        target_channels=target_channels,
        num_channels=input_channels,
        fs_lr=160 if dataset_name == 'mmi' else 200,
        fs_hr=160 if dataset_name == 'mmi' else 200,
        seconds=seconds,
        residual_global=True,
        residual_internal=dibima_params["internal_residual"],
        use_subpixel=True,
        sr_type=sr_type,
        use_mamba=dibima_params["use_mamba"],
        use_diffusion=dibima_params["use_diffusion"],
        n_mamba_layers=dibima_params["n_mamba_layers"],
        mamba_dim=dibima_params["mamba_dim"],
        mamba_d_state=dibima_params["mamba_d_state"],
        mamba_version=dibima_params["mamba_version"],
        n_mamba_blocks=dibima_params["n_mamba_blocks"],
        use_positional_encoding=False,
        use_electrode_embedding=dibima_params["use_electrode_embedding"],
        merge_type=dibima_params['merge_type'],
        use_label=dibima_params['use_label'],
        use_lr_conditioning=dibima_params['use_lr_conditioning'],
        multiplier=multiplier,
        dataset_name=dataset_name
    ).to(device)
    
    model = DiBiMa_Diff(
        model_nn,
        train_scheduler=train_scheduler,
        val_scheduler=val_scheduler,
        criterion=loss_fn,
        learning_rate=learning_rate,
        predict_type=prediction_type,
        debug=debug,
        epochs=epochs,
        plot=False
    ).to(device)
    
    model_path = os.path.join(models_path, f'DiBiMa_eeg_{input_channels}to{target_channels}chs_spatial_{dataset_name}_1.pth')
    model.model = load_model_weights(model.model, model_path)
    return model


def load_bima_model(device, dataset_name='mmi', multiplier=8):
    """Load a pre-trained BiMa regression model."""
    sr_type = "spatial"
    target_channels = 64 if dataset_name == 'mmi' else 62
    input_channels = len(unmask_channels[dataset_name][f'x{multiplier}'])
    seconds = 2
    
    bima_params = {
        "use_electrode_embedding": False,
        "use_label": False,
        "use_lr_conditioning": False,
        "use_mamba": True,
        "use_diffusion": False,
        "n_mamba_layers": 2,
        "mamba_dim": 128,
        "mamba_d_state": 16,
        "mamba_version": 1,
        "n_mamba_blocks": 3,
        "internal_residual": True,
        "merge_type": "add"
    }
    
    model_nn = DiBiMa_nn(
        target_channels=target_channels,
        num_channels=input_channels,
        fs_lr=160 if dataset_name == 'mmi' else 200,
        fs_hr=160 if dataset_name == 'mmi' else 200,
        seconds=seconds,
        residual_global=True,
        residual_internal=bima_params["internal_residual"],
        use_subpixel=True,
        sr_type=sr_type,
        use_mamba=bima_params["use_mamba"],
        use_diffusion=bima_params["use_diffusion"],
        n_mamba_layers=bima_params["n_mamba_layers"],
        mamba_dim=bima_params["mamba_dim"],
        mamba_d_state=bima_params["mamba_d_state"],
        mamba_version=bima_params["mamba_version"],
        n_mamba_blocks=bima_params["n_mamba_blocks"],
        use_positional_encoding=False,
        use_electrode_embedding=bima_params["use_electrode_embedding"],
        merge_type=bima_params['merge_type'],
        use_label=bima_params['use_label'],
        use_lr_conditioning=bima_params['use_lr_conditioning'],
        multiplier=multiplier,
        dataset_name=dataset_name
    ).to(device)
    
    model = DiBiMa(
        model_nn,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        debug=debug,
        epochs=epochs,
        plot=False
    ).to(device)
    
    model_path = os.path.join(models_path, f'BiMa_eeg_{input_channels}to{target_channels}chs_spatial_{dataset_name}_1.pth')
    model.model = load_model_weights(model.model, model_path)
    return model


def compute_topomap_psd(signals, fs_hr, dataset_name, pos, target_channel_names, multiplier=8):
    """Compute PSD for all signals and return raw objects and PSDs."""
    
    row_labels = list(signals.keys())
    
    # First pass: collect all psd values for global vlim
    all_psd_values = []
    raw_dict = {}
    psd_dict = {}
    
    for label in row_labels:
        print(f"Processing {label}... signal shape: {signals[label].shape}")
        
        if f"LR_/{multiplier}" in label:
            channel_names = list(target_channel_names[unmask_channels[dataset_name][f"x{multiplier}"]])
        else:
            channel_names = list(target_channel_names)

        signals_scaled = signals[label] * 1e3  

        print(signals_scaled.shape, len(channel_names))
        raw = set_montage(signals_scaled, dataset_name, pos, channel_names, fs_hr)
        raw_dict[label] = raw
        
        # Compute PSD across full frequency range
        n_fft = min(1024, next_power_of_2(raw.n_times))
        n_per_seg = min(256, n_fft // 4)
        n_overlap = n_per_seg // 2
        
        spectrum = raw.compute_psd(
            method='welch', 
            fmin=0.5, 
            fmax=75, 
            n_fft=n_fft,
            n_per_seg=n_per_seg, 
            n_overlap=n_overlap, 
            verbose=False
        )
        
        psds, freqs = spectrum.get_data(return_freqs=True)
        if psds.ndim == 3:
            psds = psds[0]
        
        # Average across all frequencies
        psd_avg = np.mean(psds, axis=1)
        psd_avg_log = 10 * np.log10(np.maximum(psd_avg, 1e-12))
        
        all_psd_values.extend(psd_avg_log)
        psd_dict[label] = psd_avg_log
    
    global_vmin, global_vmax = np.min(all_psd_values), np.max(all_psd_values)
    print(f"Global vlim: ({global_vmin:.2f}, {global_vmax:.2f}) dB/Hz")
    
    return raw_dict, psd_dict, global_vmin, global_vmax


def plot_topomap_comparison(models, maser_model, dataloader, index, fs_hr, dataset_name, target_channel_names, multiplier=8):
    """Plot single topomap comparing MASER, BiMa, and DiBiMa for x8 SR."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, hr, pos, label = list(iter(dataloader))[index]
    
    with torch.no_grad():
        hr = hr[0].to(device)
        pos = pos[0].to(device)
        label = label[0].to(device)
        
        # Generate LR signal (x8 downsampling)
        lr_8 = get_lr_data_spatial(hr.unsqueeze(0), dataset_name=dataset_name, sr_ratio=multiplier).squeeze(0).to(device)
        
        print(f"Generating SR signals for topomap plotting...")
        print(f"HR shape: {hr.shape}, LR_/{multiplier} shape: {lr_8.shape}")
        
        # Generate BiMa SR
        bima_sr = models['bima'](lr_8.unsqueeze(0))[0]
        
        # Generate DiBiMa SR
        dibima_sr = models['dibima'].sample_from_lr(lr_8.unsqueeze(0), pos.unsqueeze(0), label.unsqueeze(0))[0]
        
        # Generate MASER SR (if available)
        if maser_model is not None:
            # Prepare MASER input format
            from utils import map_mmi_channels, case2_mmi
            maser_inputs = hr.unsqueeze(0).permute(0, 2, 1).unsqueeze(1)  # (1, 1, time, channels)
            channels = [map_mmi_channels[i] for i in case2_mmi['x4']]
            _, _, _, maser_sr = maser_model(maser_inputs, unmasked_list=channels, test_flag=True, return_sr_eeg=True)
            maser_sr = maser_sr[0]  # Remove batch dimension
        else:
            maser_sr = None
        
        # Prepare LR with zero padding for visualization
        #lr_8_padded = add_zero_channels(hr.unsqueeze(0), dataset_name=dataset_name, multiplier=8).squeeze(0)
    
    # Prepare signals dictionary
    signals = {
        f'LR_/{multiplier}': lr_8.cpu().detach().numpy(),
    }
    if maser_sr is not None:
        signals['MASER_x4'] = maser_sr.cpu().detach().numpy()

    signals[f'BiMa_x{multiplier}'] = bima_sr.cpu().detach().numpy()
    signals[f'DiBiMa_x{multiplier}'] = dibima_sr.cpu().detach().numpy()
    signals["HR"] = hr.cpu().detach().numpy()


    # Compute PSDs
    raw_dict, psd_dict, global_vmin, global_vmax = compute_topomap_psd(
        signals, fs_hr, dataset_name, pos, target_channel_names, multiplier=multiplier
    )
    
    # Plot single row of topomaps
    row_labels = list(signals.keys())
    nrows = len(row_labels)
    
    fig, axes = plt.subplots(1, nrows, figsize=(4*nrows, 4))
    if nrows == 1:
        axes = [axes]
    
    first_im = None
    
    for i, label in enumerate(row_labels):
        ax = axes[i]
        raw = raw_dict[label]
        psd_log = psd_dict[label]
        
        im, _ = mne.viz.plot_topomap(
            psd_log, 
            raw.info, 
            axes=ax, 
            contours=10,
            cmap='hot',
            show=False, 
            vlim=(global_vmin, global_vmax)
        )
        
        if first_im is None:
            first_im = im
        
        ax.set_title(label, fontsize=14, weight='bold')
    
    # Add horizontal colorbar at bottom - more compact
    fig.subplots_adjust(bottom=0.12, top=0.95, wspace=0.25, left=0.05, right=0.95)
    cbar_ax = fig.add_axes([0.3, 0.04, 0.4, 0.02])
    cbar = fig.colorbar(first_im, cax=cbar_ax, orientation='horizontal', label='Power (dB/Hz)')
    cbar.ax.tick_params(labelsize=12)
    
    
    plt.suptitle('EEG PSD Topomap Comparison', fontsize=16, weight='bold', y=1.05)
    #plt.savefig(f'imgs/topomap_comparison_x{multiplier}_index{index}_{dataset_name}.png', dpi=600, bbox_inches='tight')
    plt.show()
    return fig

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = './eeg_data'
    dataset_name = 'mmi'
    batch_size = 32
    demo = False  
    num_channels = 64 if dataset_name == 'mmi' else 62
    seed = 2 
    seconds = 2
    multiplier = 8
    
    print("Downloading test data...")
    patients = list(range(1, 110)) if dataset_name == 'mmi' else None
    test_size = 0.2
    _, test_patients = train_test_split(patients, test_size=test_size, random_state=seed)
    data_path = os.path.join(data_folder, dataset_name)
    dataset_test = EEGDataset(
        subject_ids=test_patients, 
        data_folder=data_path, 
        dataset_name=dataset_name, 
        verbose=False, 
        demo=demo, 
        num_channels=num_channels,
        seconds=seconds
    )
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    print("Test datasets loaded successfully.")
    
    target_channel_names = np.array(dataset_test.channel_names)
    fs_hr = 160 if dataset_name == 'mmi' else 200
    
    # Load models
    print("Loading BiMa model...")
    bima_model = load_bima_model(device, dataset_name=dataset_name, multiplier=multiplier)
    bima_model.eval()
    
    print("Loading DiBiMa model...")
    dibima_model = load_dibima_model(device, dataset_name=dataset_name, multiplier=multiplier)
    dibima_model.eval()
    
    # Load MASER model (for x4, adjust config if available)
    print("Loading MASER model...")
    cwd = os.getcwd()
    sep = os.sep
    maser_project_path = cwd + sep + "maser"
    
    # Check if MASER x4 config exists, otherwise set to None
    try:
        maser_path = maser_project_path + sep + 'ckpt' + sep + 'last.ckpt'
        config_path = maser_project_path + sep + 'config' + sep + 'case2-4x-MM-state8-Mdep2.yml'
        maser_model, _ = load_maser_model(config_path, maser_path)
        maser_model.to(device)
        maser_model.eval()
        print("MASER model loaded successfully.")
    except:
        print("MASER x4 model not found, will skip MASER in comparison.")
        maser_model = None
    
    models = {
        'bima': bima_model,
        'dibima': dibima_model
    }
    
    print("Plotting topographic map comparison for a sample from the test set...")
    index = None
    current_min_loss = float('inf')
    for i, (lr, hr, pos, label) in enumerate(dataloader_test):
        lr = get_lr_data_spatial(hr, dataset_name=dataset_name, sr_ratio=8).to(device)
        pred = dibima_model.sample_from_lr(lr, pos=pos, label=label)
        loss = torch.nn.functional.mse_loss(hr, pred).item()
        if loss <= current_min_loss:
            current_min_loss = loss 
            index = i

    plot_topomap_comparison(
        models, 
        maser_model,
        dataloader_test, 
        index=index, 
        fs_hr=fs_hr,
        dataset_name=dataset_name,
        target_channel_names=target_channel_names
    )


if __name__ == "__main__":
    main()
