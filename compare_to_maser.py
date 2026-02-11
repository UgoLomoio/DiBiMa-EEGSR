import os 
import sys
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
from utils import unmask_channels
from models import DiBiMa_nn, DiBiMa_Diff
from test import *
from diffusion_conditioning_ablations import train_scheduler, val_scheduler, loss_fn, learning_rate, prediction_type, debug, epochs, models_path
from torch.utils.flop_counter import FlopCounterMode

def get_flops(model, inputs, with_backward: bool = False) -> int:
    is_train = model.training
    model.eval()
    
    flop_counter = FlopCounterMode(display=False)
    
    with flop_counter:
        if model.__class__.__name__ == "DiBiMa_Diff":
            # For DiBiMa_Diff, we need to call the sample method for FLOPs calculation
            output = model.test_step(inputs, 0)#sample(*inputs)
        else:
            output = model(*inputs)
        if with_backward:
            output.sum().backward()  # Includes backward pass FLOPs
    
    total_flops = flop_counter.get_total_flops()
    
    if is_train:
        model.train()
    
    return total_flops

cwd = os.getcwd()
sep = os.sep
maser_project_path = cwd + sep + "maser"
data_path = maser_project_path + sep + "data" + sep + "test_data.dat"
multiplier = 8
sys.path.append(maser_project_path)

def load_maser_model(config_path,model_path):
    """
    Load a pre-trained MASER model from the specified path.
    Args:
        config_path (str): Path to the model configuration file.
        model_path (str): Path to the model file.
    Returns:
        model: Loaded MASER model.
    """
    from inference import load_model

    model = load_model(config_path, model_path)
    return model

def mask_channels(eeg_data, dataset_name="mmi", multiplier=multiplier):
    """
    Mask a portion of EEG channels in the data.
    Args:
        eeg_data (torch.Tensor): EEG data of shape (channels, time).
        dataset_name (str): Name of the dataset.
        multiplier (int): Downsampling multiplier.
    Returns:
        masked_data: EEG data with masked channels.
    """
    channels_to_mask = unmask_channels[dataset_name][f'x{multiplier}'].copy()
    masked_data = eeg_data.clone()
    masked_data[:, channels_to_mask, :] = 0
    return masked_data

def load_dibima_model(device):
    """
    Load a pre-trained DiBiMa diffusion model from the specified path.
    Args:
        device (torch.device): Device to load the model onto.
    Returns:
        model: Loaded DiBiMa model.
    """
    dataset_name = "mmi"
    sr_type = "spatial"
    target_channels = 64
    multiplier = 8
    input_channels = unmask_channels[dataset_name][f'x{multiplier}'].__len__()  # Example: 4x SR
    seconds = 2
    dibima_params = {
        "use_electrode_embedding": True,
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
                        fs_lr=160,
                        fs_hr=160,
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
    model = DiBiMa_Diff(model_nn,
                        train_scheduler=train_scheduler,
                        val_scheduler=val_scheduler,
                        criterion=loss_fn,
                        learning_rate=learning_rate,
                        predict_type=prediction_type,  # "epsilon" or "sample"
                        debug=debug,
                        epochs=epochs,
                        plot=False).to(device)
            
    model.model = load_model_weights(model.model, os.path.join(models_path, f'DiBiMa_eeg_{input_channels}to{target_channels}chs_{sr_type}_{dataset_name}.pth'))
    return model


def load_bima_model(device):
    """
    Load a pre-trained BiMa Regression model from the specified path.
    Args:
        device (torch.device): Device to load the model onto.
    Returns:
        model: Loaded BiMa model.
    """
    dataset_name = "mmi"
    sr_type = "spatial"
    target_channels = 64
    multiplier = 8
    input_channels = unmask_channels[dataset_name][f'x{multiplier}'].__len__()  # Example: 4x SR
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
                        fs_lr=160,
                        fs_hr=160,
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
    model = DiBiMa(model_nn,
                        learning_rate=learning_rate,
                        loss_fn=loss_fn,
                        debug=debug,
                        epochs=epochs,  
                        plot=False).to(device)
            
    model.model = load_model_weights(model.model, os.path.join(models_path, f'BiMa_eeg_{input_channels}to{target_channels}chs_{sr_type}_{dataset_name}.pth'))

    return model

if __name__ == "__main__":

    dataset_name = "mmi"
    target_channels = 64 if dataset_name == "mmi" else 62
    subject_ids = list(range(1, dict_n_patients[dataset_name] + 1))
    runs = map_runs_dataset[dataset_name]
    project_path = os.getcwd()
    batch_size = 64
    data_path = project_path + os.sep + "eeg_data"
    data_folder = data_path + os.sep + dataset_name + os.sep

    dataset = EEGDataset(subject_ids=subject_ids, data_folder=data_folder, normalize=False, dataset_name=dataset_name, verbose=False, demo=True, num_channels=target_channels, seconds=seconds)
    dataloader_train, dataloader_val, dataloader_test = prepare_dataloaders_windows(
            dataset_name, dataset, seconds=seconds, batch_size=batch_size, return_test=True
    )

    print(f"Loaded {len(dataloader_test.dataset)} samples from test data.")
    
    num_channels = unmask_channels[dataset_name][f'x{multiplier}'].__len__()
    dataloader_test.dataset.num_channels = num_channels
    dataloader_test.dataset.multiplier = multiplier
    dataloader_test.dataset.sr_type = "spatial"
    
    idx = random.randint(0, len(dataloader_test.dataset)-1)
    print(f"Testing sample index: {idx}")
    lr, hr, pos, label = dataloader_test.dataset[idx]
    print(f"HR shape: {hr.shape}, LR shape: {lr.shape}, Pos shape: {pos.shape}, Label: {label}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr = hr.unsqueeze(0).to(device)  # (1, 64, 1600)
    lr_bima = lr.unsqueeze(0).to(device)  # (1, 16, 1600)
    pos = pos.unsqueeze(0).to(device)  # (1, 16, 3)
    label = label.unsqueeze(0).to(device)  # (1, )

    #lr_bima = get_lr_data_spatial(hr.clone(), dataset_name=dataset_name, sr_ratio=8)

    print("Generating MASER outputs...")
    maser_path = maser_project_path + sep + 'ckpt' + sep + 'last.ckpt'
    config_path = maser_project_path + sep + 'config' + sep + 'case2-4x-MM-state8-Mdep2.yml'
    maser, args = load_maser_model(config_path, maser_path)
    num_params = sum(p.numel() for p in maser.parameters())
    print(f"Loaded MASER model with {num_params:,} parameters.")
    print(args)
    maser.to(device)
    maser_inputs = hr.permute(0, 2, 1).unsqueeze(1) # (1, 1600, 16) 
    channels = [map_mmi_channels[i] for i in case2_mmi['x4']]
    inputs = [maser_inputs, channels]
    start = time.time()
    flops = get_flops(maser, inputs, with_backward=False)
    end = time.time()
    print(f"MASER FLOPs (inference): {flops:,}")
    print(f"MASER inference time: {end - start:.6f} seconds")
    maser.eval()

    with torch.no_grad():
        print(maser_inputs.shape)
        loss, nmse, ppc, maser_out = maser(maser_inputs, unmasked_list=channels, test_flag=True, return_sr_eeg=True)
 
    hr = hr.to(device)
    lr_bima = lr_bima.to(device)
    pos = pos.to(device)
    label = label.to(device)

    print("Generating BiMa outputs...")
    print(lr_bima.shape, pos.shape, label.shape)
    bima = load_bima_model(device)
    num_params = sum(p.numel() for p in bima.parameters())
    print(f"Loaded BiMa model with {num_params:,} parameters.")
    flops = get_flops(bima, [lr_bima], with_backward=False)
    print(f"BiMa FLOPs (inference): {flops:,}")
    bima.eval()
    with torch.no_grad():
        bima_out = bima(lr_bima)  # (1, 64, 1600)  # Adjust pos and label as needed
    bima_out = bima_out[0].cpu().numpy()  # (64, 1600)
    
    print("Generating DiBiMa outputs...")
    print(lr_bima.shape, hr.shape, pos.shape, label.shape)
    dibima = load_dibima_model(device)
    num_params = sum(p.numel() for p in dibima.parameters())
    print(f"Loaded DiBiMa model with {num_params:,} parameters.")
    dibima_inputs = [lr_bima, hr, pos, label]
    flops = get_flops(dibima, dibima_inputs, with_backward=False)
    print(f"DiBiMa FLOPs (inference): {flops:,}")
    dibima.eval()
    with torch.no_grad():
        dibima_out = dibima.sample(lr_bima, pos = pos.squeeze(0), label=label)  # (1, 64, 1600)  # Adjust pos and label as needed
        dibima_out_lr_up = dibima.sample_from_lr(lr_bima, pos = pos.squeeze(0), label=label)  # (1, 64, 1600)
    dibima_out = dibima_out[0].cpu().numpy()  # (64, 1600)
    dibima_out_lr_up =dibima_out_lr_up[0].cpu().detach().numpy()  # (1, 64, 1600)
    
    lr_bima = add_zero_channels(hr, dataset_name=dataset_name, multiplier=multiplier)
    lr_bima = lr_bima[0].cpu().detach().numpy().mean(axis=0)
    
    hr = hr[0].cpu().detach().numpy().mean(axis=0)
    maser_out = maser_out[0].cpu().detach().numpy().mean(axis=0)
    dibima_out = dibima_out.mean(axis=0)
    bima_out = bima_out.mean(axis=0)
    dibima_out_lr_up = dibima_out_lr_up.mean(axis=0)

    plt.figure()
    plt.plot(lr_bima, color = "red", linestyle='--', linewidth=0.5, label='Masked EEG')
    plt.plot(hr, color = "green", linewidth=2, label='Original EEG')
    plt.plot(maser_out, color = "orange", linewidth=1, label=f'MASER x4 - mse {mse_loss(torch.tensor(maser_out), torch.tensor(hr)).item():.6f}')
    
    plt.plot(bima_out, color = "cyan", linewidth=1, label=f'BiMa x8 - mse {mse_loss(torch.tensor(bima_out), torch.tensor(hr)).item():.6f}')
    plt.plot(dibima_out, color = "purple", linewidth=1, label=f'DiBiMa x8 - from pure Noise - mse {mse_loss(torch.tensor(dibima_out), torch.tensor(hr)).item():.6f}')
    plt.plot(dibima_out_lr_up, color = "blue", linewidth=1, label=f'DiBiMa x8 - from LR Upsampled - mse {mse_loss(torch.tensor(dibima_out_lr_up), torch.tensor(hr)).item():.6f}')
    
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Sample {idx} EEG Signal")
    plt.tight_layout()
    plt.savefig(f"compare_models_sample.png", dpi=600)
    plt.show() 