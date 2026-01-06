import torch    
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from models import DCAE_SR_Diff
import torch.nn.functional as F

"""1D Signal Quality Metrics"""
#Metrics works in general, seems some problems with inputs with very low values

def psnr2d(original, reconstructed):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for batched 2D signals [B, C, L].
    Returns mean PSNR across batch and channels.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Input signals must have the same shape: {original.shape} vs {reconstructed.shape}")
    
    mse = torch.mean((original - reconstructed) ** 2)  # Global MSE over [B,C,L]
    max_pixel_value = 1.0
    if mse == 0:
        return float('inf')
    psnr_value = 10 * torch.log10((max_pixel_value ** 2) / mse)
    return psnr_value.item()


def snr2d(original, reconstructed):
    """
    Compute channel-level SNR for batched 2D signals [B, C, L].
    Averages across batch, then per-channel SNRs, then mean.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shapes mismatch: {original.shape} vs {reconstructed.shape}")
    
    # Compute per-batch-per-channel: mean across L (dim=2)
    signal_power = torch.mean(original ** 2, dim=2)  # [B, C]
    noise_power = torch.mean((reconstructed - original) ** 2, dim=2)  # [B, C]
    
    # Avoid div-by-zero
    noise_power = torch.clamp(noise_power, min=1e-8)
    
    channel_snrs = 10 * torch.log10(signal_power / noise_power)  # [B, C]
    mean_snr = torch.mean(channel_snrs)  # scalar over batch and channels
    
    return mean_snr.item()


def ssim2d(original, reconstructed):
    """
    Global SSIM per channel per batch for [B, C, L].
    Uses global mean/var across L (dim=2), no sliding window.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shapes mismatch: {original.shape} vs {reconstructed.shape}")
    
    B, C, L = original.shape
    mu1 = torch.mean(original, dim=2)  # [B, C]
    mu2 = torch.mean(reconstructed, dim=2)  # [B, C]
    sigma1_sq = torch.var(original, dim=2)  # [B, C]
    sigma2_sq = torch.var(reconstructed, dim=2)  # [B, C]
    # Covariance: mean of centered products across L
    centered1 = original - mu1.unsqueeze(2)
    centered2 = reconstructed - mu2.unsqueeze(2)
    sigma12 = torch.mean(centered1 * centered2, dim=2)  # [B, C]
    
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    channel_ssim = num / den  # [B, C]
    return torch.mean(channel_ssim).item()  # Mean over batch and channels


def pcc2d(x, y):
    """
    Pearson Correlation Coefficient (PCC) for batched [B, C, L].
    Computes per-channel per-batch, then averages.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shapes mismatch: {x.shape} vs {y.shape}")
    
    x_np = x.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    r2s = []
    for b in range(x_np.shape[0]):
        for c in range(x_np.shape[1]):
            r, _ = pearsonr(x_np[b, c], y_np[b, c])
            r2s.append(r)
    return np.mean(r2s)

def pcc2d_torch(x, y):
    """
    Vectorized PCC over [B,C,L] → scalar mean(r_bc).
    Stays on GPU/CPU, no loops.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shapes mismatch: {x.shape} vs {y.shape}")
    
    # Center: [B,C,L]
    x_mu = x.mean(dim=-1, keepdim=True)
    y_mu = y.mean(dim=-1, keepdim=True)
    x_c = x - x_mu
    y_c = y - y_mu
    
    # Cov / stds: [B,C]
    cov = (x_c * y_c).mean(dim=-1)
    std_x = x_c.pow(2).mean(dim=-1).sqrt()
    std_y = y_c.pow(2).mean(dim=-1).sqrt()
    
    # Clamp zero std
    std_x = torch.clamp(std_x, min=1e-8)
    std_y = torch.clamp(std_y, min=1e-8)
    
    pcc = cov / (std_x * std_y)  # [B,C]
    return pcc.mean().item()

def nmse2d_real(original, reconstructed):
    """
    NMSE for real-valued signals: ||reconstructed - original||^2 / ||original||^2
    Averaged over batch and channels.
    Lower is better (0=perfect).
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shapes mismatch: {original.shape} vs {reconstructed.shape}")
    
    mse = torch.mean((reconstructed - original) ** 2, dim=2)  # [B, C]
    power_orig = torch.mean(original ** 2, dim=2)  # [B, C]
    power_orig = torch.clamp(power_orig, min=1e-8)
    nmse = mse / power_orig
    return torch.mean(nmse).item()

def nmse2d(original, reconstructed):
    """
    Standard NMSE: MSE / var(original) per-channel per-batch.
    Lower is better (0=perfect).
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shapes mismatch: {original.shape} vs {reconstructed.shape}")
    
    mse = torch.mean((reconstructed - original) ** 2, dim=2)  # [B, C]
    var_orig = torch.var(original, dim=2)  # [B, C] - signal power
    var_orig = torch.clamp(var_orig, min=1e-8)
    nmse = mse / var_orig
    return torch.mean(nmse).item()

# Evaluation
def evaluate_model(model, dataloader, n_timesteps=None):
    
    model.eval()

    if getattr(model, 'parameters', None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = next(model.parameters()).device

    print(f"Using device: {device}")

    model = model.to(device)        
    mse_crit = nn.MSELoss()
    mses, rmses, psnrs, ssims, inf_times = [], [], [], [], []
    nmses, ppcs, snrs = [], [], []

    print("Evaluating model performances...")
    with torch.no_grad():
        for lr_input, hr_target in tqdm(dataloader, desc="Test", unit="seg", leave=False):
            lr_input = lr_input.to(device)
            hr_target = hr_target.to(device)
            #print(lr_input.shape, hr_target.shape)
            
            start = time.time()
            if model.use_diffusion:
                #print("Using diffusion model for inference...")
                model_diff = DCAE_SR_Diff(model).to(device)
                sr_recon = model_diff.sample(lr_input, num_inference_steps=n_timesteps)
            else:
                #print("Using standard model for inference...")
                sr_recon = model(lr_input)
            inf_time = time.time() - start
            inf_times.append(inf_time)
  

            mse = mse_crit(sr_recon, hr_target).item()
            mses.append(mse)
            rmse = np.sqrt(mse)
            rmses.append(rmse)

            snr = snr2d(sr_recon, hr_target)
            #print(f'SNR: {snr:.4f}')
            ssim = ssim2d(sr_recon, hr_target)
            #print(f'SSIM: {ssim:.4f}')
            nmse = nmse2d_real(sr_recon, hr_target)
            #print(f'NMSE: {nmse:.6f}')
            pcc = pcc2d_torch(sr_recon, hr_target)
            #print(f'PCC: {pcc:.4f}')
            psnr = psnr2d(sr_recon, hr_target)
            #print(f'PSNR: {psnr:.4f}')
                
            ssims.append(ssim)
            nmses.append(nmse)
            ppcs.append(pcc)
            snrs.append(snr)
            psnrs.append(psnr)  
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = params / 1000000  # in milioni
    params = round(params, 2)

    mse = np.mean(mses)
    rmse = np.mean(rmses)
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    nmse = np.mean(nmses)
    pcc = np.mean(ppcs)
    snr = np.mean(snrs)
    inf_time = np.mean(inf_times)

    mse_std = np.std(mses)
    rmse_std = np.std(rmses)
    psnr_std = np.std(psnrs)
    ssim_std = np.std(ssims)
    nmse_std = np.std(nmses)
    pcc_std = np.std(ppcs)
    snr_std = np.std(snrs)
    inf_time_std = np.std(inf_times)      
    
    dict = {
        'MSE': f"{mse:.6f} ± {mse_std:.3f}",
        'RMSE': f"{rmse:.6f} ± {rmse_std:.3f}",
        'PSNR': f"{psnr:.4f} ± {psnr_std:.3f}",
        'SSIM': f"{ssim:.4f} ± {ssim_std:.3f}",
        'NMSE': f"{nmse:.6f} ± {nmse_std:.3f}",
        'PCC': f"{pcc:.4f} ± {pcc_std:.3f}",
        'SNR': f"{snr:.4f} ± {snr_std:.3f}",
        'Avg Inference Time': f"{inf_time:.6f} ± {inf_time_std:.3f}",
        'Parameters': f"{params}M"
    }  
    dict_raw = {
        'MSE': mses,
        'RMSE': rmses,
        'PSNR': psnrs,
        'SSIM': ssims,
        'NMSE': nmses,
        'PCC': ppcs,
        'SNR': snrs
    }
    return dict, dict_raw

import matplotlib.pyplot as plt

def plot_metric_boxplots(metrics_dict, figsize=(12, 6), name = 'metrics_boxplots', project_path='.'):
    """
    Plot boxplots for evaluation metrics.

    Args:
        metrics_dict: dict of metric names (keys) and their raw value lists (values).
                      Each value should be a list (not a mean±std string).
        figsize: size of the matplotlib figure.
    """

    data = {}

    # Only include metrics that are lists/numpy arrays (not strings)
    for model_name, values in metrics_dict.items():
        if model_name not in data:
            data[model_name] = {}

        for metric_name, metric_values in values.items():
            if metric_name not in data[model_name]:
                data[model_name][metric_name] = []
            data[model_name][metric_name] = metric_values

    for metric_name in data[list(data.keys())[0]].keys():
        model_names = list(data.keys())
        data_to_plot = [data[model][metric_name] for model in model_names]
        fig = plt.figure(figsize=figsize)
        plt.boxplot(data_to_plot, labels=model_names)
        plt.ylabel(metric_name)
        plt.title('Models Performance Metrics Boxplots')
        plt.xticks(rotation=45)
        #plt.tight_layout()
        fig.savefig(f'{project_path}/{name}_{metric_name}_boxplots.png')
        plt.close(fig)  #Close figure to free memory
        #plt.show()


if __name__ == "__main__":
    # Example usage
    import torch

    # Dummy data
    batch_size = 8
    num_channels = 64
    timepoints = 1600
    original = 0.0005*torch.rand(batch_size, num_channels, timepoints)  # [batch, channels, timepoints]
    reconstructed = original - 0.00001 * torch.randn(batch_size, num_channels, timepoints)  # Add slight noise

    plt.figure()
    plt.plot(original[0].mean(dim=0).cpu().numpy(), label='Original')
    plt.plot(reconstructed[0].mean(dim=0).cpu().numpy(), label='Reconstructed')
    plt.legend()
    plt.title('Original vs Reconstructed Signal (Channel 0)')
    plt.show()

    print("PSNR:", psnr2d(original, reconstructed))
    print("SNR:", snr2d(original, reconstructed))
    print("SSIM:", ssim2d(original, reconstructed))
    print("NMSE:", nmse2d(original, reconstructed))
    print("NMSE (real):", nmse2d_real(original, reconstructed))
    print("PCC:", pcc2d(original, reconstructed))
    print("PCC (torch):", pcc2d_torch(original, reconstructed))
