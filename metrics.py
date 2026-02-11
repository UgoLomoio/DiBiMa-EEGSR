import torch    
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
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
    
    if original.ndim == 1 and reconstructed.ndim == 1: #signal are flattened       

        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((reconstructed - original) ** 2)
        noise_power = torch.clamp(noise_power, min=1e-8)
        mean_snr = 10 * torch.log10(signal_power / noise_power)
        return mean_snr.item()
    
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
    
    if original.ndim == 1 and reconstructed.ndim == 1:  #signal are flattened
        #compute ssim 1d
        mu1 = torch.mean(original)
        mu2 = torch.mean(reconstructed)
        sigma1_sq = torch.var(original)
        sigma2_sq = torch.var(reconstructed)
        centered1 = original - mu1
        centered2 = reconstructed - mu2
        sigma12 = torch.mean(centered1 * centered2) 
        C1, C2 = 0.01**2, 0.03**2
        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = num / den
        return ssim.cpu().item()

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
    
    if x.ndim == 1 and y.ndim == 1:  #signal are flattened
        r, _ = pearsonr(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        return r
    
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
    
    if x.ndim == 1 and y.ndim == 1:  #signal are flattened
        x_mu = x.mean()
        y_mu = y.mean()
        x_c = x - x_mu
        y_c = y - y_mu
        cov = (x_c * y_c).mean()
        std_x = x_c.pow(2).mean().sqrt()
        std_y = y_c.pow(2).mean().sqrt()
        std_x = torch.clamp(std_x, min=1e-8)
        std_y = torch.clamp(std_y, min=1e-8)
        pcc = cov / (std_x * std_y)
        return pcc.item()

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

def nmse2d(original, reconstructed):
    """
    Standard NMSE: MSE / var(original) per-channel per-batch.
    Lower is better (0=perfect).
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shapes mismatch: {original.shape} vs {reconstructed.shape}")
    
    if original.ndim == 1 and reconstructed.ndim == 1:  #signal are flattened
        mse = torch.mean((reconstructed - original) ** 2)
        var_orig = torch.var(original)
        var_orig = torch.clamp(var_orig, min=1e-8)
        nmse = mse / var_orig
        return nmse.item()

    mse = torch.mean((reconstructed - original) ** 2, dim=2)  # [B, C]
    var_orig = torch.var(original, dim=2)  # [B, C] - signal power
    var_orig = torch.clamp(var_orig, min=1e-8)
    nmse = mse / var_orig
    return torch.mean(nmse).item()

# Evaluation
def evaluate_model(model, dataloader, sample_type="noise", evaluate_mean=False, flatten = True):

    #evaluate_mean: if True, evaluate the mean signal as baseline
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
        for batch_idx, (lr_input, hr_target, pos, label) in enumerate(tqdm(dataloader, desc="Test", unit="seg", leave=False)):
            lr_input = lr_input.to(device)
            hr_target = hr_target.to(device)
            pos = pos.to(device)
            label = label.to(device)

            if model.model.use_electrode_embedding:
                pos = pos.float()
            else:
                pos = None
            #print(lr_input.shape, hr_target.shape)
            
            start = time.time()
            if model.model.use_diffusion:
                
                #print("Using diffusion model for inference...")
                if sample_type == "noise":
                    sr_recon = model.sample(lr_input, pos=pos, label=label)
                else:
                    #if model.model.use_lr_conditioning:
                        #print("Using LR conditioning for inference...")
                    sr_recon = model.sample_from_lr(lr_input, pos=pos, label=label)
                    #else:
                        #print("Not using LR conditioning for inference...")
                    #    sr_recon = model.sample(lr_input, pos=pos, label=label)
                
            else:
                #print("Using standard model for inference...")
                sr_recon = model(lr_input)
            
            inf_time = time.time() - start
            inf_times.append(inf_time)
  
            if evaluate_mean:
                # 1-channel mean signal as baseline
                sr_recon = sr_recon.mean(dim=1, keepdim=True)
                hr_target = hr_target.mean(dim=1, keepdim=True)
            else:
                if flatten:
                    # Flatten channels and batch
                    sr_recon = sr_recon.flatten()
                    hr_target = hr_target.flatten()

            mse = mse_crit(sr_recon, hr_target).item()
            mses.append(mse)
            rmse = np.sqrt(mse)
            rmses.append(rmse)

            snr = snr2d(hr_target, sr_recon)
            #print(f'SNR: {snr:.4f}')
            ssim = ssim2d(hr_target, sr_recon)
            #print(f'SSIM: {ssim:.4f}')
            nmse = nmse2d(hr_target, sr_recon)
            #print(f'NMSE: {nmse:.6f}')
            pcc = pcc2d_torch(hr_target, sr_recon)
            #print(f'PCC: {pcc:.4f}')
            psnr = psnr2d(hr_target, sr_recon)
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
        'SNR': snrs,
    }
    return dict, dict_raw

import matplotlib.pyplot as plt

def plot_metric_barplots(metrics_dict, name='metrics_barplots', project_path='.'):
    """
    Plot grouped bar plots with error bars for evaluation metrics.
    Creates two separate figures: one for PSNR/SNR and one for other metrics.
    
    Args:
        metrics_dict: dict of model names (keys) and their metrics dict (values).
                      Structure: {model_name: {metric_name: [values]}}
        name: base name for saved figures.
        project_path: path to save the figures.
    """
    
    import numpy as np
    
    model_names = list(metrics_dict.keys())
    all_metrics = list(next(iter(metrics_dict.values())).keys())
    
    # Separate metrics into two groups
    signal_metrics = [m for m in all_metrics if m.upper() in ['PSNR', 'SNR']]
    other_metrics = [m for m in all_metrics if m.upper() not in ['PSNR', 'SNR']]
    
    # Calculate means and stds
    means = {model: {metric: np.mean(metrics_dict[model][metric]) 
                     for metric in all_metrics} for model in model_names}
    stds = {model: {metric: np.std(metrics_dict[model][metric]) 
                    for metric in all_metrics} for model in model_names}
    
    # Function to create a bar plot for a group of metrics
    def create_barplot(metrics_list, title, filename_suffix, color_palette=None):
        if not metrics_list:
            return
        
        x = np.arange(len(metrics_list))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(max(8, len(metrics_list) * 2), 6))
        
        # Default color palette if not provided
        if color_palette is None:
            color_palette = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for idx, model in enumerate(model_names):
            offset = width * idx - (width * len(model_names) / 2 - width / 2)
            mean_values = [means[model][metric] for metric in metrics_list]
            std_values = [stds[model][metric] for metric in metrics_list]
            
            ax.bar(x + offset, mean_values, width, label=model, 
                   yerr=std_values, capsize=5, alpha=0.8, color=color_palette[idx])
        
        ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_list, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path = f'{project_path}/{name}_{filename_suffix}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to: {save_path}")
    
    # Create figures
    create_barplot(signal_metrics, 'Signal Quality Metrics (PSNR & SNR)', 
                   'signal_metrics', plt.cm.Reds(np.linspace(0.4, 0.8, len(model_names))))
    create_barplot(other_metrics, 'Other Performance Metrics', 
                   'other_metrics', plt.cm.Blues(np.linspace(0.4, 0.8, len(model_names))))


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
    print("PCC:", pcc2d(original, reconstructed))
    print("PCC (torch):", pcc2d_torch(original, reconstructed))
