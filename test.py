import torch
import os
import pandas as pd
from tabulate import tabulate
from torch.utils.data import DataLoader
from models import DCAE_SR_nn
from utils import plot_umap_latent_space, map_runs
from metrics import evaluate_model, plot_metric_boxplots
import mne
import gc

torch.cuda.empty_cache()
gc.collect()

mne.set_log_level('ERROR')

# Configuration matching training script
project_path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fs_hr = 160
seconds = 10
hr_window_length = fs_hr * seconds
batch_size = 16
models_path = project_path + os.sep + "model_weights"
preprocessed_data_path = project_path + os.sep + "eeg_data" + os.sep + "preprocessed"
data_path = project_path + os.sep + "eeg_data"

# Test configuration - ALL POSSIBLE CONFIGURATIONS
sr_types = ["temporal"]  # "temporal", "spatial" 
temporal_factors = [4]   # [2, 4, 8]
spatial_channels = [8, 16, 32]  # Spatial SR input channels
fold = 0  # fold 0-0 (fold=1 in files)

# SUPPORTS ALL MODELS FROM TRAINING
AVAILABLE_MODELS = {
    'base': {'residual_global': False, 'residual_internal': False, 'use_subpixel': False},
    'with_subpixel': {'residual_global': False, 'residual_internal': False, 'use_subpixel': True},
    'with_subpixel_internal_residual': {'residual_global': False, 'residual_internal': True, 'use_subpixel': True},
    'with_subpixel_and_both_residuals': {'residual_global': True, 'residual_internal': True, 'use_subpixel': True}
}

models_to_test = ['with_subpixel_and_both_residuals']  # Add more as needed
imgs_path = project_path + os.sep + "imgs"
os.makedirs(imgs_path, exist_ok=True)

umap_path = imgs_path + os.sep + "umap_latent_spaces"
os.makedirs(umap_path, exist_ok=True)

def load_test_dataset(sr_type, param, fold=0):
    """Load preprocessed test dataset for given sr_type and param"""
    
    if sr_type == "temporal":
        str_param = f"x{param}"
        test_path = os.path.join(preprocessed_data_path, f'test_sr_{sr_type}_{str_param}_{fold+1}.pt')
    else:
        str_param = f"{param}to64chs"
        test_path = os.path.join(preprocessed_data_path, f'test_sr_{sr_type}_{str_param}_{fold+1}.pt')
    
    if os.path.exists(test_path):
        print(f"✅ Loading: {test_path}")
        dataset_test = torch.load(test_path, weights_only=False)
    else:
        print(f"⚠️  Preprocessed data not found: {test_path}")
        print("Skipping this configuration...")
        return None
    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    print(f"   Dataset size: {len(dataset_test)}")
    return dataloader_test, dataset_test

def create_model(num_channels_lr, num_channels_hr, lr_len, hr_len, model_config, sr_type):
    """Create model with exact training configuration"""
    return DCAE_SR_nn(
        num_channels=num_channels_lr,  # Input channels
        lr_len=lr_len,
        hr_len=hr_len,
        residual_global=model_config['residual_global'],
        residual_internal=model_config['residual_internal'],
        use_subpixel=model_config['use_subpixel'],
        sr_type=sr_type
    )

def test_single_model(model_name, sr_type, param, fold, dataloader_test):
    """Test one model variant"""
    
    num_channels_hr = 64
    if sr_type == "temporal":
        fs_lr = int(fs_hr / param)
        lr_window_length = fs_lr * seconds
        num_channels_lr = num_channels_hr
        model_path = os.path.join(models_path, f'fold_{fold+1}', 
                                f'dcae_sr_eeg_{model_name}_x{param}_{fold+1}.pth')
    else:  # spatial
        lr_window_length = hr_window_length
        num_channels_lr = param
        model_path = os.path.join(models_path, f'fold_{fold+1}', 
                                f'dcae_sr_eeg_{model_name}_{param}to64chs_{fold+1}.pth')
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {os.path.basename(model_path)}")
        return None, None, None
    
    model_config = AVAILABLE_MODELS[model_name]
    model = create_model(num_channels_lr, num_channels_hr, lr_window_length, 
                        hr_window_length, model_config, sr_type).to(device)
    
    print(f"✅ Loading: {os.path.basename(model_path)}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        results, results_raw = evaluate_model(model.to(device), dataloader_test)
    
    return results, results_raw, model

def comprehensive_test():
    """Run ALL possible configurations: sr_types × params × models"""
    
    all_results = {}
    
    for sr_type in sr_types:
        print(f"\n{'='*80}")
        print(f"🚀 TESTING {sr_type.upper()} SUPER-RESOLUTION")
        print(f"{'='*80}")
        
        # Get parameters for this sr_type
        if sr_type == "temporal":
            params = temporal_factors
            param_label = "Downsample Factor"
        else:
            params = spatial_channels
            param_label = "Input Channels"
        
        for param in params:
            print(f"\n--- {param_label}: {param} ---")
            
            # Load test dataset once per param
            result = load_test_dataset(sr_type, param, fold)
            if result is None:
                continue
            dataloader_test, dataset_test = result
            
            param_results = {}
            param_results_raw = {}
            
            # Test all models for this param
            for model_name in models_to_test:
                print(f"\n  📊 Testing {model_name}...")
                results, results_raw, model = test_single_model(
                    model_name, sr_type, param, fold, dataloader_test)
                
                if results is not None:
                    param_results[model_name] = results
                    param_results_raw[model_name] = results_raw
                    
                    # UMAP Visualization of Latent Space
                    print(f"  🗺️  Generating UMAP latent space for {model_name}...")
                    with torch.no_grad():
                        model = model.to(device)
                        model.eval()
                        filepath = os.path.join(umap_path, 
                                                f'umap_{sr_type}_{"x"+str(param) if sr_type=="temporal" else str(param)+"to64chs"}_{model_name}_fold{fold+1}.png')
                        plot_umap_latent_space(model, dataset_test.data_lr, 
                                            labels=dataset_test.labels, 
                                            save_path=filepath,
                                            map_labels=map_runs)
                        print(f"     Saved: {os.path.basename(filepath)}")
            
            # Save results for this param
            if param_results:
                df = pd.DataFrame(param_results).T
                str_param = f"x{param}" if sr_type == "temporal" else f"{param}to64chs"
                output_path = os.path.join(project_path, f'test_results_{sr_type}_{str_param}_{fold+1}.csv')
                df.to_csv(output_path)
                print(f"\n📊 Results saved: {output_path}")
                print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True))
                
                # Plot metrics
                plot_metric_boxplots(param_results_raw)
                
                all_results[f"{sr_type}_{str_param}"] = param_results
    
    return all_results

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    
    print("🎯 COMPREHENSIVE EEG SUPER-RESOLUTION TESTING")
    print(f"Models: {models_to_test}")
    print(f"Temporal factors: {temporal_factors}")
    print(f"Spatial channels: {spatial_channels}")
    print(f"Fold: {fold+1}")
    print("-" * 80)
    
    all_results = comprehensive_test()
    
    print("\n🎉 COMPREHENSIVE TESTING COMPLETED!")
    print(f"\n📁 Generated:")
    print(f"   • {len(all_results)} results CSV files")
    print(f"   • UMAP plots for each model+param combination")
    print(f"   • Metrics boxplots for each configuration")
    
    print(f"\nAvailable models: {list(AVAILABLE_MODELS.keys())}")
