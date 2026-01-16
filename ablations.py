import diffusers
import torch
from torch import nn
from models import *
from utils import *
import os 
from metrics import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from tabulate import tabulate
import mne
from pytorch_lightning import Trainer
import torchinfo 
from visualize import plot_mean_timeseries, plot_mean_timeseries_plotly
from utils import unmask_channels
import gc 
from torch.cuda import empty_cache
import sys

gc.collect()
empty_cache()
mne.set_log_level('ERROR') 

project_path = os.getcwd()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
epochs = 1
learning_rate = 0.001 #0.001
seed = 2
seconds = 10 #2 #9760 samples /160 Hz = 61 seconds
set_seed(seed)

nfolds = 1 # Number of folds for cross-validation

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = True # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")
    epochs = 2
    quick_load = False
    nfolds = 1

dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
n_patients = dict_n_patients["mmi"]  # Number of patients in the dataset   

models_path = project_path + os.sep + "model_weights"
data_path = project_path + os.sep + "eeg_data"
#preprocessed_data_path = data_path + os.sep + "preprocessed"
imgs_path = project_path + os.sep + "imgs"
ablations_path = project_path + os.sep + "ablation_results"
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)
if not os.path.exists(ablations_path):
    os.makedirs(ablations_path)

loss_fn = nn.MSELoss() #nn.MSELoss() #ReconstructionLoss()  # Loss function: callable function

sr_types = ["spatial", "temporal"]  # Types of super-resolution to evaluate

n_mamba_blocks = [1, 3, 5]  # Number of Mamba blocks in each Bi-Mamba layer
mamba_versions = [1, 2]  # Mamba version to use (1 or 2), 3 is not implemented yet but exists
mamba_dims = [32, 64, 128]  # Mamba dimension (number of channels in Mamba layer)
mamba_d_state = [8, 16, 32] # Mamba state dimension (number of channels in Mamba state)
n_mamba_layers = [1]  # Number of Bi-Mamba layers in the model
mamba_presence = [True, False]  # Whether to include Mamba layers or not
diffusion_presence = [True, False]  # Whether to include Diffusion or not

n_timesteps = 1000  # Number of diffusion timesteps

best_params = {
    "version": 2,
    "dim": 64, #64,  
    "d_state": 8, #16,  
    "n_mamba_blocks": 2, #5
    "n_mamba_layers": 1,
    "use_mamba": True,
    "use_diffusion": True,
    "use_electrode_embedding": True
}
best_params = None  # Set to None to perform full HPO

if best_params is not None:
    epochs = 10  # Increase epochs if using best params directly

def prepare_dataloaders(dataset_name, models_nn, train_patients, test_patients, data_folder, quick_load=True, ref_position=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {}

    num_channels = 64 if dataset_name == "mmi" else 62

    if not quick_load:    
        
        print("Downloading training data...")
        dataset_train = EEGDataset(subject_ids=train_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=num_channels)
        print("Downloading testing data...")
        dataset_test = EEGDataset(subject_ids=test_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=num_channels)

        if len(dataset_train) == 0:
            print("No data loaded. Check dataset creation process.")
            exit(1)
            
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        print("Train and Test datasets loaded successfully.")    

        ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
    
    for name, model in models_nn.items():
        
        model.ref_position = ref_position  # Set reference positions in the model

        if model.use_diffusion == False:
            models[name] = DiBiMa(model, learning_rate=learning_rate, loss_fn=loss_fn, debug=debug).to(device)
        else:
            prediction_type = "sample"  # "epsilon", "sample" or "v_prediction"
            diffusion_params = {
                    "num_train_timesteps": n_timesteps, #100,
                    "beta_start": 1e-5, 
                    "beta_end": 1e-2,        #1e-3                                                                               
                    "beta_schedule": "linear",
                    "prediction_type": prediction_type,
                    #"clip_sample": True,
                    #"clip_sample_range": 1,
            }
            models[name] = DiBiMa_Diff(model,
                                        loss_fn,
                                        diffusion_params=diffusion_params,
                                        learning_rate=learning_rate,
                                        scheduler_params=None,
                                        predict_type=prediction_type,  # "epsilon" or "sample"
                                        debug=debug,
                                        epochs=epochs,
                                        plot=True).to(device)

    if quick_load:
        return models 
    else:       
        return models, dataloader_train, dataloader_test

def sequential_mamba_hpo(dataset_name, sr_type, fold, train_patients, test_patients, data_folder,
                        fs_lr=20, seconds=10, fs_hr=160, target_channels=64, input_channels=16,
                        mamba_versions=[1, 2], mamba_dims=[64,128,256], mamba_d_state=[16,64], 
                        n_mamba_layers=[1], n_mamba_blocks=[1, 2, 3], metric_key="val_mse", multiplier = 8):
    """
    Sequential HPO: version → dim → d_state → n_layers.
    
    Returns: dict with best params and their results.
    """
    if sr_type == "spatial":
        fs_lr = fs_hr #temporal resolution is not changed in spatial SR
        sr_nam = f"{input_channels}to64chs_{sr_type}"
    elif sr_type == "temporal":
        input_channels = target_channels #all channels are used for temporal SR
        sr_nam = f"x{multiplier}_{sr_type}"

    def run_search(configs, fs_hr, fs_lr, target_channels, input_channels, ablation_type="", dataloader_train=None, dataloader_test=None, multiplier=8):
        
        clear_memory()
        models_nn = {}
        for cfg in configs:
            models_nn[cfg["name"]] = DiBiMa_nn(
                target_channels=target_channels,
                num_channels=input_channels,
                fs_lr=fs_lr,
                fs_hr=fs_hr,
                seconds=seconds,
                residual_global=False,
                residual_internal=True,
                use_subpixel=True,
                use_positional_encoding=False,
                use_electrode_embedding=False, #changing to True in the final ablation
                sr_type=sr_type,
                use_diffusion=cfg["use_diffusion"],
                use_mamba=cfg["use_mamba"],
                n_mamba_blocks=cfg["n_mamba_blocks"],
                n_mamba_layers=cfg["n_mamba_layers"],
                mamba_dim=cfg["mamba_dim"],
                mamba_d_state=cfg["mamba_d_state"],
                mamba_version=cfg["mamba_version"],
        )        
        if (dataloader_train is not None) and (dataloader_test is not None):
            quick_load = True
            ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
            models = prepare_dataloaders(
                dataset_name, models_nn, train_patients, test_patients,
                data_folder, quick_load=quick_load, ref_position=ref_position
            )
        else:
            quick_load = False
            models, dataloader_train, dataloader_test = prepare_dataloaders(
                dataset_name, models_nn, train_patients, test_patients,
                data_folder, quick_load=quick_load, ref_position=None
            )
        _, results = train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_test, fs_hr, fs_lr, target_channels, input_channels, multiplier=multiplier, ablation_type=ablation_type)

        del models, models_nn
        if quick_load:
            return results
        else:
            return results, dataloader_train, dataloader_test
    
    print(f"\n=== Sequential HPO for {sr_type} ===")
    
    # Stage 1: mamba_version
    configs = [{"name": f"{sr_nam}_mamba{v}", "use_mamba": True, "use_diffusion": False, "n_mamba_layers": 1, "n_mamba_blocks": 1, "mamba_dim": 64, "mamba_d_state": 16, "mamba_version": v}
               for v in mamba_versions]
    results, dataloader_train, dataloader_test = run_search(configs, ablation_type="mamba_version", fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, dataloader_train=None, dataloader_test=None)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_mamba_version_results_fold{fold+1}_{dataset_name}.csv'))
    best_version = min(configs, key=lambda c: results[c["name"]][metric_key])["mamba_version"]
    print(f"✓ Best version: {best_version}")
    
    
    # Stage 2: mamba_dim
    configs = [{"name": f"{sr_nam}_mamba{best_version}_d{d}", "use_mamba": True, "use_diffusion": False, "n_mamba_layers": 1, "n_mamba_blocks": 1, "mamba_dim": d, "mamba_d_state": 16, "mamba_version": best_version}
            for d in mamba_dims]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, ablation_type="mamba_dim", dataloader_train=dataloader_train, dataloader_test=dataloader_test)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_mambadim_fold{fold+1}_{dataset_name}.csv'))
    best_dim = min(configs, key=lambda c: results[c["name"]][metric_key])["mamba_dim"]
    print(f"✓ Best dim: {best_dim}")
        
    # Stage 3: mamba_d_state
    configs = [{"name": f"{sr_nam}_mamba{best_version}_d{best_dim}_ds{ds}", "use_mamba": True, "use_diffusion": False, "n_mamba_layers": 1, "n_mamba_blocks": 1, "mamba_dim": best_dim, "mamba_d_state": ds, "mamba_version": best_version}
                for ds in mamba_d_state]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels,  ablation_type="mamba_d_state", dataloader_train=dataloader_train, dataloader_test=dataloader_test)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_mambadstate_fold{fold+1}_{dataset_name}.csv'))
    best_d_state = min(configs, key=lambda c: results[c["name"]][metric_key])["mamba_d_state"]
    print(f"✓ Best d_state: {best_d_state}")
        
    # Stage 4: n_mamba_blocks
    configs = [{"name": f"{sr_nam}_mamba{best_version}_nb{nb}_d{best_dim}_ds{best_d_state}", "use_mamba": True, "use_diffusion": False,
                    "n_mamba_layers": 1, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version,
                    "n_mamba_blocks": nb}
                for nb in n_mamba_blocks]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, ablation_type="mamba_n_blocks", dataloader_train=dataloader_train, dataloader_test=dataloader_test)      
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_mambanblocks_fold{fold+1}_{dataset_name}.csv'))
    best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    best_namba_blocks = best_config["n_mamba_blocks"]
    print(f"✓ Best n_mamba_blocks: {best_namba_blocks}")   

    # Stage 5: n_mamba_layers
    configs = [{"name": f"{sr_nam}_mamba{best_version}_nb{best_namba_blocks}_d{best_dim}_ds{best_d_state}_nl{nl}", 
                    "use_mamba": True, "use_diffusion": False, "n_mamba_layers": nl, "n_mamba_blocks": best_namba_blocks, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version}
                for nl in n_mamba_layers]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, ablation_type="mamba_n_layers", dataloader_train=dataloader_train, dataloader_test=dataloader_test)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_mambanlayers_fold{fold+1}_{dataset_name}.csv'))
    best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    best_n_mamba_layers = best_config["n_mamba_layers"]
    print(f"✓ Best n_mamba_layers: {best_n_mamba_layers}")

    #Stage 6: Mamba Presence
    configs = [
        {"name": f"{sr_nam}_with_mamba{best_version}_nb{best_namba_blocks}_d{best_dim}_ds{best_d_state}_nl{best_config['n_mamba_layers']}", "use_mamba": True, "use_diffusion": False,
        "n_mamba_layers": best_config["n_mamba_layers"], "n_mamba_blocks": best_namba_blocks, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version},
        {"name": f"{sr_nam}_no_mamba", "use_mamba": False, "use_diffusion": False, 
        "n_mamba_layers": 0, "n_mamba_blocks": 0, "mamba_dim": 0, "mamba_d_state": 0, "mamba_version": 0}
    ]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, ablation_type="mamba_presence", dataloader_train=dataloader_train, dataloader_test=dataloader_test)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_mambapresence_fold{fold+1}_{dataset_name}.csv'))
    best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    best_mamba_presence = best_config["use_mamba"]
    print(f"✓ Best mamba presence: {best_mamba_presence}")

    #Stage 7: Diffusion Presence
    if best_mamba_presence:  # Mamba was used
        configs = [
            {"name": f"{sr_nam}_mamba{best_version}_nb{best_namba_blocks}_d{best_dim}_ds{best_d_state}_nl{best_config['n_mamba_layers']}_no_diffusion", "use_mamba": True, "use_diffusion": False,
            "n_mamba_layers": best_config["n_mamba_layers"], "n_mamba_blocks": best_namba_blocks, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version},
            {"name": f"{sr_nam}_mamba{best_version}_nb{best_namba_blocks}_d{best_dim}_ds{best_d_state}_nl{best_config['n_mamba_layers']}_with_diffusion", 
            "use_mamba": True, "use_diffusion": True, "n_mamba_layers": best_config["n_mamba_layers"], "n_mamba_blocks": best_namba_blocks, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version}
        ]
    else:
        configs = [
            {"name": f"{sr_nam}_no_mamba_no_diffusion", "use_mamba": False, "use_diffusion": False,
            "n_mamba_layers": 0, "n_mamba_blocks": 0, "mamba_dim": 0, "mamba_d_state": 0, "mamba_version": 0},
            {"name": f"{sr_nam}_no_mamba_with_diffusion", "use_mamba": False, "use_diffusion": True,
            "n_mamba_layers": 0, "n_mamba_blocks": 0, "mamba_dim": 0, "mamba_d_state": 0, "mamba_version": 0}
        ]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, ablation_type="diffusion_presence", dataloader_train=dataloader_train, dataloader_test=dataloader_test)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_diffusionpresence_fold{fold+1}_{dataset_name}.csv'))
    best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    best_diff_presence = best_config["use_diffusion"]
    print(f"✓ Best diffusion presence: {best_diff_presence}")
        
    results_diff_no_electrode = results[best_config["name"]].copy()  # Save results before electrode position embedding ablation

    # Stage 8: Electrode Position Embedding conditioning
    if best_diff_presence:  # Mamba was used
        configs = [
            {"name": f"{sr_nam}_mamba{best_version}_nb{best_namba_blocks}_d{best_dim}_ds{best_d_state}_nl{best_config['n_mamba_layers']}_with_posenc", "use_mamba": best_mamba_presence, "use_diffusion": True,
            "n_mamba_layers": best_config["n_mamba_layers"], "n_mamba_blocks": best_namba_blocks, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version, "use_electrode_embedding": True},
        ]
        results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, target_channels=target_channels, input_channels=input_channels, ablation_type="electrode_position_embedding", dataloader_train=dataloader_train, dataloader_test=dataloader_test)
        results["no_electrode_position_embedding"] = results_diff_no_electrode  # Add back the no electrode embedding results
        df = pd.DataFrame(results).T
        df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_electrode_position_embedding_fold{fold+1}_{dataset_name}.csv'))
        best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
        best_electrode_embedding = best_config.get("use_electrode_embedding", False)
        print(f"✓ Best electrode position embedding: {best_electrode_embedding}")
    else:
        best_electrode_embedding = False  # No electrode embedding if no diffusion

    best_results = results[best_config["name"]]
    
    if best_version == 0:
        print(f"\n🎯 FINAL BEST: No Mamba used, diffusion_presence={best_diff_presence}, use_electrode_embedding: {best_electrode_embedding}")
    else:
        print(f"\n🎯 FINAL BEST: version={best_version}, dim={best_dim}, d_state={best_d_state}, n_blocks={best_namba_blocks}, n_layers={best_n_mamba_layers}, diffusion_presence={best_diff_presence}, use_electrode_embedding={best_electrode_embedding}")
    print(f"   Best metric ({metric_key}): {best_results[metric_key]}")
    
    return {
        "best_config": best_config,
        "best_results": best_results,
        "all_results": results
    }

def train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_test, fs_hr, fs_lr, target_channels, input_channels, fold=0, multiplier = 8, plot_one_example=True, ablation_type="Final"):

    results = {}
    results_raw = {}

    if multiplier is None:
        multipliers = [2, 4, 8]
    else:
        multipliers = [multiplier]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for multiplier in multipliers:

        print(f" - Multiplier: {multiplier}, SR Type: {sr_type}")
        dataloader_train.dataset.multiplier = multiplier
        dataloader_test.dataset.multiplier = multiplier
        dataloader_train.dataset.sr_type = sr_type
        dataloader_test.dataset.sr_type = sr_type
        dataloader_train.dataset.fs_lr = fs_lr
        dataloader_test.dataset.fs_lr = fs_lr
        dataloader_train.dataset.num_channels = input_channels
        dataloader_test.dataset.num_channels = input_channels
    
        for name, model in models.items():

            print(f"\nTraining {name}...")

            model = model.to(device)
            summary = torchinfo.summary(model.model)
            with open(os.path.join(models_path, f'ablations', f'{name}_{dataset_name}_model_summary.txt'), 'w') as f:
                f.write(str(summary))
                #early_stopping_callback = EarlyStopping(monitor='avg_val_loss', patience=20, verbose=False, mode='min')

            trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1 if torch.cuda.is_available() else None, logger=False, enable_checkpointing=False)#, callbacks=[early_stopping_callback])
            trainer.fit(model, dataloader_train, val_dataloaders=dataloader_test)
                
            print(f"Evaluating {name}...")
            model_nn = model.model.to(device)
            if model_nn.use_diffusion:
                num_train_timesteps = model.scheduler.num_train_timesteps
                inference_timesteps = num_train_timesteps
                #if num_train_timesteps <= 100:
                #    inference_timesteps = num_train_timesteps 
                #else:
                #    inference_timesteps = num_train_timesteps // 10
            else:
                inference_timesteps = None

            results[name], results_raw[name] = evaluate_model(model, dataloader_test, n_timesteps=inference_timesteps, evaluate_mean=True)
            model_path = os.path.join(models_path, f'ablations', f'DiBiMa_eeg_{name}_{fold+1}.pth')
            torch.save(model.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Plot boxplots for raw results       
    print("\nPlotting metric boxplots...")
    plot_metric_boxplots(results_raw, name = f'ablation_{ablation_type}_{sr_type}_fold{fold+1}_{dataset_name}', project_path=imgs_path)

    # Create DataFrame
    df = pd.DataFrame(results).T  # Transpose to have models as rows
    # Print formatted table using tabulate
    print("\n=== Ablation Study Results ===")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True))
    df.to_csv(os.path.join(ablations_path, f'ablation_{ablation_type}_{sr_type}_{fold+1}_{dataset_name}.csv'))

    if plot_one_example:

        print("\nInference timeseries...")
        data = next(iter(dataloader_test))
        input, target, pos, _ = data
        input = input.to(device)
        target = target.to(device)
        pos = pos.to(device)

        timeseries = {}
        timeseries["GT"] = target.squeeze(0).cpu().detach().numpy()
        
        if sr_type == "temporal":
            scale = int(fs_hr // fs_lr)
            interpolated_input = nn.functional.interpolate(input, scale_factor=scale, mode='linear', align_corners=False)
            timeseries["LR Interpolated"] = interpolated_input.squeeze(0).cpu().detach().numpy()
            channel_to_plot = None # Plot first channel
        else:
            unmask_chs = unmask_channels[input_channels]
            channel_to_plot = None
            for ch in range(target_channels):
                if ch not in unmask_chs:
                    channel_to_plot = ch
                    break
            if channel_to_plot is None:
                channel_to_plot = 0
            lr_up = add_zero_channels(input, target_channels).to(device)
            timeseries["LR Input"] = lr_up.squeeze(0).cpu().detach().numpy()

        for name, model in models.items():
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                if model.model.use_diffusion:
                    #pred_sr = model.sample(input, pos, num_inference_steps=100)
                    batch_size = input.size(0)
                    # Sample timesteps as in training
                    t = torch.randint(
                        0,
                        num_train_timesteps,
                        (batch_size,),
                        device=device,
                        dtype=torch.long
                    )
                    # Diffuse HR
                    x_t_hr = torch.randn_like(target).to(device)
                    # Model prediction (same signature as training)
                    pred_sr = model(x_t_hr, t, input, pos=None)
                else:
                    pred_sr = model.model(input)
            timeseries[name] = pred_sr.squeeze(0).detach().cpu().numpy()

        print("Creating plots...")
        save_path = os.path.join(imgs_path, f'{sr_type}_{ablation_type}_example_{dataset_name}.png')
        plot_mean_timeseries(timeseries, save_path=save_path)
        #save_path_html = os.path.join(imgs_path, f'{sr_type}_{ablation_type}_example_{dataset_name}.html')  
        #plot_mean_timeseries_plotly(timeseries, channel_to_plot=channel_to_plot, save_path=save_path_html)

    return df, results

def split_result(str):
    if '±' not in str:
        return None
    else:
        mean, _ = str.strip().split('±')
        mean = float(mean)
        return mean

def final_validation(dataset_name, results_final, nfolds=1, ablations_path='./ablations'):
    os.makedirs(ablations_path, exist_ok=True)
    for sr_type, results_final_sr in results_final.items():
        if results_final_sr == {}:
            print(f"No results for SR type: {sr_type}")
            continue
        
        values_dict = {}
        for fold in range(nfolds):
            fold_key = fold + 1
            if fold_key not in results_final_sr:
                continue
            results_param_fold = results_final_sr[fold_key]
            for model_name, metric_dict in results_param_fold.items():
                if not isinstance(metric_dict, dict):
                    print(f"Warning: Skipping non-dict for {sr_type} fold {fold_key} model {model_name}")
                    continue
                if model_name not in values_dict:
                    values_dict[model_name] = {}
                for metric, str_val in metric_dict.items():
                    mean_val = split_result(str_val)
                    if mean_val is not None:
                        if metric not in values_dict[model_name]:
                            values_dict[model_name][metric] = []
                        values_dict[model_name][metric].append(mean_val)
        
        to_df = []
        for model_name, metric_vals in values_dict.items():
            row = {'Model': model_name}
            for metric, fold_means in metric_vals.items():
                m = np.mean(fold_means)
                s = np.std(fold_means)
                row[metric] = f"{m:.6f}±{s:.4f}"
            # Copy non-parsable metrics (e.g., Parameters)
            sample_fold = results_final_sr.get(1, {}).get(model_name, {})
            for metric, val in sample_fold.items():
                if metric not in row:
                    row[metric] = str(val)
            to_df.append(row)
        
        print(f"\n=== Final Validation Results for {sr_type} SR ===")
        df_final = pd.DataFrame(to_df).set_index('Model')
        print(tabulate(df_final, headers='keys', tablefmt='fancy_grid', showindex=True))
        
        param_str = 'temporal_sr_mamba_ablation' if 'temporal' in sr_type else 'spatial_sr_mamba_ablation'
        csv_path = os.path.join(ablations_path, f'{param_str}_{dataset_name}.csv')
        df_final.to_csv(csv_path)
        print(f"Saved: {csv_path}")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_ablation_study(dataset_name, fs_hr, fs_lr, target_channels, input_channels, multiplier, nfolds=1):

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    results_final = {"temporal": {}, "spatial": {}}

    for fold in range(nfolds):

        clear_memory()

        os.makedirs(os.path.join(models_path, f'ablations'), exist_ok=True)

        # Create train-test split
        patients = list(range(1, n_patients + 1))
        test_size = 0.2
        train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=seed)
        data_folder = data_path + os.sep + dataset_name

        dataloader_test = None  # Initialize dataloader_test
        dataloader_train = None  # Initialize dataloader_train

        print("\n=== Training and Evaluating Models ===")
        for sr_type in sr_types:
            clear_memory()
            print(f"\n--- SR Type: {sr_type} ---")
            results_final[sr_type] = {}

            # In your main loop, for each sr_type:
            if best_params is not None:
                print(f"Using predefined best params: {best_params}")
                best_hpo = {
                    "best_config": best_params,
                    "best_results": {}
                }
                models_nn = {}
                name = f"x{multiplier}_temporal" if sr_type == "temporal" else f"{input_channels}to64chs_spatial"
                models_nn[name] = DiBiMa_nn(
                    target_channels=64 if dataset_name=="mmi" else 62,
                    num_channels=target_channels if sr_type == "temporal" else input_channels,
                    fs_lr=fs_lr if sr_type == "temporal" else fs_hr,
                    fs_hr=fs_hr,
                    seconds=seconds,
                    residual_global=True,
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
                )
                if dataloader_train is None or dataloader_test is None:
                    models, dataloader_train, dataloader_test = prepare_dataloaders(
                        dataset_name, models_nn, train_patients, test_patients,
                        data_folder, quick_load=False, ref_position=None
                    )
                    ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
                else:
                    models = prepare_dataloaders(
                        dataset_name, models_nn, train_patients, test_patients,
                        data_folder, quick_load=True, ref_position=ref_position
                    )
                _, results = train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_test, fs_hr, fs_lr, target_channels, input_channels, multiplier=multiplier, ablation_type="Final")
                results_final[sr_type][fold+1] = results
            else:
                clear_memory()
                best_hpo = sequential_mamba_hpo(
                    dataset_name=dataset_name,
                    sr_type=sr_type,
                    fold=fold,
                    train_patients=train_patients,
                    test_patients=test_patients,
                    data_folder=data_folder,
                    metric_key="NMSE",
                    mamba_versions=mamba_versions,
                    mamba_dims=mamba_dims,
                    mamba_d_state=mamba_d_state,
                    n_mamba_layers=n_mamba_layers,
                    n_mamba_blocks=n_mamba_blocks,
                    target_channels=target_channels,
                    input_channels=input_channels,
                )
                # Access best params:
                print(f"Best: {best_hpo['best_config']}")
                results_final[sr_type][fold+1] = {best_hpo['best_config']['name']: best_hpo["best_results"]}  # or aggregate as needed
            #break  # Remove this break to run all sr_types
    
    print(results_final)
    print("\n\n=================== Final Validation Across Folds ===================")
    final_validation(dataset_name, results_final, nfolds=nfolds, ablations_path=ablations_path)

def main():
    dataset_names = ["mmi", "seed"]
    multiplier = 8
    for dataset_name in dataset_names:
        print(f"\n\n########## Running Ablation Study for Dataset: {dataset_name} ##########")
        if dataset_name == "mmi":
            fs_hr = 160
            target_channels = 64
        else:
            fs_hr = 200
            target_channels = 62
        input_channels = target_channels // multiplier  #only spatial sr 
        fs_lr = fs_hr // multiplier
        run_ablation_study(dataset_name, fs_hr, fs_lr, target_channels, input_channels, multiplier, nfolds=nfolds)
        break

if __name__ == '__main__':

    sys.exit(main())

#add tomography (location of electrodes) to guide-condition diffusion process: Done
#add SEED dataset: Done, testing now

#Downstream task: brain state classification (sleep stages, cognitive load, autism detection), more exploratory with explainability,


#Working on metrics.py line 180-193: using sample or forward? sample is to slow