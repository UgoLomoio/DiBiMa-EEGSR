from diffusers import DDPMScheduler, DDIMScheduler
import torch
from torch import nn
from models import *
from utils import *
import os 
from metrics import *
from torch.utils.data import DataLoader
import pandas as pd
from tabulate import tabulate
import mne
from pytorch_lightning import Trainer
import torchinfo 
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

#Diffusion scheduler
prediction_type = "sample"  # "epsilon", "sample" or "v_prediction"
n_timesteps = 1000 # Number of diffusion timesteps
diffusion_params = {
                "num_train_timesteps": n_timesteps, #100,
                "beta_start": 1e-4,
                "beta_end": 0.015,                                                                          
                "beta_schedule": "squaredcos_cap_v2",  #"linear" or "squaredcos_cap_v2"
                "prediction_type": prediction_type,
                "clip_sample": False
}

train_scheduler = DDPMScheduler(
    num_train_timesteps=diffusion_params["num_train_timesteps"],
    beta_start=diffusion_params["beta_start"],
    beta_end=diffusion_params["beta_end"],
    beta_schedule=diffusion_params["beta_schedule"],
    prediction_type=diffusion_params["prediction_type"],
    clip_sample=diffusion_params["clip_sample"]
)

val_scheduler = DDIMScheduler(
    num_train_timesteps=diffusion_params["num_train_timesteps"],
    beta_start=diffusion_params["beta_start"],
    beta_end=diffusion_params["beta_end"],
    beta_schedule=diffusion_params["beta_schedule"],
    prediction_type=diffusion_params["prediction_type"],
    clip_sample=diffusion_params["clip_sample"]
)
val_scheduler.eta = 1.0  # DDIM eta parameter

# Hyperparameters
batch_size = 32
epochs = 30
learning_rate = 1e-3 #0.0001
seed = 2
seconds = 2 #9760 samples /160 Hz = 61 seconds
set_seed(seed)
split_windows_first = True  # Whether to split windows or splitting patients first

nfolds = 1 # Number of folds for cross-validation

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = False  # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging
validate_only = True  # Set to True to only run validation without training

dict_n_patients = {
    "mmi": 109,
    "seed": 15
}

models_path = project_path + os.sep + "model_weights"
data_path = project_path + os.sep + "eeg_data"
#preprocessed_data_path = data_path + os.sep + "preprocessed"
imgs_path = project_path + os.sep + "imgs"
ablations_path = project_path + os.sep + "ablation_results"
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)
if not os.path.exists(ablations_path):
    os.makedirs(ablations_path)

loss_fn = EEGSuperResolutionLoss(
    lambda_grad=0.5,      # Start with equal weight to MSE
    lambda_corr=0.2,      # Moderate correlation enforcement
    lambda_freq=0.5,      # Lower weight for frequency
    use_freq_loss=True
)                                                   #temporal
loss_fn = nn.MSELoss()     # spatial

sr_types = ["temporal"]#, "spatial"]  # Types of super-resolution to evaluate

#need to fix epsilon diffusion, it's important!!
base_params_temporal = {
            "mamba_version": 2,
            "mamba_dim": 128, #64,   
            "mamba_d_state": 8, #8, 
            "n_mamba_blocks": 2, #4,
            "n_mamba_layers": 2, #2,
            "use_mamba": True,
            'merge_type': 'add',  # 'concat' or 'add'
            'internal_residual': True,
            'use_diffusion': True
}
base_params_spatial = {
            "mamba_version": 1,#2
            "mamba_dim": 128,  
            "mamba_d_state": 16, #8, 
            "n_mamba_blocks": 3, #4,
            "n_mamba_layers": 2, #2,
            "use_mamba": True,
            'merge_type': 'add',  # 'concat' or 'add'
            'internal_residual': True,
            'use_diffusion': True
}
base_params_cond = {
    "use_electrode_embedding": True,
    "use_label": False,
    "use_lr_conditioning": True
}
base_params_cond = None  # Set to None to run full HPO, or define as above to use specific params for final validation

#quick_load = False

if base_params_cond is not None:
    epochs = 20  # Increase epochs if using best params directly

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")
    epochs = 20
    quick_load = False
    nfolds = 1

def prepare_dataloaders_patients(dataset_name, train_patients, val_patients, data_folder, quick_load=True, ref_position=None):

    num_channels = 64 if dataset_name == "mmi" else 62

    print("Downloading training data...")
    dataset_train = EEGDataset(subject_ids=train_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=num_channels, seconds=seconds)
    print("Downloading testing data...")
    dataset_val = EEGDataset(subject_ids=val_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=num_channels, seconds=seconds)

    if len(dataset_train) == 0:
        print("No data loaded. Check dataset creation process.")
        exit(1)
            
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    print("Train and val datasets loaded successfully.")    
    
    return dataloader_train, dataloader_val

def prepare_dataloaders_windows(dataset_name, dataset, seconds = 2, batch_size=32, return_test=False):

    num_channels = 64 if dataset_name == "mmi" else 62

    windows = dataset.datas_hr
    positions = dataset.positions
    labels = dataset.labels
    channel_names = dataset.channel_names

    if return_test:
        (train_windows, train_labels, train_positions), (val_windows, val_labels, val_positions), (test_windows, test_labels, test_positions) = train_test_val_split(windows, labels, positions, test_size=0.2, val_size=0.1, random_state=seed)
    else:
        (train_windows, train_labels, train_positions), (val_windows, val_labels, val_positions), _ = train_test_val_split(windows, labels, positions, test_size=0.2, val_size=0.1, random_state=seed)
    
    print("Creating training and val datasets from windows...")
    dataset_train = EEGWindowsDataset(windows=train_windows, positions=train_positions,
                                     labels=train_labels, channel_names=channel_names, dataset_name=dataset_name,
                                     target_channels=num_channels, fs_hr=dataset.fs_hr,
                                     multiplier=dataset.multiplier)
    dataset_val = EEGWindowsDataset(windows=val_windows, positions=val_positions,
                                    labels=val_labels, channel_names=channel_names, dataset_name=dataset_name,
                                    target_channels=num_channels, fs_hr=dataset.fs_hr,
                                    multiplier=dataset.multiplier)
    if return_test:
        dataset_test = EEGWindowsDataset(windows=test_windows, positions=test_positions,
                                        labels=test_labels, channel_names=channel_names, dataset_name=dataset_name,
                                        target_channels=num_channels, fs_hr=dataset.fs_hr,
                                        multiplier=dataset.multiplier)
        
    print("Preparing training and val dataloaders from windows...")
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    if return_test:
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        print("Train, Val and Test dataloaders prepared successfully.")    
        return dataloader_train, dataloader_val, dataloader_test
  
    print("Train and Val dataloaders prepared successfully.")    
    return dataloader_train, dataloader_val


def select_best_model(results, configs, metric_key="NMSE", selected_name=""):
    if metric_key in ["MSE", "NMSE", "MAE"]:
        # Lower is better
        best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    else:
        # Higher is better
        best_config = max(configs, key=lambda c: results[c["name"]][metric_key])
    best_param = best_config[selected_name]
    best_model_name = best_config["name"]
    best_results = results[best_model_name]
    return best_param, best_config, best_results

def sequential_conditioning_hpo(dataset_name, sr_type, fold, train_patients, val_patients, test_patients, data_folder,
                        base_params, fs_lr=20, seconds=2, fs_hr=160, target_channels=64, input_channels=16,
                        metric_key="NMSE", multiplier = 8, split_windows_first=True):
    """
    Sequential HPO: no conditioning, +lr_conditioning, +electrode embedding, +label conditioning
    
    Returns: dict with best params and their results.
    """
    if sr_type == "spatial":
        fs_lr = fs_hr #temporal resolution is not changed in spatial SR
        sr_name = f"{input_channels}to{target_channels}chs_{sr_type}"
    elif sr_type == "temporal":
        input_channels = target_channels #all channels are used for temporal SR
        sr_name = f"x{multiplier}_{sr_type}"
    multiplier = multiplier if sr_type == "spatial" else 2 
    
    def run_search(configs, fs_hr, fs_lr, target_channels, input_channels, base_params, ablation_type="", dataloader_train=None, dataloader_val=None, multiplier=8):
        
        clear_memory()

        if split_windows_first:
            dataset = EEGDataset(subject_ids=train_patients + test_patients + val_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)
            dataloader_train, dataloader_val = prepare_dataloaders_windows(
                dataset_name, dataset, seconds=seconds, batch_size=batch_size
            )
        else:
            dataloader_train, dataloader_val = prepare_dataloaders_patients(
                dataset_name, train_patients, val_patients,
                data_folder, quick_load=quick_load, ref_position=None
            )
        ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
        if demo:
            num_classes = 6 if dataset_name == "mmi" else 3
        else:
            num_classes = dataloader_train.dataset.num_classes  # Number of classes for label conditioning
        print(f"Number of classes for label conditioning: {num_classes}")
        models_nn = {}
        for cfg in configs:
            models_nn[cfg["name"]] = DiBiMa_nn(
                target_channels=target_channels,
                num_channels=input_channels,
                fs_lr=fs_lr,
                fs_hr=fs_hr,
                seconds=seconds,
                residual_global=True, #in diffusion
                residual_internal=base_params["internal_residual"],
                use_subpixel=True,
                use_positional_encoding=False,
                use_electrode_embedding=cfg["use_electrode_embedding"],
                use_lr_conditioning=cfg["use_lr_conditioning"],
                use_label=cfg["use_label"],
                sr_type=sr_type,
                use_diffusion=base_params["use_diffusion"],
                use_mamba=base_params["use_mamba"],
                n_mamba_blocks=base_params["n_mamba_blocks"],
                n_mamba_layers=base_params["n_mamba_layers"],
                mamba_dim=base_params["mamba_dim"],
                mamba_d_state=base_params["mamba_d_state"],
                mamba_version=base_params["mamba_version"],
                merge_type=base_params["merge_type"],  # 'concat' or 'add'
                num_classes=num_classes,
                ref_position=ref_position, 
                dataset_name=dataset_name,
                multiplier=multiplier
        )        
        
        models = {}
        for name, model in models_nn.items():
            
            models[name] = DiBiMa_Diff(model,
                                        train_scheduler=train_scheduler,
                                        val_scheduler=val_scheduler,
                                        criterion=loss_fn,
                                        learning_rate=learning_rate,
                                        debug=debug,
                                        predict_type=prediction_type,
                                        epochs=epochs,
                                        plot=True).to(device)
            
        _, results = train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_val, fs_hr, fs_lr, target_channels, input_channels, multiplier=multiplier, ablation_type=ablation_type)

        del models, models_nn
        torch.cuda.empty_cache()
        gc.collect()
        return results
    
    print(f"\n=== Sequential HPO for {sr_type} ===")
    
    configs = [
        {"name": f"{sr_name}_no_conditioning", "conditioning": "no_conditioning", "use_electrode_embedding": False, "use_lr_conditioning": False, "use_label": False},
        {"name": f"{sr_name}_lr_conditioning", "conditioning": "lr_conditioning", "use_electrode_embedding": False, "use_lr_conditioning": True, "use_label": False},
        {"name": f"{sr_name}_lr_and_electrode_conditioning", "conditioning": "lr_and_electrode_conditioning", "use_electrode_embedding": True, "use_lr_conditioning": True, "use_label": False},
        {"name": f"{sr_name}_all_conditioning", "conditioning": "all_conditioning", "use_electrode_embedding": True, "use_lr_conditioning": True, "use_label": True}
    ]
    results = run_search(configs, fs_hr=fs_hr, fs_lr=fs_lr, base_params=base_params, target_channels=target_channels, input_channels=input_channels, ablation_type="conditioning")
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_{sr_type}_{dataset_name}_conditioning.csv'))
    best_conditioning, best_config, best_results = select_best_model(results, configs, metric_key=metric_key, selected_name="conditioning")
    print(f"✓ Best conditioning: {best_conditioning}")
    
    return {
        "best_config": best_config,
        "best_results": best_results,
        "all_results": results
    }

def train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_val, fs_hr, fs_lr, target_channels, input_channels, fold=0, multiplier = 8, plot_one_example=True, ablation_type="Final"):

    results = {}
    results_raw = {}

    if multiplier is None:
        multipliers = [8, 4, 2]
    else:
        multipliers = [multiplier]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for multiplier in multipliers:

        print(f" - Multiplier: {multiplier}, SR Type: {sr_type}")
        dataloader_train.dataset.multiplier = multiplier
        dataloader_val.dataset.multiplier = multiplier
        dataloader_train.dataset.sr_type = sr_type
        dataloader_val.dataset.sr_type = sr_type
        dataloader_train.dataset.fs_lr = fs_lr
        dataloader_val.dataset.fs_lr = fs_lr
        dataloader_train.dataset.num_channels = input_channels
        dataloader_val.dataset.num_channels = input_channels
    
        for name, model in models.items():

            model_path = os.path.join(models_path, f'ablations', f'DiBiMa_eeg_{name}_{dataset_name}_{fold+1}.pth')

            if os.path.exists(model_path) and validate_only:
                print(f"\nLoading pre-trained model for {name} from {model_path}...")
                model.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                model.model = model.model.to(device)
                model = model.to(device)
            else:
                print(f"\nTraining {name}...")
                model = model.to(device)
                summary = torchinfo.summary(model.model)
                with open(os.path.join(models_path, f'ablations', f'{name}_{dataset_name}_model_summary.txt'), 'w') as f:
                    f.write(str(summary))
                    #early_stopping_callback = EarlyStopping(monitor='avg_val_loss', patience=20, verbose=False, mode='min')

                trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1 if torch.cuda.is_available() else None, logger=False, enable_checkpointing=False, check_val_every_n_epoch=1)#, callbacks=[early_stopping_callback])
                trainer.fit(model, dataloader_train, val_dataloaders=dataloader_val)
                torch.save(model.model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            print(f"Evaluating {name}...")
            sample_type = "noise" if "no_conditioning" in name else "lr_upsampled"
            results[name], results_raw[name] = evaluate_model(model, dataloader_val, evaluate_mean=False, sample_type="noise")
            results[name+"lr_upsampled"], results_raw[name] = evaluate_model(model, dataloader_val, evaluate_mean=False, sample_type="lr_upsampled")

    # Plot boxplots for raw results       
    print("\nPlotting metric boxplots...")
    plot_metric_barplots(results_raw, name = f'ablation_{ablation_type}_{sr_type}_fold{fold+1}_{dataset_name}', project_path=imgs_path)

    # Create DataFrame
    df = pd.DataFrame(results).T  # Transpose to have models as rows
    # Print formatted table using tabulate
    print("\n=== Ablation Study Results ===")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True))
    df.to_csv(os.path.join(ablations_path, f'ablation_{ablation_type}_{sr_type}_{fold+1}_{dataset_name}.csv'))

    if plot_one_example:

        print("\nInference timeseries...")
        data = next(iter(dataloader_val))
        input, target, pos, label = data
        input = input.to(device)
        target = target.to(device)
        pos = pos.to(device)
        label = label.to(device)

        timeseries = {}
        timeseries["GT"] = target.squeeze(0).cpu().detach().numpy()
        
        if sr_type == "temporal":
            scale = int(fs_hr // fs_lr)
            interpolated_input = nn.functional.interpolate(input, scale_factor=scale, mode='linear', align_corners=False)
            timeseries["LR Interpolated"] = interpolated_input.squeeze(0).cpu().detach().numpy()
            channel_to_plot = None # Plot first channel
        else:
            unmask_chs = unmask_channels[dataset_name][f"x{multiplier}"]
            channel_to_plot = None
            for ch in range(target_channels):
                if ch not in unmask_chs:
                    channel_to_plot = ch
                    break
            if channel_to_plot is None:
                channel_to_plot = 0
            lr_up = add_zero_channels(input, target_channels=target_channels, dataset_name=dataset_name, multiplier=multiplier).to(device)
            timeseries["LR Input"] = lr_up.squeeze(0).cpu().detach().numpy()

        for name, model in models.items():
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                if model.model.use_diffusion:
                    pred_sr = model.sample(input, pos=pos, label=label)
                    if "no_conditioning" not in name:
                        pred_sr_from_lr = model.sample_from_lr(input, pos=pos, label=label)
                        timeseries[f"{name}_from_LR"] = pred_sr_from_lr.squeeze(0).detach().cpu().numpy()
                    timeseries[f"{name}_from_noise"] = pred_sr.squeeze(0).detach().cpu().numpy()
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

def run_ablation_study(dataset_name, fs_hr, fs_lr, target_channels, input_channels, multiplier, nfolds=1, seconds=2, num_patients=None):

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    results_final = {"temporal": {}, "spatial": {}}

    for fold in range(nfolds):

        clear_memory()

        os.makedirs(os.path.join(models_path, f'ablations'), exist_ok=True)

        # Create train-test split
        n_patients = dict_n_patients[dataset_name]
        patients = list(range(1, n_patients + 1))
        test_size = 0.2
        val_size = 0.1
        train_patients, val_patients, test_patients = train_test_val_split_patients(patients, test_size=test_size, val_size=val_size, random_state=seed)
        data_folder = data_path + os.sep + dataset_name

        dataloader_val = None  # Initialize dataloader_val
        dataloader_train = None  # Initialize dataloader_train

        print("\n=== Training and Evaluating Models ===")
        for sr_type in sr_types:
            clear_memory()
            print(f"\n--- SR Type: {sr_type} ---")
            results_final[sr_type] = {}
            base_params = base_params_temporal if sr_type == "temporal" else base_params_spatial
            best_params = base_params.copy()  # Start with base params
            if base_params_cond is not None:
                for key in base_params_cond:
                    best_params[key] = base_params_cond[key]

            # In your main loop, for each sr_type:
            if base_params_cond is not None:
                print(f"Using predefined best params: {best_params}")
                best_hpo = {
                    "best_config": best_params,
                    "best_results": {}
                }
                models_nn = {}
                target_channels = 64 if dataset_name=="mmi" else 62
                num_channels = target_channels if sr_type == "temporal" else input_channels
                name = f"x{multiplier}_temporal" if sr_type == "temporal" else f"{input_channels}to{target_channels}chs_spatial"
                if dataloader_train is None or dataloader_val is None:
                    if split_windows_first:
                        dataset = EEGDataset(subject_ids=train_patients+val_patients+test_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)
                        dataloader_train, dataloader_val = prepare_dataloaders_windows(
                            dataset_name, dataset, seconds=seconds, batch_size=batch_size
                        )
                        del dataset
                    else:
                        dataloader_train, dataloader_val = prepare_dataloaders_patients(
                            dataset_name, train_patients, val_patients, test_patients,
                            data_folder, quick_load=False, ref_position=None
                        )
                    ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
                if demo:
                    num_classes = 6 if dataset_name == "mmi" else 3
                else:
                    num_classes = dataloader_train.dataset.num_classes  # Number of classes for label conditioning
                models_nn[name] = DiBiMa_nn(
                    target_channels=target_channels,
                    num_channels=num_channels,
                    fs_lr=fs_lr if sr_type == "temporal" else fs_hr,
                    fs_hr=fs_hr,
                    seconds=seconds,
                    residual_global=True, #in diffusion
                    residual_internal=base_params["internal_residual"],
                    use_subpixel=True,
                    sr_type=sr_type,
                    use_mamba=base_params["use_mamba"],
                    use_diffusion=base_params["use_diffusion"],
                    n_mamba_layers=base_params["n_mamba_layers"],
                    mamba_dim=base_params["mamba_dim"],
                    mamba_d_state=base_params["mamba_d_state"],
                    mamba_version=base_params["mamba_version"],
                    n_mamba_blocks=base_params["n_mamba_blocks"],
                    use_positional_encoding=False,
                    use_electrode_embedding=best_params["use_electrode_embedding"],  
                    use_lr_conditioning=best_params["use_lr_conditioning"],
                    use_label=best_params["use_label"],
                    merge_type=base_params['merge_type'], 
                    ref_position=ref_position,
                    num_classes=num_classes,
                    dataset_name = dataset_name,
                    multiplier=multiplier
                )
                models = {}
                for name, model in models_nn.items():
                    models[name] = DiBiMa_Diff(model,
                                                train_scheduler=train_scheduler,
                                                val_scheduler=val_scheduler,
                                                criterion=loss_fn,
                                                learning_rate=learning_rate,
                                                predict_type=prediction_type,
                                                debug=debug,
                                                epochs=epochs,
                                                plot=True).to(device)
                _, results = train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_val, fs_hr, fs_lr, target_channels, input_channels, multiplier=multiplier, ablation_type="Final", plot_one_example=False)
                results_final[sr_type][fold+1] = results
            else:
                clear_memory()
                best_hpo = sequential_conditioning_hpo(
                    dataset_name=dataset_name,
                    sr_type=sr_type,
                    fold=fold,
                    target_channels=64 if dataset_name=="mmi" else 62,
                    input_channels=target_channels if sr_type == "temporal" else input_channels,
                    fs_lr=fs_lr if sr_type == "temporal" else fs_hr,
                    fs_hr=fs_hr,
                    seconds=seconds,
                    train_patients=train_patients,
                    val_patients=val_patients,
                    test_patients=test_patients,
                    data_folder=data_folder,
                    metric_key="PCC",
                    base_params=base_params,
                    multiplier=multiplier      
                )
                # Access best params:
                print(f"Best: {best_hpo['best_config']}")
                results_final[sr_type][fold+1] = {best_hpo['best_config']['name']: best_hpo["best_results"]}  # or aggregate as needed
            #break  # Remove this break to run all sr_types
        #break  # Remove this break to run all folds
    print(results_final)
    print("\n\n=================== Final Validation Across Folds ===================")
    final_validation(dataset_name, results_final, nfolds=nfolds, ablations_path=ablations_path)

def main():
    dataset_names = ["seed"]  # Add other dataset names as needed
    multiplier = 8  # Upsampling factor
    for dataset_name in dataset_names:
        print(f"\n\n########## Running Ablation Study for Dataset: {dataset_name} ##########")
        if dataset_name == "mmi":
            fs_hr = 160
            target_channels = 64
        else:
            fs_hr = 200
            target_channels = 62
        input_channels = len(unmask_channels[dataset_name][f"x{multiplier}"])
        #print(f"Input channels: {input_channels}, Target channels: {target_channels}")
        fs_lr = fs_hr // multiplier
        n_patients = dict_n_patients[dataset_name]
        run_ablation_study(dataset_name, fs_hr, fs_lr, target_channels, input_channels, multiplier, nfolds=nfolds, seconds=seconds, num_patients=n_patients)
        #break  # Remove this break to run all datasets
if __name__ == '__main__':
    sys.exit(main())
