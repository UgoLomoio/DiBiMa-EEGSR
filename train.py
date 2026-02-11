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
from utils import unmask_channels, plot_mean_timeseries
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
n_timesteps = 1000  # Number of diffusion timesteps
diffusion_params = {
                "num_train_timesteps": n_timesteps, #100,
                "beta_start": 1e-4,
                "beta_end": 1.5e-2,                                                                          
                "beta_schedule": "squaredcos_cap_v2",  #"linear" or "squaredcos_cap_v2"
                "prediction_type": prediction_type,
}

train_scheduler = DDPMScheduler(
    num_train_timesteps=diffusion_params["num_train_timesteps"],
    beta_start=diffusion_params["beta_start"],
    beta_end=diffusion_params["beta_end"],
    beta_schedule=diffusion_params["beta_schedule"],
    prediction_type=diffusion_params["prediction_type"],
)

val_scheduler = DDIMScheduler(
    num_train_timesteps=diffusion_params["num_train_timesteps"],
    beta_start=diffusion_params["beta_start"],
    beta_end=diffusion_params["beta_end"],
    beta_schedule=diffusion_params["beta_schedule"],
    prediction_type=diffusion_params["prediction_type"]
)
val_scheduler.eta = 1

# Hyperparameters
batch_size = 32
epochs = 30
learning_rate = 1e-3 
seed = 2
seconds = 2 #2 #9760 samples /160 Hz = 61 seconds
split_windows_first = True  # Whether to split windows or splitting patients first

nfolds = 1 # Number of folds for cross-validation

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = False # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")
    epochs = 1
    quick_load = False
    nfolds = 2

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

sr_types = ["temporal"]#, "temporal"]  # Types of super-resolution to evaluate

n_timesteps = 1000  # Number of diffusion timesteps

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
            "mamba_dim": 64,#128,  
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


def prepare_dataloaders_patients(dataset_name, train_patients, val_patients, data_folder, ref_position=None, target_channels=64):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Downloading training data...")
    dataset_train = EEGDataset(subject_ids=train_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)
    print("Downloading validation data...")
    dataset_val = EEGDataset(subject_ids=val_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)

    if len(dataset_train) == 0:
        print("No data loaded. Check dataset creation process.")
        exit(1)
            
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    print("Train and Val datasets loaded successfully.")    

    ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
    dataloader_val.dataset.ref_position = ref_position
    return dataloader_train, dataloader_val
    

def train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_val, fs_hr, fs_lr, target_channels, input_channels, fold=0, multiplier = None, plot_one_example=True, ablation_type="Final"):

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
        dataloader_train.dataset.fs_lr = fs_lr if sr_type == "temporal" else fs_hr
        dataloader_val.dataset.fs_lr = fs_lr if sr_type == "temporal" else fs_hr
        dataloader_train.dataset.num_channels = target_channels if sr_type == "temporal" else input_channels
        dataloader_val.dataset.num_channels = target_channels if sr_type == "temporal" else input_channels
    
        for name, model in models.items():

            print(f"\nTraining {name}...")

            model = model.to(device)
            summary = torchinfo.summary(model.model)
            with open(os.path.join(models_path, f'fold_{fold+1}', f'{name}_{dataset_name}_model_summary.txt'), 'w') as f:
                f.write(str(summary))
                #early_stopping_callback = EarlyStopping(monitor='avg_val_loss', patience=20, verbose=False, mode='min')

            trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1 if torch.cuda.is_available() else None, logger=False, enable_checkpointing=False)#, callbacks=[early_stopping_callback])
            trainer.fit(model, dataloader_train, val_dataloaders=dataloader_val)
                
            print(f"Evaluating {name}...")
            model = model.to(device)
            model.eval()
            results[name], results_raw[name] = evaluate_model(model, dataloader_val, sample_type="lr_upsampled", flatten=True, evaluate_mean=False)
            if model.__class__.__name__ == "DiBiMa_Diff":
                path = os.path.join(models_path, f'fold_{fold+1}', f'DiBiMa_eeg_{name}_{dataset_name}_{fold+1}.pth')
            else:
                path = os.path.join(models_path, f'fold_{fold+1}', f'BiMa_eeg_{name}_{dataset_name}_{fold+1}.pth')
            model_path = os.path.join(path)
            torch.save(model.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Plot barplots for raw results       
    print("\nPlotting metric barplots...")
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
            unmask_chs = unmask_channels[dataset_name][f'x{multiplier}'].copy()
            channel_to_plot = None
            for ch in range(target_channels):
                if ch not in unmask_chs:
                    channel_to_plot = ch
                    break
            if channel_to_plot is None:
                channel_to_plot = 0
            lr_up = add_zero_channels(input, target_channels, dataset_name=dataset_name, multiplier=multiplier).to(device)
            timeseries["LR Input"] = lr_up.squeeze(0).cpu().detach().numpy()

        for name, model in models.items():
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                if model.__class__.__name__ == "DiBiMa_Diff":
                    pred_sr = model.sample_from_lr(input, pos=pos, label=label) 
                else:
                    pred_sr = model.model(input)
            timeseries[name] = pred_sr.squeeze(0).detach().cpu().numpy()

        print("Creating plots...")
        save_path = os.path.join(imgs_path, f'{sr_type}_{ablation_type}_example_{dataset_name}.png')
        plot_mean_timeseries(timeseries, save_path=save_path)

    return df, results

def split_result(str):
    if '±' not in str:
        return None
    else:
        mean, _ = str.strip().split('±')
        mean = float(mean)
        return mean

def final_validation(dataset_name, results_final):

    for sr_type, results_final_sr in results_final.items():
        
        if results_final_sr == {}:
            print(f"No results for SR type: {sr_type}")
            continue
        
        values_dict = {}
        for fold, results_param_fold in results_final_sr.items():
            for model_name, metric_dict in results_param_fold.items():
                if model_name not in values_dict:
                    values_dict[model_name] = {}
                for metric_name, metric_value in metric_dict.items():
                    metric_value = split_result(metric_value)
                    if metric_value is not None:
                        values_dict[model_name][metric_name] = metric_value

        #print(values_dict)

        to_df = []
        # Compute overall mean and std across folds
        for model_name, dict1 in values_dict.items():
            dict_final = {}
            dict_final['Model'] = model_name
            #print(f"\nFinal results for model: {model_name}")
            for metric, values in dict1.items():       
                #print(f" - Metric: {metric}, Values across folds: {values}")
                overall_mean = np.mean(values)
                overall_std = np.std(values)
                dict_final[metric] = f"{overall_mean:.6f}±{overall_std:.4f}"
            to_df.append(dict_final)

        print(f"\n=== Final Validation Results for {sr_type} SR ===")    
        #print(dict_final)
        df_final = pd.DataFrame(to_df).set_index('Model')   
        print(tabulate(df_final, headers='keys', tablefmt='fancy_grid', showindex=True))
        str_param = f"temporal_sr_mamba_ablation_{dataset_name}" if sr_type == "temporal" else f"spatial_sr_mamba_ablation_{dataset_name}"
        df_final.to_csv(os.path.join(ablations_path, f'{str_param}.csv'))
        
def run(dataset_name, fs_hr=160, target_channels=64, multipliers=[8,4,2], nfolds=1):

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    results_final = {"temporal": {}, "spatial": {}}

    for fold in range(nfolds):
        
        gc.collect()
        empty_cache()

        os.makedirs(os.path.join(models_path, f'fold_{fold+1}'), exist_ok=True)

        # Create train-test split
        patients = list(range(1, n_patients + 1))
        test_size = 0.2
        train_patients, val_patients, test_patients = train_test_val_split_patients(patients, test_size=test_size, random_state=seed)
        data_folder = data_path + os.sep + dataset_name

        dataloader_val = None  # Initialize dataloader_val
        dataloader_train = None  # Initialize dataloader_train

        print("\n=== Training and Evaluating Models ===")
        for sr_type in sr_types:
            print(f"\n--- SR Type: {sr_type} ---")
            results_final[sr_type] = {}

            for multiplier in multipliers:
                print(f"\n### Fold {fold+1}, Multiplier: {multiplier} ###")
                input_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f'x{multiplier}'])
                print(f"Input channels: {input_channels}, Target channels: {target_channels}")
                fs_hr = fs_hr
                fs_lr = fs_hr // multiplier if sr_type == "temporal" else fs_hr
                
                for diff in [False, True]:
                    
                    print(f" Use Diffusion: {diff}")
                    best_params = base_params_temporal if sr_type == "temporal" else base_params_spatial
                    if diff:
                        best_params["use_diffusion"] = True
                        for key in base_params_cond:
                            best_params[key] = base_params_cond[key]
                    else:
                        best_params["use_diffusion"] = False
                        for key in base_params_cond:
                            best_params[key] = False

                    # In your main loop, for each sr_type:
                    if best_params is not None:
                        print(f"Using predefined best params: {best_params}")
                        best_hpo = {
                            "best_config": best_params,
                            "best_results": {}
                        }
                        models_nn = {}
                        name = f"x{multiplier}_temporal" if sr_type == "temporal" else f"{input_channels}to{target_channels}chs_spatial"
                        models_nn[name] = DiBiMa_nn(
                            target_channels=target_channels,
                            num_channels=input_channels,
                            fs_lr=fs_lr,
                            fs_hr=fs_hr,
                            seconds=seconds,
                            residual_global=True,
                            residual_internal=best_params['internal_residual'],
                            use_subpixel=True,
                            sr_type=sr_type,
                            use_mamba=best_params["use_mamba"],
                            use_diffusion=best_params["use_diffusion"],
                            n_mamba_layers=best_params["n_mamba_layers"],
                            mamba_dim=best_params["mamba_dim"],
                            mamba_d_state=best_params["mamba_d_state"],
                            mamba_version=best_params["mamba_version"],
                            n_mamba_blocks=best_params["n_mamba_blocks"],
                            use_positional_encoding=False,
                            use_electrode_embedding=best_params["use_electrode_embedding"],  
                            merge_type=best_params['merge_type'],
                            use_label=best_params['use_label'],
                            use_lr_conditioning=best_params['use_lr_conditioning'],
                            dataset_name=dataset_name,
                            multiplier=multiplier
                        )
                        if dataloader_train is None or dataloader_val is None:
                            if split_windows_first:
                                dataset = EEGDataset(subject_ids=train_patients+val_patients+test_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)
                                dataloader_train, dataloader_val = prepare_dataloaders_windows(
                                    dataset_name, dataset, seconds=seconds, batch_size=batch_size
                                )
                                del dataset
                            else:
                                dataloader_train, dataloader_val = prepare_dataloaders_patients(
                                    dataset_name, train_patients, val_patients, data_folder, ref_position=None
                                )
                        models = {}
                        for name, model in models_nn.items():
                            ref_position = dataloader_train.dataset.ref_position.to(device)  # Reference electrode positions
                            model.ref_position = ref_position  # Set reference positions in the model

                            if model.use_diffusion == False:
                                models[name] = DiBiMa(model, learning_rate=learning_rate, loss_fn=loss_fn, debug=debug).to(device)
                            else:
                                models[name] = DiBiMa_Diff(model,
                                                            train_scheduler=train_scheduler,
                                                            val_scheduler=val_scheduler,
                                                            criterion=loss_fn,
                                                            learning_rate=learning_rate,
                                                            predict_type=prediction_type,  # "epsilon" or "sample"
                                                            debug=debug,
                                                            epochs=epochs,
                                                            plot=False).to(device)
                                
                        _, results = train_validate_models(dataset_name, models, sr_type, dataloader_train, dataloader_val, fs_hr, fs_lr, target_channels, input_channels, fold=fold, multiplier=multiplier, ablation_type="Final")
                        if fold+1 not in results_final[sr_type]:
                            results_final[sr_type][fold+1] = {name : results[name]}
                        else:
                            results_final[sr_type][fold+1][name] = results[name]
                    #break # Remove this break to run on all diffusion settings
                #break  # Remove this break to run on all multipliers
            #break  # Remove this break to run on all SR types
    print("\n\n=================== Final Validation Across Folds ===================")
    final_validation(dataset_name, results_final)

def main():
     
    set_seed(seed)
    #clear_directory(models_path, ignore=["ablations"])
    dataset_names = ["mmi", "seed"]
    multipliers = [8, 4, 2]  # SR multipliers
    for dataset_name in dataset_names:
        print(f"\n\n########## Running Ablation Study for Dataset: {dataset_name} ##########")
        if dataset_name == "mmi":
            fs_hr = 160
            target_channels = 64
        else:
            fs_hr = 200
            target_channels = 62
        run(dataset_name, fs_hr=fs_hr, target_channels=target_channels, multipliers=multipliers, nfolds=nfolds)
        #break  # Remove this break to run on all datasets

if __name__ == '__main__':
    sys.exit(main())
