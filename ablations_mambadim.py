from curses import nl
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

mne.set_log_level('ERROR') 

project_path = os.getcwd()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 16 #64
epochs = 10
learning_rate = 0.01
seeds = [42, 2, 21, 84]
fs_hr = 160
fs_lr = 20
input_channels = 16  # For spatial SR
target_channels = 64  # For spatial SR
seconds = 10 #2 #9760 samples /160 Hz = 61 seconds

#runs = range(1, 14)  # EEG recording runs per subject
runs = [3, 4]  # For quicker testing, use only first 4 runs

quick_load = False # if True, load preprocessed data if available

nfolds = 1 # Number of folds for cross-validation

lr_window_length = fs_hr * seconds  # Low-resolution window length
hr_window_length = fs_hr * seconds  # High-resolution window length

#print(fs_lr, fs_hr, lr_window_length, hr_window_length)

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = False # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")
    epochs = 2
    quick_load = False
    nfolds = 1

n_patients = 110  # Number of patients in the dataset   

models_path = project_path + os.sep + "model_weights"
data_path = project_path + os.sep + "eeg_data"
preprocessed_data_path = data_path + os.sep + "preprocessed"
imgs_path = project_path + os.sep + "imgs"
ablations_path = project_path + os.sep + "ablation_results"
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)
if not os.path.exists(ablations_path):
    os.makedirs(ablations_path)

loss_fn = nn.L1Loss() #nn.MSELoss() #ReconstructionLoss()  # Loss function: callable function

sr_types = ["spatial", "temporal"]  # Types of super-resolution to evaluate

n_mamba_blocks = [1, 2, 3]  # Number of Mamba blocks in each Bi-Mamba layer
mamba_versions = [1, 2]  # Mamba version to use (1 or 2), 3 is not implemented yet but exists
mamba_dims = [128, 256, 512]  # Mamba dimension (number of channels in Mamba layers)
mamba_d_state = [16, 64, 128] # Mamba state dimension (number of channels in Mamba state)
n_mamba_layers = [1]  # Number of Bi-Mamba layers in the model

mamba = True  # Use Mamba layers in the model
diff = False  # Use diffusion models
n_timesteps = 100  # Number of diffusion timesteps

best_params = {
    "version": 2,
    "dim": 64,
    "d_state": 32, #16
    "n_mamba_blocks": 5, #7
    "n_mamba_layers": 1
}
#best_params = None  # Set to None to perform full HPO

if best_params is not None:
    epochs = 10  # Increase epochs if using best params directly


def prepare_dataloaders(models_nn, sr_type, train_patients, test_patients, train_folder, test_folder, fs_hr=160, quick_load=True, add_noise=False, fold=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {}
    dataloaders_train = {"x2_temporal": None, "x4_temporal": None, "x8_temporal": None, "8to64chs_spatial": None, "16to64chs_spatial": None, "32to64chs_spatial": None}
    dataloaders_test = {"x2_temporal": None, "x4_temporal": None, "x8_temporal": None, "8to64chs_spatial": None, "16to64chs_spatial": None, "32to64chs_spatial": None}
    
    for name, model in models_nn.items():

        if model.use_diffusion == False:
            models[name] = DCAE_SR(model, learning_rate=learning_rate, loss_fn=loss_fn, debug=debug).to(device)
        else:
            prediction_type = "epsilon"  # "epsilon" or "sample"
            diffusion_params = {
                    "num_train_timesteps": n_timesteps, #100,
                    "beta_start": 1e-6, 
                    "beta_end": 1e-3,
                    "beta_schedule": "squaredcos_cap_v2",
                    "prediction_type": prediction_type,
                    "clip_sample": True,
                    "clip_sample_range": 1.0
            }
            models[name] = DCAE_SR_Diff(model,
                                        loss_fn,
                                        diffusion_params=diffusion_params,
                                        learning_rate=learning_rate,
                                        scheduler_params=None,
                                        predict_type=prediction_type,  # "epsilon" or "sample"
                                        debug=debug,
                                        plot=False).to(device)

        sr = name.split('_')[0]
        multiplier = int(sr[1:]) if sr_type == "temporal" else None
        sr_name = sr + '_' + sr_type
        if dataloaders_train[sr_name] is not None and dataloaders_test[sr_name] is not None:
            continue  # Already loaded
        
        if quick_load:

            train_path = os.path.join(preprocessed_data_path, f'train_sr_{sr_name}_{fold+1}.pt')
            test_path = os.path.join(preprocessed_data_path, f'test_sr_{sr_name}_{fold+1}.pt')

            if os.path.exists(train_path) and os.path.exists(test_path):
                print("=== Loading Preprocessed Datasets ===")
                dataset_train = torch.load(train_path, weights_only=False)
                dataset_test = torch.load(test_path, weights_only=False)
                dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
                print("Train and Test datasets loaded successfully from preprocessed files.")
            else:
                print("Preprocessed data not found. Loading and preprocessing datasets...")
                quick_load = False  # Fallback to full loading and preprocessing

        if not quick_load:
            
            print("=== Loading Datasets ===")
            print("Downloading training data...")
            if sr_type == "temporal":
                fs_lr = fs_hr // multiplier
                num_channels = 64  # All channels for temporal SR
                # Temporal SR
                dataset_train = EEGDataset(subject_ids=train_patients, runs=runs, project_path=train_folder, add_noise=add_noise, verbose=False, fs_hr=fs_hr, fs_lr=fs_lr, seconds=seconds, demo=demo, num_channels=num_channels)
                print("Downloading testing data...")
                dataset_test = EEGDataset(subject_ids=test_patients, runs=runs, project_path=test_folder, add_noise=add_noise, verbose=False, fs_hr=fs_hr, fs_lr=fs_lr, seconds=seconds, demo=demo, num_channels=num_channels)
            else:  # channel-wise SR
                input_channels = int(sr.split('to')[0])
                num_channels = input_channels  # Only a subset of channels for spatial SR
                fs_lr = fs_hr  # No change in temporal resolution for spatial SR
                # Spatial SR
                dataset_train = EEGDataset(subject_ids=train_patients, runs=runs, project_path=train_folder, add_noise=add_noise, verbose=False, fs_lr=fs_hr, fs_hr=fs_hr, seconds=seconds, demo=demo, num_channels=num_channels)
                print("Downloading testing data...")
                dataset_test = EEGDataset(subject_ids=test_patients, runs=runs, project_path=test_folder, add_noise=add_noise, verbose=False, fs_lr=fs_hr, fs_hr=fs_hr, seconds=seconds, demo=demo, num_channels=num_channels)

            if len(dataset_train) == 0:
                print("No data loaded. Check dataset creation process.")
                exit(1)
            
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

            os.makedirs(preprocessed_data_path, exist_ok=True)
            if not demo:
                torch.save(dataset_train, os.path.join(preprocessed_data_path, f'train_sr_{sr_name}_{fold+1}.pt'))
                torch.save(dataset_test, os.path.join(preprocessed_data_path, f'test_sr_{sr_name}_{fold+1}.pt'))
            print("Train and Test datasets loaded successfully.")
        
        dataloaders_train[sr_name] = dataloader_train
        dataloaders_test[sr_name] = dataloader_test

    return models, dataloaders_train, dataloaders_test

def sequential_mamba_hpo(sr_type, fold, train_patients, test_patients, train_folder, test_folder, 
                        quick_load=True, mamba=True, fs_lr=20, seconds=10, fs_hr=160, num_channels=64, input_channels=16,
                        mamba_versions=[1,2], mamba_dims=[64,128,256], mamba_d_state=[16,64], 
                        n_mamba_layers=[1], n_mamba_blocks=[1, 2, 3], metric_key="val_mse", use_diffusion=True):
    """
    Sequential HPO: version → dim → d_state → n_layers.
    
    Returns: dict with best params and their results.
    """
    if sr_type == "spatial":
        fs_lr = fs_hr #temporal resolution is not changed in spatial SR
        sr_nam = f"{input_channels}to64chs_{sr_type}"
    elif sr_type == "temporal":
        multiplier = fs_hr // fs_lr #160/20=8 
        input_channels = num_channels #all channels are used for temporal SR
        sr_nam = f"x{multiplier}_{sr_type}"

    def run_search(configs, use_diffusion=use_diffusion):
        models_nn = {}
        for cfg in configs:
            models_nn[cfg["name"]] = DCAE_SR_nn(
                num_channels=input_channels,
                fs_lr=fs_lr,
                fs_hr=fs_hr,
                seconds=seconds,
                residual_global=False,
                residual_internal=True,
                use_subpixel=True,
                sr_type=sr_type,
                use_diffusion=use_diffusion,
                use_mamba=mamba,
                n_mamba_blocks=cfg["n_mamba_blocks"],
                n_mamba_layers=cfg["n_mamba_layers"],
                mamba_dim=cfg["mamba_dim"],
                mamba_d_state=cfg["mamba_d_state"],
                mamba_version=cfg["mamba_version"],
            )
        models, dataloaders_train, dataloaders_test = prepare_dataloaders(
            models_nn, sr_type, train_patients, test_patients,
            train_folder, test_folder, quick_load=quick_load, fold=fold, add_noise=False, fs_hr=fs_hr
        )
        df, results = train_validate_models(models, sr_type, dataloaders_train, dataloaders_test, fold)
        return results
    
    print(f"\n=== Sequential HPO for {sr_type} ===")
    
    # Stage 1: mamba_version
    configs = [{"name": f"{sr_nam}_mamba{v}", "n_mamba_layers": 1, "mamba_dim": 256, "mamba_d_state": 16, "mamba_version": v}
               for v in mamba_versions]
    results = run_search(configs, use_diffusion=use_diffusion)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_sequential_hpo_{sr_type}_mamba_version_results_fold{fold+1}.csv'))
    best_version = min(configs, key=lambda c: results[c["name"]][metric_key])["mamba_version"]
    print(f"✓ Best version: {best_version}")
    
    # Stage 2: mamba_dim
    configs = [{"name": f"{sr_nam}_mamba{best_version}_d{d}", "n_mamba_layers": 1, "mamba_dim": d, "mamba_d_state": 16, "mamba_version": best_version}
               for d in mamba_dims]
    results = run_search(configs, use_diffusion=use_diffusion)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_sequential_hpo_{sr_type}_mamba_dim_results_fold{fold+1}.csv'))
    best_dim = min(configs, key=lambda c: results[c["name"]][metric_key])["mamba_dim"]
    print(f"✓ Best dim: {best_dim}")
    
    # Stage 3: mamba_d_state
    configs = [{"name": f"{sr_nam}_mamba{best_version}_d{best_dim}_ds{ds}", "n_mamba_layers": 1, "mamba_dim": best_dim, "mamba_d_state": ds, "mamba_version": best_version}
               for ds in mamba_d_state]
    results = run_search(configs, use_diffusion=use_diffusion)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_sequential_hpo_{sr_type}_mamba_d_state_results_fold{fold+1}.csv'))
    best_d_state = min(configs, key=lambda c: results[c["name"]][metric_key])["mamba_d_state"]
    print(f"✓ Best d_state: {best_d_state}")
    
    # Stage 4: n_mamba_blocks
    configs = [{"name": f"{sr_nam}_mamba{best_version}_nb{nb}_d{best_dim}_ds{best_d_state}", 
                "n_mamba_layers": 1, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version,
                "n_mamba_blocks": nb}
               for nb in n_mamba_blocks]
    results = run_search(configs, use_diffusion=use_diffusion)      
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_sequential_hpo_{sr_type}_mamba_n_blocks_results_fold{fold+1}.csv'))
    best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    best_namba_blocks = best_config["n_mamba_blocks"]
    print(f"✓ Best n_mamba_blocks: {best_namba_blocks}")   

    # Stage 5: n_mamba_layers
    configs = [{"name": f"{sr_nam}_mamba{best_version}_nb{best_namba_blocks}_d{best_dim}_ds{best_d_state}", 
                "n_mamba_layers": nl, "mamba_dim": best_dim, "mamba_d_state": best_d_state, "mamba_version": best_version}
               for nl in n_mamba_layers]
    results = run_search(configs, use_diffusion=use_diffusion)
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(ablations_path, f'ablation_sequential_hpo_{sr_type}_mamba_n_layers_results_fold{fold+1}.csv'))
    best_config = min(configs, key=lambda c: results[c["name"]][metric_key])
    best_results = results[best_config["name"]]
    
    print(f"\n🎯 FINAL BEST: version={best_version}, dim={best_dim}, d_state={best_d_state}, n_blocks={best_namba_blocks}, n_layers={best_config['n_mamba_layers']}")
    print(f"   Best metric ({metric_key}): {best_results[metric_key]}")
    
    return {
        "best_config": best_config,
        "best_results": best_results,
        "all_results": results
    }

def train_validate_models(models, sr_type, dataloaders_train, dataloaders_test, fold=0, plot_one_example=True):

    results = {}
    results_raw = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, model in models.items():

        print(f"\nTraining {name}...")
        
        sr_name = name.split('_')[0] + '_' + sr_type
        print(f"Using dataloaders for SR: {sr_name}")
        print(dataloaders_train.keys())
        dataloader_train = dataloaders_train[sr_name]
        dataloader_test = dataloaders_test[sr_name]

        model = model.to(device)
        summary = torchinfo.summary(model.model)
        with open(os.path.join(models_path, f'fold_{fold+1}', f'{name}_model_summary.txt'), 'w') as f:
            f.write(str(summary))
            #early_stopping_callback = EarlyStopping(monitor='avg_val_loss', patience=20, verbose=False, mode='min')

        trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1 if torch.cuda.is_available() else None, logger=False, enable_checkpointing=False)#, callbacks=[early_stopping_callback])
        trainer.fit(model, dataloader_train, val_dataloaders=dataloader_test)
              
        print(f"Evaluating {name}...")
        model_nn = model.model.to(device)
        if model_nn.use_diffusion:
            num_train_timesteps = model.scheduler.num_train_timesteps
            if num_train_timesteps <= 100:
                inference_timesteps = num_train_timesteps 
            else:
                inference_timesteps = num_train_timesteps // 10
        else:
            inference_timesteps = None

        results[name], results_raw[name] = evaluate_model(model_nn, dataloader_test, n_timesteps=inference_timesteps)
        model_path = os.path.join(models_path, f'fold_{fold+1}', f'dcae_sr_eeg_{name}_{fold+1}.pth')
        torch.save(model.model.state_dict(), model_path)

    # Plot boxplots for raw results       
    plot_metric_boxplots(results_raw, name = f'ablation_{sr_type}_fold{fold+1}', project_path=imgs_path)

    # Create DataFrame
    df = pd.DataFrame(results).T  # Transpose to have models as rows
    # Print formatted table using tabulate
    print("\n=== Ablation Study Results ===")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True))

    df.to_csv(os.path.join(ablations_path, f'ablation_study_results_{sr_type}_mamba_ablations_{fold+1}.csv'))

    if plot_one_example:

        data = next(iter(dataloader_test))
        input, target = data
        input = input.to(device)
        target = target.to(device)

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
            timeseries["LR Input"] = input.squeeze(0).cpu().detach().numpy()

        for name, model in models.items():
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                if model.model.use_diffusion:
                    pred_sr = model.sample(input, num_inference_steps=100)
                else:
                    pred_sr = model.model(input)
            timeseries[name] = pred_sr.squeeze(0).detach().cpu().numpy()

        save_path = os.path.join(imgs_path, f'{sr_type}_mamba_ablation_example.png')
        plot_mean_timeseries(timeseries, save_path=save_path)
        save_path_html = os.path.join(imgs_path, f'{sr_type}_mamba_ablation_example.html')  
        plot_mean_timeseries_plotly(timeseries, channel_to_plot=channel_to_plot, save_path=save_path_html)

    return df, results

def split_result(str):
    if '±' not in str:
        return None
    else:
        mean, _ = str.strip().split('±')
        mean = float(mean)
        return mean

def final_validation(results_final):

    for sr_type, results_final_sr in results_final.items():
        
        if results_final_sr == {}:
            print(f"No results for SR type: {sr_type}")
            continue
                
        values_dict = {}
        for fold in range(nfolds):
            results_param_fold = results_final_sr[fold+1]  
            for model_name, metric in results_param_fold.items():
                values_dict[model_name] = {}
                for metric in results_param_fold[model_name].keys():
                    # Aggregate across folds
                    str = results_param_fold[model_name][metric]
                    mean = split_result(str)
                    if mean is not None:
                        values_dict[model_name][metric] = mean

        to_df = []
        # Compute overall mean and std across folds
        for model_name, dict1 in values_dict.items():
            dict_final = {}
            dict_final['Model'] = model_name
            for metric, values in dict1.items():       
                overall_mean = np.mean(values)
                overall_std = np.std(values)
                dict_final[metric] = f"{overall_mean:.6f}±{overall_std:.4f}"
            to_df.append(dict_final)

        print(f"\n=== Final Validation Results for {sr_type} SR ===")    
        #print(dict_final)
        df_final = pd.DataFrame(to_df).set_index('Model')   
        print(tabulate(df_final, headers='keys', tablefmt='fancy_grid', showindex=True))
        str_param = f"temporal_sr_mamba_ablation" if sr_type == "temporal" else f"spatial_sr_mamba_ablation"
        df_final.to_csv(os.path.join(ablations_path, f'{str_param}.csv'))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    results_final = {"temporal": {}, "spatial": {}}

    for fold in range(nfolds):

        os.makedirs(os.path.join(models_path, f'fold_{fold+1}'), exist_ok=True)
        seed = seeds[fold]
        print(f"\n\n=================== Fold {fold+1}/{nfolds} - Seed: {seed} ===================")
        set_seed(seed)

        # Create train-test split
        patients = list(range(1, n_patients + 1))
        test_size = 0.2
        train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=seed)
        train_folder = data_path + os.sep + "train_data"
        test_folder = data_path + os.sep + "test_data"
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        print("\n=== Training and Evaluating Models ===")
        for sr_type in sr_types:
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
                name = f"x8_temporal" if sr_type == "temporal" else f"16to64chs_spatial"
                models_nn[name] = DCAE_SR_nn(
                    num_channels=target_channels if sr_type == "temporal" else input_channels,
                    fs_lr=fs_lr if sr_type == "temporal" else fs_hr,
                    fs_hr=fs_hr,
                    seconds=seconds,
                    residual_global=True,
                    residual_internal=True,
                    use_subpixel=True,
                    sr_type=sr_type,
                    use_mamba=mamba,
                    use_diffusion=diff,
                    n_mamba_layers=best_params["n_mamba_layers"],
                    mamba_dim=best_params["dim"],
                    mamba_d_state=best_params["d_state"],
                    mamba_version=best_params["version"],
                    n_mamba_blocks=best_params["n_mamba_blocks"],
                )
                models, dataloaders_train, dataloaders_test = prepare_dataloaders(
                    models_nn, sr_type, train_patients, test_patients,
                    train_folder, test_folder, quick_load=quick_load, fold=fold, add_noise=False, fs_hr=fs_hr
                )
                df, results = train_validate_models(models, sr_type, dataloaders_train, dataloaders_test, fold)
                results_final[sr_type][fold+1] = results
            else:
                best_hpo = sequential_mamba_hpo(
                    sr_type=sr_type,
                    fold=fold,
                    train_patients=train_patients,
                    test_patients=test_patients,
                    train_folder=train_folder,
                    test_folder=test_folder,
                    quick_load=quick_load,
                    metric_key="NMSE",
                    use_diffusion=diff,
                    mamba_versions=mamba_versions,
                    mamba_dims=mamba_dims,
                    mamba_d_state=mamba_d_state,
                    n_mamba_layers=n_mamba_layers,
                    n_mamba_blocks=n_mamba_blocks
                )
                # Access best params:
                print(f"Best: {best_hpo['best_config']}")
                results_final[sr_type][fold+1] = best_hpo["best_results"]  # or aggregate as needed
            break  # Remove this break to run all folds
    print("\n\n=================== Final Validation Across Folds ===================")
    final_validation(results_final)
