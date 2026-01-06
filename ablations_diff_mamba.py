import diffusers
import torch
import torch.nn as nn
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

mne.set_log_level('ERROR') 

project_path = os.getcwd()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 16
epochs = 50
learning_rate = 0.001
seeds = [42, 7, 21, 84]
fs_hr = 160
seconds = 10 #9760 samples /160 Hz = 61 seconds

#runs = range(1, 14)  # EEG recording runs per subject
runs = [3, 4]  # For quicker testing, use only first 4 runs

quick_load = True # if True, load preprocessed data if available
debug = False  # Set to True to enable debug mode

nfolds = 1 # Number of folds for cross-validation

lr_window_length = fs_hr * seconds  # Low-resolution window length
hr_window_length = fs_hr * seconds  # High-resolution window length

#print(fs_lr, fs_hr, lr_window_length, hr_window_length)

torch.set_float32_matmul_precision('medium')

demo = False # Set to True for a quick demo run
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

loss_fn = ReconstructionLoss()  # Loss function: 'mse' or 'mae' or a callable function

sr_types = ["temporal", "spatial"]  # Types of super-resolution to evaluate

mamba_ablation = [True, False]
diffusion_ablation = [True, False]

def prepare_dataloaders(models_nn, sr_type, train_patients, test_patients, train_folder, test_folder, fs_hr=160, quick_load=True, add_noise=False, fold=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {}
    dataloaders_train = {"x2_temporal": None, "x4_temporal": None, "x8_temporal": None, "8to64chs_spatial": None, "16to64chs_spatial": None, "32to64chs_spatial": None}
    dataloaders_test = {"x2_temporal": None, "x4_temporal": None, "x8_temporal": None, "8to64chs_spatial": None, "16to64chs_spatial": None, "32to64chs_spatial": None}
    
    for name, model in models_nn.items():

        if "no_diffusion" in name:
            models[name] = DCAE_SR(model, learning_rate=learning_rate, loss_fn=loss_fn, debug=debug).to(device)
        else:
            prediction_type = "sample"  # "epsilon" or "sample"
            diffusion_params = {
                    "num_train_timesteps": 1000,
                    "beta_start": 1e-6, 
                    "beta_end": 1e-2, 
                    "beta_schedule": "squaredcos_cap_v2",
                    "prediction_type": prediction_type,
                    "clip_sample": False,
                    #"clip_sample_range": 1.0
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
                dataset_train = EEGDataset(subject_ids=train_patients, runs=runs, project_path=train_folder, add_noise=add_noise, verbose=False, fs_hr=fs_hr, fs_lr=fs_lr, seconds=seconds, demo=demo, num_channels=num_channels)
                print("Downloading testing data...")
                dataset_test = EEGDataset(subject_ids=test_patients, runs=runs, project_path=test_folder, add_noise=add_noise, verbose=False, fs_hr=fs_hr, fs_lr=fs_lr, seconds=seconds, demo=demo, num_channels=num_channels)
            else:  # channel-wise SR
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

def train_validate_models(models, sr_type, dataloaders_train, dataloaders_test, fold=0):

    results = {}
    results_raw = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, model in models.items():

        print(f"\nTraining {name}...")
        
        sr_name = name.split('_')[0] + '_' + sr_type
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
        results[name], results_raw[name] = evaluate_model(model_nn, dataloader_test)

        model_path = os.path.join(models_path, f'fold_{fold+1}', f'dcae_sr_eeg_{name}_{fold+1}.pth')
        torch.save(model.model.state_dict(), model_path)

    # Plot boxplots for raw results       
    plot_metric_boxplots(results_raw, save_path=f'ablation_boxplots_{sr_type}_diffmamba_fold{fold+1}.png', project_path=imgs_path)

    # Create DataFrame
    df = pd.DataFrame(results).T  # Transpose to have models as rows
    # Print formatted table using tabulate
    print("\n=== Ablation Study Results ===")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True))

    df.to_csv(os.path.join(ablations_path, f'ablation_study_results_{sr_type}_diffmamba_{fold+1}.csv'))

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
                dict_final[metric] = f"{overall_mean:.4f}±{overall_std:.4f}"
            to_df.append(dict_final)

        print(f"\n=== Final Validation Results for {sr_type} SR ===")    
        #print(dict_final)
        df_final = pd.DataFrame(to_df).set_index('Model')   
        print(tabulate(df_final, headers='keys', tablefmt='fancy_grid', showindex=True))
        str_param = f"temporal_sr_diffmamba_ablations" if sr_type == "temporal" else f"spatial_sr_diffmamba_ablations"
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
            if sr_type == "temporal":
                num_channels = 64  # EEG channels
                downsample_factors = [8]  # Temporal SR upsampling factors

                models_nn = {}
                for downsample_factor in downsample_factors:
                    fs_lr = int(fs_hr / downsample_factor)
                    for mamba in mamba_ablation:
                        for diffusion in diffusion_ablation:
                            if mamba and diffusion:
                                sr_type_model = 'temporal_mamba_diffusion'
                            elif mamba and not diffusion:
                                sr_type_model = 'temporal_mamba_no_diffusion'
                            elif not mamba and diffusion:
                                sr_type_model = 'temporal_diffusion_no_mamba'
                            else:
                                sr_type_model = 'temporal_no_mamba_no_diffusion'
                            models_nn[f'x{downsample_factor}_{sr_type_model}'] = DCAE_SR_nn(num_channels=num_channels, fs_lr=fs_lr, fs_hr=fs_hr, seconds=seconds, residual_global=True, residual_internal=True, use_subpixel=True, sr_type=sr_type, use_mamba=mamba)
                models, dataloaders_train, dataloaders_test = prepare_dataloaders(models_nn, sr_type, train_patients, test_patients, train_folder, test_folder, quick_load=quick_load, fold=fold, add_noise=False)                    
                df, results = train_validate_models(models, sr_type, dataloaders_train, dataloaders_test, fold)
                results_final[sr_type] = {}
                results_final[sr_type][fold+1] = results

            elif sr_type == "spatial":

                #nums_channels = [8, 16, 32]  # Spatial SR scale factors
                nums_channels = [8]
                fs_lr = fs_hr  # No temporal downsampling for spatial SR
                models_nn = {}
                for num_channels in nums_channels:
                    for mamba in mamba_ablation:
                        for diffusion in diffusion_ablation:
                            if mamba and diffusion:
                                sr_type_model = 'spatial_mamba_diffusion'
                            elif mamba and not diffusion:
                                sr_type_model = 'spatial_mamba_no_diffusion'
                            elif not mamba and diffusion:
                                sr_type_model = 'spatial_diffusion_no_mamba'
                            else:
                                sr_type_model = 'spatial_no_mamba_no_diffusion'
                            models_nn[f'{num_channels}to64chs_{sr_type_model}'] = DCAE_SR_nn(num_channels=num_channels, fs_lr=fs_lr, fs_hr=fs_hr, seconds=seconds, residual_global=True, residual_internal=True, use_subpixel=True, sr_type=sr_type, use_mamba=mamba)
                
                models, dataloaders_train, dataloaders_test = prepare_dataloaders(models_nn, sr_type, train_patients, test_patients, train_folder, test_folder, quick_load=quick_load, fold=fold, add_noise=False)  
                df, results = train_validate_models(models, sr_type, dataloaders_train, dataloaders_test, fold)
                results_final[sr_type] = {}
                results_final[sr_type][fold+1] = results
    
            print("\n\n=================== Final Validation Across All Folds ===================")
            final_validation(results_final)

        