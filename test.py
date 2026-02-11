import torch
from torch import nn
from models import *
from utils import *
import os 
from metrics import *
import pandas as pd
from tabulate import tabulate
import mne
import gc 
from torch.cuda import empty_cache
import sys
from utils import set_seed, plot_mean_timeseries, unmask_channels, add_zero_channels
from train import * 

gc.collect()
empty_cache()
mne.set_log_level('ERROR') 

project_path = os.getcwd()

sr_types = ["spatial"] #["temporal", "spatial"]
nfolds = 4

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = False # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging
skip_umap = True  # Set to True to skip UMAP plotting for faster execution

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")
    quick_load = False
    nfolds = 2

dict_n_patients = {
    "mmi": 109,
    "seed": 15
} 

models_path = project_path + os.sep + "model_weights"
data_path = project_path + os.sep + "eeg_data"
imgs_path = project_path + os.sep + "imgs"

if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

def load_model_weights(model, model_path):

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Model weights loaded from {model_path}")
    else:
        raise Exception(f"Model weights file not found at {model_path}.")
    return model

def prepare_models(models_nn, ref_position, dataset_name, fold=0):

    models = {}
    for model_name, model in models_nn.items():
        model.ref_position = ref_position  # Set reference positions in the model
        print(f"Preparing model: {model_name}, Diffusion: {model.use_diffusion}")
        name_clean = model_name.replace(f"_diff{int(model.use_diffusion)}", "")
        if model.use_diffusion == False:
            print(f"Loading non-diffusion model: {model_name}")
            name = f"BiMa_eeg_{name_clean}_{dataset_name}_{fold+1}"
            models[name] = DiBiMa(model, learning_rate=learning_rate, loss_fn=loss_fn, debug=debug).to(device)
        else:
            name = f"DiBiMa_eeg_{name_clean}_{dataset_name}_{fold+1}"
            models[name] = DiBiMa_Diff(model,
                                       train_scheduler=train_scheduler,
                                       val_scheduler=val_scheduler,
                                       criterion=loss_fn,
                                       learning_rate=learning_rate,
                                       predict_type=prediction_type,  # "epsilon" or "sample"
                                       debug=debug,
                                       epochs=epochs,
                                       plot=False).to(device)
        path = os.path.join(models_path, f'fold_{fold+1}', f'{name}.pth')
        print(f"Loading model weights from: {path}")
        models[name].model = load_model_weights(models[name].model, path)
    return models     

def validate_models(dataset_name, models, sr_type, dataloader_test, fs_hr, fs_lr, target_channels, input_channels, fold=0, multiplier = None, plot_one_example=True, ablation_type="Final"):

    results = {}
    results_raw = {}

    if multiplier is None:
        multipliers = [2, 4, 8]
    else:
        multipliers = [multiplier]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for multiplier in multipliers:

        print(f" - Multiplier: {multiplier}, SR Type: {sr_type}")
        dataloader_test.dataset.multiplier = multiplier
        dataloader_test.dataset.sr_type = sr_type
        dataloader_test.dataset.fs_lr = fs_lr if sr_type == "temporal" else fs_hr
        dataloader_test.dataset.num_channels = target_channels if sr_type == "temporal" else input_channels

        for name, model in models.items():

            print(f"\nTesting {name}...")
            model = model.to(device)
            model.eval()
            if model.model.use_diffusion:
                if model.model.use_lr_conditioning:
                    results[name], results_raw[name] = evaluate_model(model, dataloader_test, flatten=True, sample_type="lr_conditioned", evaluate_mean=False)
                else:
                    results[name], results_raw[name] = evaluate_model(model, dataloader_test, flatten=True, sample_type="noise", evaluate_mean=False)
            else:
                results[name], results_raw[name] = evaluate_model(model, dataloader_test, flatten=True, sample_type=None, evaluate_mean=False)
         
    # Plot barplots for raw results       
    print("\nPlotting metric barplots...")
    plot_metric_barplots(results_raw, name = f'ablation_{ablation_type}_{sr_type}_fold{fold+1}_{dataset_name}', project_path=imgs_path)

    if plot_one_example:

        print("\nInference timeseries...")
        data = next(iter(dataloader_test))
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
            unmask_chs = unmask_channels[dataset_name][f'x{multiplier}']
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
                if model.model.use_diffusion:
                    if model.model.use_lr_conditioning:
                        pred_sr = model.sample_from_lr(input, pos=pos, label=label)
                    else:
                        pred_sr = model.sample(input, pos=pos, label=label)
                else:
                    pred_sr = model.model(input)
            timeseries[name] = pred_sr.squeeze(0).detach().cpu().numpy()

        print("Creating plots...")
        save_path = os.path.join(imgs_path, f'{sr_type}_{ablation_type}_example_{dataset_name}.png')
        plot_mean_timeseries(timeseries, save_path=save_path)

    return results

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
                print(f"Fold {fold}, Model: {model_name}, Metrics: {metric_dict}")
                if model_name not in values_dict:
                    values_dict[model_name] = {}
                for metric_name, metric_value in metric_dict.items():
                    if metric_name in ["Parameters"]:
                        continue
                    metric_value = split_result(metric_value)
                    if metric_name not in values_dict[model_name]:
                        #print(f" - Metric: {metric_name}, Value: {metric_value}")
                        values_dict[model_name][metric_name] = [metric_value]
                    else:
                        values_dict[model_name][metric_name].append(metric_value)

        #print(values_dict)

        to_df = []
        # Compute overall mean and std across folds
        for model_name, dict1 in values_dict.items():
            dict_final = {}
            dict_final['Model'] = model_name
            print(f"\nFinal results for model: {model_name}")
            for metric, values in dict1.items():       
                print(f" - Metric: {metric}, Values across folds: {values}")
                overall_mean = np.mean(values)
                overall_std = np.std(values)
                print(f" - Metric: {metric}, Overall Mean: {overall_mean}, Overall Std: {overall_std}")
                dict_final[metric] = f"{overall_mean:.6f}±{overall_std:.4f}"
            to_df.append(dict_final)

        print(f"\n=== Final Validation Results for {sr_type} SR ===")    
        #print(dict_final)
        df_final = pd.DataFrame(to_df).set_index('Model')   
        print(tabulate(df_final, headers='keys', tablefmt='fancy_grid', showindex=True))
        str_param = f"temporal_sr_mamba_test_{dataset_name}" if sr_type == "temporal" else f"spatial_sr_mamba_test_{dataset_name}"
        df_final.to_csv(f'{str_param}.csv')
        
def run(dataset_name, fs_hr=160, target_channels=64, multipliers=[2,4,8], nfolds=1):

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    results_final = {"temporal": {}, "spatial": {}}
    n_patients = dict_n_patients[dataset_name]  # Number of patients in the dataset  
    
    for fold in range(nfolds):
        
        gc.collect()
        empty_cache()

        os.makedirs(os.path.join(models_path, f'fold_{fold+1}'), exist_ok=True)

        # Create train-test split
        patients = list(range(1, n_patients + 1))
        test_size = 0.2
        train_patients, val_patients, test_patients = train_test_val_split_patients(patients, test_size=test_size, random_state=seed)
        data_folder = data_path + os.sep + dataset_name

        # Prepare dataloaders
        if split_windows_first:
            dataset = EEGDataset(subject_ids=train_patients+val_patients+test_patients, data_folder=data_folder, dataset_name=dataset_name, verbose=False, demo=demo, num_channels=target_channels, seconds=seconds)
            dataloader_train, _, dataloader_test = prepare_dataloaders_windows(
                        dataset_name, dataset, seconds=seconds, batch_size=batch_size, return_test=True
            )
            del dataset
        else:
            dataloader_train, dataloader_test = prepare_dataloaders_windows(
                        dataset_name, dataset, seconds=seconds, batch_size=batch_size, return_test=True
            )

        #torch.save(dataloader_test.dataset, os.path.join("eeg_data", 'dataset_test.pth'))
        #break
        print("\n=== Training and Evaluating Models ===")
        for sr_type in sr_types:
            dataloader_test.dataset.sr_type = sr_type
        
            print(f"\n--- SR Type: {sr_type} ---")
            for multiplier in multipliers:
                dataloader_test.dataset.multiplier = multiplier
                dataloader_test.dataset.fs_lr = fs_hr // multiplier if sr_type == "temporal" else fs_hr
                dataloader_test.dataset.num_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f"x{multiplier}"])
                
                print(f"\n### Fold {fold+1}, Multiplier: {multiplier} ###")
                input_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f"x{multiplier}"])
                fs_hr = fs_hr
                fs_lr = fs_hr // multiplier if sr_type == "temporal" else fs_hr
                base_params = base_params_temporal if sr_type == "temporal" else base_params_spatial
                best_params = base_params.copy()

                models_nn = {}
                for diffusion in [False, True]:
                    best_params["use_diffusion"] = diffusion
                    
                    for param_name in base_params_cond.keys():
                        if diffusion:
                            best_params[param_name] = base_params_cond[param_name]
                        else:
                            best_params[param_name] = False 
    
                    print(f"Using predefined best params: {best_params}")
                    name = f"x{multiplier}_temporal_diff{int(diffusion)}" if sr_type == "temporal" else f"{input_channels}to{target_channels}chs_spatial_diff{int(diffusion)}"
                    models_nn[name] = DiBiMa_nn(
                            target_channels=target_channels,
                            num_channels=input_channels,
                            fs_lr=fs_lr,
                            fs_hr=fs_hr,
                            seconds=seconds,
                            residual_global=True,
                            residual_internal=best_params["internal_residual"],
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
                    
                models = prepare_models(models_nn, ref_position=None, dataset_name=dataset_name, fold=fold)
                results = validate_models(dataset_name, models, sr_type, dataloader_test, fs_hr, fs_lr, target_channels, input_channels, fold=fold, multiplier=multiplier, ablation_type="Final")
                for model_name, metric_dict in results.items():
                    model_name_clean = model_name[:-2]#exclude fold number from name
                    if fold+1 not in results_final[sr_type]:
                        results_final[sr_type][fold+1] = {model_name_clean: metric_dict}
                    else: 
                        results_final[sr_type][fold+1][model_name_clean] = results[model_name]

                if not skip_umap:
                    for name, model in models.items():
                        with torch.no_grad():
                            model.eval()
                            # UMAP of latent space
                            dataloader_train.dataset.multiplier = multiplier
                            dataloader_train.dataset.sr_type = sr_type
                            dataloader_train.dataset.fs_lr = fs_lr if sr_type == "temporal" else fs_hr
                            dataloader_train.dataset.num_channels = target_channels if sr_type == "temporal" else input_channels
                            dataloader_test.dataset.multiplier = multiplier
                            dataloader_test.dataset.sr_type = sr_type
                            dataloader_test.dataset.fs_lr = fs_lr if sr_type == "temporal" else fs_hr
                            dataloader_test.dataset.num_channels = target_channels if sr_type == "temporal" else input_channels

                            filepath = os.path.join(imgs_path,f'umap_{name}_fold{fold+1}.png')
                            plot_umap_latent_space( 
                                                    model, 
                                                    dataloader_train,
                                                    dataloader_test,
                                                    save_path=filepath
                            )
                            print(f" Saved: {os.path.basename(filepath)}")                            
                            model.to('cpu')
                            del model
    print("\n\n=================== Final Validation Across Folds ===================")
    final_validation(dataset_name, results_final)

def main():
     
    set_seed(seed)
    dataset_names = ["seed"]#, "mmi"]#, "seed"]
    multipliers = [2, 4, 8]  # SR multipliers
    for dataset_name in dataset_names:
        print(f"\n\n########## Running Ablation Study for Dataset: {dataset_name} ##########")
        if dataset_name == "mmi":
            fs_hr = 160
            target_channels = 64
        else:
            fs_hr = 200
            target_channels = 62
        run(dataset_name, fs_hr=fs_hr, target_channels=target_channels, multipliers=multipliers, nfolds=nfolds)
        #break

if __name__ == '__main__':

    sys.exit(main())
