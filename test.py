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
from visualize import plot_mean_timeseries
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
seeds = 2
epochs = 1
learning_rate = 1e-3
seconds = 10 #2 #9760 samples /160 Hz = 61 seconds
nfolds = 4 # Number of folds for cross-validation

torch.set_float32_matmul_precision('high')  # For better performance on GPUs with Tensor Cores

demo = True # Set to True for a quick demo run
debug = False  # Set to True to enable debug mode with additional logging

if demo:
    print("Demo mode activated: Using smaller dataset and fewer epochs for quick testing.")
    quick_load = False
    nfolds = 1

dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
n_patients = dict_n_patients["mmi"]  # Number of patients in the dataset   

models_path = project_path + os.sep + "model_weights"
data_path = project_path + os.sep + "eeg_data"
imgs_path = project_path + os.sep + "imgs"

if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

loss_fn = nn.MSELoss() #nn.MSELoss() #ReconstructionLoss()  # Loss function: callable function

sr_types = ["spatial", "temporal"]  # Types of super-resolution to evaluate

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

def load_model_weights(model, model_path):

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        raise Exception(f"Model weights file not found at {model_path}.")
    return model

def prepare_dataloaders(dataset_name, models_nn, train_patients, test_patients, data_folder, fold=0, quick_load=True, ref_position=None):

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
            models[name] = load_model_weights(models[name], os.path.join(models_path, f'fold_{fold+1}', f'DiBiMa_eeg_{name}_{fold+1}.pth'))
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
                                        plot=False).to(device)
            
            models[name].model = load_model_weights(models[name].model, os.path.join(models_path, f'fold_{fold+1}', f'DiBiMa_eeg_{name}_{fold+1}.pth'))

    if quick_load:
        return models 
    else:       
        return models, dataloader_train, dataloader_test
    

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

            print(f"\nTraining {name}...")

            model = model.to(device)
            model_nn = model.model.to(device)
            if model_nn.use_diffusion:
                num_train_timesteps = model.scheduler.num_train_timesteps
                inference_timesteps = num_train_timesteps
            else:
                inference_timesteps = None

            results[name], results_raw[name] = evaluate_model(model, dataloader_test, n_timesteps=inference_timesteps, evaluate_mean=True)
            model_path = os.path.join(models_path, f'fold_{fold+1}', f'DiBiMa_eeg_{name}_{fold+1}.pth')
            torch.save(model.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Plot boxplots for raw results       
    print("\nPlotting metric boxplots...")
    plot_metric_boxplots(results_raw, name = f'ablation_{ablation_type}_{sr_type}_fold{fold+1}_{dataset_name}', project_path=imgs_path)

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
        str_param = f"temporal_sr_mamba_test_{dataset_name}" if sr_type == "temporal" else f"spatial_sr_mamba_test_{dataset_name}"
        df_final.to_csv(f'{str_param}.csv')
        
def run(dataset_name, fs_hr=160, target_channels=64, multipliers=[2,4,8], nfolds=1):

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
        train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=seed)
        data_folder = data_path + os.sep + dataset_name

        dataloader_test = None  # Initialize dataloader_test
        dataloader_train = None  # Initialize dataloader_train

        print("\n=== Training and Evaluating Models ===")
        for sr_type in sr_types:
            print(f"\n--- SR Type: {sr_type} ---")
            results_final[sr_type] = {}

            for multiplier in multipliers:
                print(f"\n### Fold {fold+1}, Multiplier: {multiplier} ###")
                input_channels = target_channels if sr_type == "temporal" else int(target_channels // multiplier)
                fs_hr = fs_hr
                fs_lr = fs_hr // multiplier if sr_type == "temporal" else fs_hr
                
                # In your main loop, for each sr_type:
                if best_params is not None:
                    print(f"Using predefined best params: {best_params}")
                    models_nn = {}
                    name = f"x{multiplier}_temporal" if sr_type == "temporal" else f"{input_channels}to64chs_spatial"
                    models_nn[name] = DiBiMa_nn(
                        target_channels=target_channels,
                        num_channels=input_channels,
                        fs_lr=fs_lr,
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
                    results = validate_models(dataset_name, models, sr_type, dataloader_test, fs_hr, fs_lr, target_channels, input_channels, fold=fold, multiplier=multiplier, ablation_type="Final")
                    if fold+1 not in results_final[sr_type]:
                        results_final[sr_type][fold+1] = {name : results[name]}
                    else:
                        results_final[sr_type][fold+1][name] = results[name]

                    for name, model in models.items():
                        with torch.no_grad():
                            model.eval()
                            # UMAP of latent space
                            filepath = os.path.join(imgs_path,f'umap_{name}_fold{fold+1}.png')
                            plot_umap_latent_space( 
                                                    model, 
                                                    dataloader_test,
                                                    save_path=filepath,
                                                    map_labels=map_runs_mmi
                            )
                            print(f" Saved: {os.path.basename(filepath)}")                            
                            model.to('cpu')
                            del model
    print("\n\n=================== Final Validation Across Folds ===================")
    final_validation(dataset_name, results_final)

def main():
     
    set_seed(seed)
    dataset_names = ["mmi", "seed"]
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
        break

if __name__ == '__main__':

    sys.exit(main())
