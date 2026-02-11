import math
import os
import sys
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from utils_class import load_model_weights, EEGDatasetClassification
import pandas as pd
from resnet50 import ResNet50, ResNetPL  
from metrics_class import compute_metrics, plot_multiple_roc_curves
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.getcwd()
parent_dir = os.path.join(current_dir, os.pardir)  # or os.path.dirname(current_dir)
parent_abs = os.path.abspath(parent_dir)
print(f"Adding to sys.path: {parent_abs}")
sys.path.insert(0, parent_abs)
from test import *
from models import DiBiMa_Diff, DiBiMa_nn

cwd = os.getcwd()

os.makedirs('model_weights', exist_ok=True)

# Data directory
DATA_DIR = os.path.join(parent_dir, 'eeg_data')

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 1    # Adjust based on your system
MULTIPLIER = 8
LR = 1e-4
MAX_EPOCHS = 50

demo = False
seed = 2
dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = DATA_DIR

if demo:
    MAX_EPOCHS = 2
    BATCH_SIZE = 4

map_labels = {
    0: 'eyes_closed',
    1: 'eyes_open'
}

def plot_umap_embeddings(model, dataloader, dataset_name, input_type, sr_type):

    print(f"Generating UMAP embeddings for {dataset_name} dataset...")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.model = model.model.to(device)
    device = next(model.parameters()).device
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Processing batch {i+1}/{len(dataloader)}.", end='\r')
            inputs, labels = batch
            inputs = inputs.to(device)
            _, _, _, embeddings = model(inputs, return_embeddings=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu()) 
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    print("Fitting UMAP...")
    umap = UMAP(n_components=2, random_state=seed)
    embeddings_2d = umap.fit_transform(all_embeddings)
    
    print("Plotting UMAP embeddings...")
    plt.figure(figsize=(8, 6))
    for label in np.unique(all_labels):
        idx = all_labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'{map_labels[label]}', c=colors[label], s=5)
    plt.title(f'UMAP Embeddings for {dataset_name} Dataset, {input_type} {sr_type} Data')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.savefig(f'umap_{dataset_name}_{input_type}_{sr_type}.png')
    plt.close()

def main():

    for dataset_name in ["mmi"]:#not using 'seed' in classification
        target_channels = 64 if dataset_name == "mmi" else 62
        fs_hr = 160 if dataset_name == "mmi" else 200
       
        # Data
        num_subjects = dict_n_patients[dataset_name]
        all_ids = list(range(1, num_subjects + 1)) 
        train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=seed)
                
        model_sr = None

        dataset_path = os.path.join(data_folder, dataset_name)
                  
        dataset_test = EEGDatasetClassification(test_ids, dataset_path, dataset_name=dataset_name, model_sr=model_sr, demo=demo) 
        val_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
               
        seconds = 2
        results = {}

        for i, sr_type in enumerate(["spatial"]):
            dict_for_auc = {}
            results[sr_type] = {}
            for input_type in ["hr", "lr", "sr"]:
                if input_type == "hr" and i >= 1:
                    print(f"Skipping {input_type} {sr_type} since HR spatial data is not available.")
                    continue
                results[sr_type][input_type] = {}
                if input_type == "lr":
                    in_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f"x{MULTIPLIER}"])
                    fs_lr = int(fs_hr//MULTIPLIER) if sr_type == "temporal" else fs_hr
                elif input_type == "hr":
                    in_channels = target_channels
                else:
                    #Model
                    in_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f"x{MULTIPLIER}"])
                    fs_lr = int(fs_hr//MULTIPLIER) if sr_type == "temporal" else fs_hr
                    str_param = f"x{MULTIPLIER}" if sr_type == "temporal" else f"{in_channels}to{target_channels}chs"
                    best_params = base_params_temporal if sr_type == "temporal" else base_params_spatial
                    for key, value in base_params_cond.items():
                        best_params[key] = value
                    model_path = f'{parent_abs}/model_weights/fold_1/DiBiMa_eeg_{str_param}_{sr_type}_{dataset_name}_1.pth'
                    model = DiBiMa_nn(
                                    target_channels=target_channels,
                                    num_channels=in_channels,
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
                                    merge_type=best_params["merge_type"],
                                    use_label=best_params["use_label"],
                                    use_lr_conditioning=best_params["use_lr_conditioning"],
                                    use_electrode_embedding=best_params["use_electrode_embedding"],  
                    )
                    model_pl = DiBiMa_Diff(model,
                                           train_scheduler=train_scheduler,
                                           val_scheduler=val_scheduler,
                                            criterion=loss_fn,
                                            learning_rate=learning_rate,
                                            predict_type=prediction_type,  # "epsilon" or "sample"
                                            debug=debug,
                                            epochs=epochs,
                                            plot=False
                    ).to(device)
                    model_sr = load_model_weights(model_pl, model_path).to(device)
                    val_loader.dataset.model_sr = model_sr

                val_loader.dataset.input_type = input_type
                val_loader.dataset.sr_type = sr_type
                val_loader.dataset.num_channels = in_channels
                val_loader.dataset.multiplier = MULTIPLIER

                NUM_CLASSES = dataset_test.labels.unique().shape[0] #+ 1
                print(f"Training ResNet50 on {dataset_name} dataset with {input_type} {sr_type} data: in_channels={in_channels}, num_classes={NUM_CLASSES}")

                model = ResNet50(in_channels=in_channels, classes=NUM_CLASSES).to(device)
            
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                weights = torch.load(os.path.join(parent_abs, 'downstream_task', 'model_weights', f'class_weights_{dataset_name}.pt'), weights_only=False).to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=weights)
                lit_model = ResNetPL(model, optimizer, criterion)

                # Load best model
                if input_type in ["lr", "sr"]:
                    model_name = f"best_model_{dataset_name}_{input_type}_{sr_type}.pth"
                else:
                    model_name = f"best_model_{dataset_name}_hr.pth"
                lit_model.load_state_dict(torch.load(os.path.join('model_weights', model_name)))
                
                results[sr_type][input_type] = {"name": f"x{MULTIPLIER}_{sr_type}_{input_type}"}
                
                # Evaluate
                print(f"Evaluating on test set for {dataset_name} with {input_type} {sr_type} data.")
                y_true = val_loader.dataset.labels.numpy()
                y_logits, y_probs, y_preds = lit_model.predict(val_loader)
                y_logits = y_logits.cpu().numpy()
                y_probs = y_probs.cpu().numpy()
                y_preds = y_preds.cpu().numpy()
                dict_for_auc[f"x{MULTIPLIER}_{sr_type}_{input_type}"] = {"y_trues": y_true}
                dict_for_auc[f"x{MULTIPLIER}_{sr_type}_{input_type}"]["y_preds"] = y_preds
                dict_for_auc[f"x{MULTIPLIER}_{sr_type}_{input_type}"]["y_probs"] = y_probs
                dict_for_auc[f"x{MULTIPLIER}_{sr_type}_{input_type}"]["y_logits"] = y_logits
                print("Computing metrics...")
                metrics = compute_metrics(y_true, y_preds, y_probs)
                print(f"Metrics for {dataset_name} with {input_type} {sr_type}: {metrics}")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
                    results[sr_type][input_type][key] = value
                
                # Plot UMAP embeddings
                print("Generating UMAP plots...")
                plot_umap_embeddings(lit_model, val_loader, dataset_name, input_type, sr_type)

            # Plot ROC curves
            print("Plotting ROC curves...")
            plot_multiple_roc_curves(dict_for_auc, dataset_name, input_type='probs', sr_type=sr_type)

        # Save results to DataFrame
        print("Saving results to CSV...")
        rows = []
        for sr_type in results:
            for input_type in results[sr_type]:
                row = results[sr_type][input_type]
                rows.append(row)
        df = pd.DataFrame(rows)
        print(f"Final results for {dataset_name}:\n{df}")
        results_path = os.path.join(cwd, f'results_{dataset_name}.csv')
        df.to_csv(results_path, index=False)       

if __name__ == "__main__":
    main()
