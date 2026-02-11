import gc
import os
import sys
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils_class import load_model_weights, EEGDatasetClassification
import pandas as pd
from resnet50 import ResNet50, ResNetPL  
from metrics_class import compute_metrics, plot_multiple_roc_curves
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

current_dir = os.getcwd()
parent_dir = os.path.join(current_dir, os.pardir)  # or os.path.dirname(current_dir)
parent_abs = os.path.abspath(parent_dir)
print(f"Adding to sys.path: {parent_abs}")
sys.path.insert(0, parent_abs)
from test import base_params_temporal, base_params_spatial, base_params_cond, unmask_channels, loss_fn, learning_rate, debug, epochs, train_scheduler, val_scheduler, prediction_type
from models import DiBiMa, DiBiMa_Diff, DiBiMa_nn

cwd = os.getcwd()

os.makedirs('model_weights', exist_ok=True)

# Data directory
DATA_DIR = os.path.join(parent_dir, 'eeg_data')

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 1    # Adjust based on your system
MULTIPLIER = 8
LR = 1e-4
MAX_EPOCHS = 30

demo = False
seed = 2
dict_n_patients = {
    "mmi": 109,
    "seed": 15
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = DATA_DIR

if demo:
    MAX_EPOCHS = 50

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
                  
        dataset_train = EEGDatasetClassification(train_ids, dataset_path, dataset_name=dataset_name, model_sr=model_sr, demo=demo)
        dataset_test = EEGDatasetClassification(test_ids, dataset_path, dataset_name=dataset_name, model_sr=model_sr, demo=demo) 

        train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
               
        seconds = 2
        results = {}
        dict_for_auc = {}

        for i, sr_type in enumerate(["spatial"]):
            results[sr_type] = {}
            for input_type in ["hr", "lr", "sr"]:
                if input_type in ["lr", "sr"]:
                    model_name = f"best_model_{dataset_name}_{input_type}_{sr_type}.pth"
                else:
                    model_name = f"best_model_{dataset_name}_hr.pth"
                model_path = os.path.join('model_weights', model_name)
        
                results[sr_type][input_type] = {}
                if input_type == "hr" and i >= 1:
                    print(f"Skipping {input_type} {sr_type} since it's the same as HR.")
                    continue
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
                    model_sr_path = f'{parent_abs}/model_weights/fold_1/BiMa_eeg_{str_param}_{sr_type}_{dataset_name}_1.pth'
                    
                    best_params = base_params_temporal if sr_type == "temporal" else base_params_spatial
                    for key, value in base_params_cond.items():
                        best_params[key] = False #value
                    best_params["use_diffusion"] = False

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
                                    merge_type=best_params["merge_type"],
                                    use_positional_encoding=False,
                                    use_electrode_embedding=best_params["use_electrode_embedding"],  
                                    use_label=best_params["use_label"],
                                    use_lr_conditioning=best_params["use_lr_conditioning"]
                    )
                    #model_sr = DiBiMa_Diff(model,
                    #                        criterion=loss_fn,
                    #                        train_scheduler=train_scheduler,
                    #                        val_scheduler=val_scheduler,
                    #                        learning_rate=learning_rate,
                    #                        predict_type=prediction_type,  # "epsilon" or "sample"
                    #                        debug=debug,
                    #                        epochs=epochs,
                    #                        plot=False
                    #).to(device)
                    model = load_model_weights(model, model_sr_path).to(device)
                    model_sr = DiBiMa(model, dataset_name=dataset_name, multiplier=MULTIPLIER).to(device)
                    train_loader.dataset.model_sr = model_sr
                    val_loader.dataset.model_sr = model_sr

                train_loader.dataset.input_type = input_type
                train_loader.dataset.sr_type = sr_type
                val_loader.dataset.input_type = input_type
                val_loader.dataset.sr_type = sr_type
                train_loader.dataset.num_channels = in_channels
                val_loader.dataset.num_channels = in_channels
                train_loader.dataset.multiplier = MULTIPLIER
                val_loader.dataset.multiplier = MULTIPLIER
                
                NUM_CLASSES = len(np.unique(dataset_train.labels.numpy()))
                print(f"Training ResNet50 on {dataset_name} dataset with {input_type} {sr_type} data: in_channels={in_channels}, num_classes={NUM_CLASSES}")

                y_true = val_loader.dataset.labels.numpy()
                weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
                weights = torch.tensor(weights).float().to(device)

                torch.save(weights, os.path.join('model_weights', f'class_weights_{dataset_name}.pt'))
                criterion = torch.nn.CrossEntropyLoss(weight=weights)

                model = ResNet50(in_channels=in_channels, classes=NUM_CLASSES).float().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                lit_model = ResNetPL(model, optimizer, criterion)
                
                # Logger and checkpoint
                checkpoint = ModelCheckpoint(monitor='avg_val_loss', mode='min', save_top_k=1, dirpath='checkpoints/')
                if os.path.exists(model_path):
                    print(f"Model {model_name} already exists. Skipping training for {input_type} {sr_type}.")  
                    lit_model.load_state_dict(torch.load(model_path))
                else:
                    # Trainer
                    trainer = pl.Trainer(
                        max_epochs=MAX_EPOCHS,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        devices=1,
                        callbacks=[checkpoint],
                        log_every_n_steps=10
                    )

                    # Train
                    trainer.fit(lit_model, train_loader, val_loader)
                    print(f"Best model saved at: {checkpoint.best_model_path}")

                    # Load best model
                    lit_model = ResNetPL.load_from_checkpoint(checkpoint.best_model_path, model=model, optimizer=optimizer, criterion=criterion)
                    torch.save(lit_model.state_dict(), model_path)
                
                results[sr_type][input_type] = {"name": f"x8_{sr_type}_{input_type}"}
                
                # Evaluate
                print(f"Evaluating on test set for {dataset_name} with {input_type} {sr_type} data.")
                y_true = val_loader.dataset.labels.numpy()
                y_logits, y_probs, y_preds = lit_model.predict(val_loader)
                y_logits = y_logits.cpu().numpy()
                y_probs = y_probs.cpu().numpy()
                y_preds = y_preds.cpu().numpy()
                dict_for_auc[f"x8_{sr_type}_{input_type}"] = {"y_trues": y_true}
                dict_for_auc[f"x8_{sr_type}_{input_type}"]["y_preds"] = y_preds
                dict_for_auc[f"x8_{sr_type}_{input_type}"]["y_probs"] = y_probs
                dict_for_auc[f"x8_{sr_type}_{input_type}"]["y_logits"] = y_logits
                print("Computing metrics...")
                metrics = compute_metrics(y_true, y_preds, y_probs)
                print(f"Metrics for {dataset_name} with {input_type} {sr_type}: {metrics}")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
                    results[sr_type][input_type][key] = value

            del model, lit_model, model_sr
            torch.cuda.empty_cache()
            gc.collect()

        # Plot ROC curves
        plot_multiple_roc_curves(dict_for_auc, dataset_name, input_type='probs')

        # Save results to DataFrame
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
