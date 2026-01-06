
import os 
import torch
from resnet50 import ResNet50
from pytorch_lightning import Trainer
from metrics import validate_model  

cwd = os.path.dirname(os.path.abspath(__file__))
sep = os.sep

input_types = ["lr", "hr", "sr-temporal", "sr-spatial"]

datapath = cwd + sep + "data" 
ckptpath = cwd + sep + "ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(datapath, exist_ok=True)
os.makedirs(ckptpath, exist_ok=True)

epochs = 100
batch_size = 16
num_classes = 2
learning_rate = 0.001

if __name__ == "__main__":
    
    print("Data path:", datapath)
    print("Checkpoint path:", ckptpath)

    to_df = []

    for it in input_types:

        print(f"Preparing filenames for input type: {it}")
        
        if "sr" in it:
            train_filename = f"{datapath}{sep}train_{it}.pt"
            val_filename = f"{datapath}{sep}val_{it}.pt"
            num_channels = 64 if "spatial" in it else 16
            upscale_factor = 4 if "temporal" in it else 1
        else:
            str_it = f"x{upscale_factor}" if "temporal" in it else f"{num_channels}to64chs"
            train_filename = f"{datapath}{sep}train_sr_{str_it}.pt"
            val_filename = f"{datapath}{sep}val_sr_{str_it}.pt"
            num_channels = 64
            upscale_factor = 1
        
        train_dataset = torch.load(train_filename)
        val_dataset = torch.load(val_filename)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = ResNet50(num_channels=num_channels, num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        trainer = Trainer(max_epochs=epochs, gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(model, train_dataloader, val_dataloader)

        if "sr" in it:
            model_name = f"ResNet50_SR_{str_it}"
        else:
            model_name = f"ResNet50_{it}"
        ckpt_filename = f"{ckptpath}{sep}{model_name}.ckpt"

        trainer.save_checkpoint(ckpt_filename)
        print(f"Model checkpoint saved at: {ckpt_filename}")

        fig_cm, metrics, summary = validate_model(model, val_dataloader, device, model_name=model_name, classes={0: "Class0", 1: "Class1"})
        print(f"Validation Summary for input type {it}:\n{summary}")
        
        fig_cm_filename = f"{cwd}{sep}confusion_matrix_{it}.html"
        fig_cm.write_html(fig_cm_filename)
        print(f"Confusion matrix saved at: {fig_cm_filename}")

        to_df.append({
            "input_type": it,
            "checkpoint": ckpt_filename,
        })
        for key, value in metrics.items():
            to_df[-1][key] = value

    import pandas as pd
    results_df = pd.DataFrame(to_df)
    results_csv = f"{cwd}{sep}classification_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"All results saved to: {results_csv}")
