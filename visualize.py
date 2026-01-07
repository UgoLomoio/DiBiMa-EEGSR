from operator import gt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models import DiBiMa_nn
from utils import generate_colors
import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact, IntSlider
from utils import add_zero_channels

cwd = os.path.dirname(os.path.abspath(__file__))
sep = os.sep

#from .maser import inference
maser_project_path = cwd + sep + "maser"

if maser_project_path not in sys.path:
    sys.path.append(maser_project_path)

# Now import by filename (no dot prefix)
from inference import load_model

def plot_mean_timeseries_plotly(timeseries, channel_to_plot=None, save_path=None):
    """
    timeseries: dict key -> array (B, C, L) or (C, L)
    All arrays must have same C, L. We view channel i across all keys.
    """

    # Normalize shapes to (C, L)
    processed = {}
    any_key = None
    for key, value in timeseries.items():
        arr = np.asarray(value)
        if arr.ndim == 3:
            arr = arr[0]  # (C, L)
        if arr.ndim != 2:
            raise ValueError(f"{key} has shape {arr.shape}, expected (C, L) or (B, C, L)")
        #add zero channels to LR if needed (spatial SR case)
        if arr.shape[0] < 64:
            processed[key] = add_zero_channels(torch.tensor(arr).unsqueeze(0)).squeeze(0).numpy()
        else:
            processed[key] = arr
        if any_key is None:
            any_key = key
      
    C, L = processed[any_key].shape

    # Check consistency
    if channel_to_plot is None:
        for key, arr in processed.items():
            if arr.shape != (C, L):
                raise ValueError(f"{key} has shape {arr.shape}, expected {(C, L)}")

    # Precompute per-channel data: channel i -> dict(key -> 1D array)
    channel_data = []
    for c in range(C):
        if channel_to_plot is not None:
            if c == channel_to_plot:
                ch_dict = {key: arr[c] for key, arr in processed.items()}
                channel_data.append(ch_dict)
            else:
                continue
        else:
            ch_dict = {key: arr[c] for key, arr in processed.items()}
            channel_data.append(ch_dict)        
    

    #print(f"Prepared data for {len(channel_data)} channels, each with length {L}.")
    if channel_to_plot is not None:
        C = len(channel_data)

    if channel_to_plot is None:
        channel_to_plot = 0  # Default to first channel

    # Interactive plot (works in Jupyter / notebook)
    def _plot_channel(c_idx):
        fig = go.Figure()
        for key, arr in channel_data[c_idx].items():
            fig.add_trace(go.Scatter(
                y=arr,
                mode="lines",
                name=key
            ))
        fig.update_layout(
            title=f"Channel {c_idx} across all time series",
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude",
            template="plotly_white"
        )
        fig.show()

        if save_path:
            # Save last shown figure as static image (requires kaleido)
            fig.write_image(save_path)
            print(f"Saved figure to {save_path}")

    
    interact(_plot_channel, c_idx=IntSlider(min=0, max=C-1, step=1, value=0))

def plot_mean_timeseries(timeseries, save_path=None):
        
    fig = plt.figure(figsize=(12, 4))
    for key, value in timeseries.items():
        if value.ndim == 3:
            value = value[0]  # Take the first sample in the batch
        print(f"Plotting {key} with shape {value.shape}, min {value.min()}, max {value.max()}")
        mean_signal = np.mean(value, axis=0)  # Mean across channels
        plt.plot(mean_signal, label=key)
    plt.title(f'Mean Timeseries Across Channels')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    

def plot_physionet_style(timeseries, save_path="physionet_style_visualization.png",
                         dataset_name="PhysioNet MMI"):
    """
    timeseries: dict like
      {
        16: {"gt": gt_16, "maser": maser_16, "dcae": dcae_16},   # (64, 1600)
        32: {"gt": gt_32, "dcae": dcae_32},                      # MASER optional
        ...
      }
    """

    n_scales = len(timeseries)
    print(f"Plotting PhysioNet-style visualization for {n_scales} scales...")

    color_map = {}
    colors = generate_colors(n_colors=20, method="tab20")

    i = 0
    for name in timeseries[list(timeseries.keys())[0]].keys():
        if "gt" in name.lower():
            color_map["gt"] = ("green", "Ground Truth")
        elif "maser" in name.lower():
            color_map["maser"] = ("blue", "MASER")
        elif "dcae" in name.lower():
            color_map["dcae"] = ("red", "DCAE-SR")
        else:
            color = colors[i]
            color_map[name] = (color, name)
            i += 1
    print("Color map:", color_map.keys())

    fig, axes = plt.subplots(
        n_scales, 1,
        figsize=(6, 5),
        sharex=True,
        sharey=True
    )
    if n_scales == 1:
        axes = [axes]

    labels_used = set()
    for ax, (nchs, dict_) in zip(axes, timeseries.items()):
        # dict_[key] has shape (64, 1600); take mean over channel dimension
        for key, arr in dict_.items():
            if arr.ndim == 3:
                arr = arr[0]  # (1, 64, 1600) -> (64, 1600)

            print(f"Processing {key} for {nchs} channels...")
            print("Array shape:", arr.shape)
            if key not in color_map:
                continue

            color, label = color_map[key]

            # mean over 64 channels -> (1600,)
            mean_ts = arr.mean(axis=0)

            x_axis = np.arange(mean_ts.shape[0])

            if label not in labels_used:
                print(f"Plotting {label} for {nchs} channels...")
                print(x_axis.shape, mean_ts.shape, color, label)
                ax.plot(x_axis, mean_ts, color=color, label=label, linewidth=1)
                labels_used.add(label)
            else:
                print(f"Plotting {label} for {nchs} channels...")
                print(x_axis.shape, mean_ts.shape, color, label)
                ax.plot(x_axis, mean_ts, color=color, linewidth=1)

        ax.set_ylabel(f"{nchs}→64", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Sampling Point", fontsize=10)
    fig.suptitle(dataset_name, fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=min(len(labels), 3), fontsize=9)
    plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.12, hspace=0.15)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def load_test_data(test_data_path):
    dataset = torch.load(test_data_path, weights_only=False)
    return dataset

def load_dcae(checkpoint_path, model_class, *model_args, **model_kwargs):
    """
    Loads a DCAE model from various checkpoint formats.
    model_class: your DCAE class, e.g. DiBiMa_nn
    *model_args/**model_kwargs: args to construct the model
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
    # Case 1: checkpoint is already a pure state_dict (OrderedDict)
    if not isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        # Try common keys in order
        for key in ["model_state_dict", "state_dict", "net", "model"]:
            if key in ckpt:
                state_dict = ckpt[key]
                break
        else:
            # Fallback: maybe the dict itself is the state dict (all tensor values)
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt
            else:
                raise KeyError(
                    f"No state_dict-like key found in checkpoint. "
                    f"Available keys: {list(ckpt.keys())}"
                )

    model = model_class(*model_args, **model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("DCAE loaded. Missing keys:", missing, "Unexpected keys:", unexpected)
    return model


nums_channels = [8, 16, 32]

if __name__ == "__main__":

    config_path = maser_project_path + sep + 'config/case2-4x-MM-state8-Mdep2.yml'
    checkpoint_path = maser_project_path + sep + 'ckpt/last.ckpt'
    test_data_path = maser_project_path + sep + 'data/test_data.dat'

    maser, args = load_model(config_path, checkpoint_path)

    timeseries = {}
    
    for num_ch in nums_channels:
        
        print(f"Visualizing for {num_ch} channels...")
        test_data_path = cwd + sep + 'eeg_data' + sep + 'preprocessed' + sep + f'test_sr_spatial_{num_ch}to64chs_1.pt'
        eegs = load_test_data(test_data_path)
        eegs = DataLoader(eegs, batch_size=args.batch_size, shuffle=False, num_workers=0)  
        lr, gt = next(iter(eegs))
        lr, gt = lr[0], gt[0]  # Take first sample in batch
        print("LR shape:", lr.shape)
        print("GT shape:", gt.shape)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dcae_model_path = cwd + sep + 'model_weights' + sep + 'fold_1' + sep + f'DiBiMa_eeg_with_subpixel_and_both_residuals_{num_ch}to64chs_1.pth'
        model_args = {
            "num_channels": num_ch, 
            "fs_lr": 160, 
            "fs_hr": 160, 
            "seconds": 10, 
            "residual_global": True, 
            "residual_internal": True, 
            "use_subpixel": True, 
            "sr_type": "spatial"
        }
        model_args = tuple(model_args.values())
        
        dcae = load_dcae(dcae_model_path, DiBiMa_nn, *model_args)
        dcae.to(device)

        print("Running DCAE inference...")
        dcae_out = dcae(lr.unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy()
        
        unmask_list = args.unmasked_list   
        print("Running MASER inference...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        maser.to(device)
        #LR shape: torch.Size(16, 1600)
        lr_maser = lr.unsqueeze(0) # (1, 16, 1600)
        print("LR MASER initial shape:", lr_maser.shape)
        lr_maser = lr_maser.swapaxes(1, 2).to(device)  # (1, 1600, 16)
        lr_maser = lr_maser.unsqueeze(0)  # (1, 1, 1600, 16)
        print("LR MASER shape:", lr_maser.shape)
        #dataloader_maser = DataLoader(lr_maser, batch_size=1, shuffle=False, num_workers=0)
        loss, nmse, ppc, maser_out = maser(lr_maser, unmask_list, test_flag=True, return_sr_eeg=True)  
        maser_out = maser_out.T.detach().cpu().numpy()        

        timeseries[num_ch] = {
            "dcae": dcae_out,
            "gt": gt.numpy(),
            "input": lr.numpy()
        }    
        if num_ch == 16:
            timeseries[num_ch]["maser"] = maser_out
            #continue  # MASER part commented out for now

    save_path = cwd + sep + "imgs" + sep + "physionet_style_visualization.png"
    plot_physionet_style(timeseries, save_path=save_path, dataset_name="PhysioNet MMI")



    #MASER SIGNALS ARE 320, 64, MAYBE THEY ARE 2 SECS LONG?