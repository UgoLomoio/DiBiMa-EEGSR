import gradio as gr
from compare_to_maser import *
from compare_maser_topomap import *
from explain_super_resolution import *

def load_eeg_dataset():
    cwd = os.getcwd()
    sep = os.sep
    dataset_path = cwd + sep + "eeg_data" + sep + "dataset_test.pth"
    dataset = torch.load(dataset_path, weights_only=False)
    return dataset

def load_eeg_sample(dataset, index):
    return dataset[index]  

def load_models(multiplier=multiplier):

    # Load models
    print("Loading BiMa model...")
    bima_model = load_bima_model(device, multiplier=multiplier)
    bima_model.eval()
    
    print("Loading DiBiMa model...")
    dibima_model = load_dibima_model(device, multiplier=multiplier)
    dibima_model.eval()
    
    # Load MASER model (for x4, adjust config if available)
    print("Loading MASER model...")
    cwd = os.getcwd()
    sep = os.sep
    maser_project_path = cwd + sep + "maser"
    
    maser_path = maser_project_path + sep + 'ckpt' + sep + 'last.ckpt'
    config_path = maser_project_path + sep + 'config' + sep + 'case2-4x-MM-state8-Mdep2.yml'
    maser_model, _ = load_maser_model(config_path, maser_path)
    maser_model.to(device)
    maser_model.eval()
    print("MASER model loaded successfully.")

    models = {
        'bima': bima_model,
        'dibima': dibima_model,
        'maser': maser_model
    }
    return models
    

def generate_sr_outputs(inputs, models, channels):
    outputs = {}
    lr, hr, pos, label = inputs  # Unpack inputs
    lr = lr.unsqueeze(0).to(device)  # Add batch dimension and move to device
    hr = hr.unsqueeze(0).to(device)  # Add batch dimension and move to device
    pos = pos.unsqueeze(0).to(device)  # Add batch dimension and move to device
    label = label.unsqueeze(0).to(device)  # Add batch dimension and move to device

    for model_name, model in models.items():
        print(f"Generating output for {model_name}...")
        with torch.no_grad():
            if model_name == "maser":
                maser_input = hr.permute(0, 2, 1).unsqueeze(1).to(device)  # Reshape to (1, 1, channels, time)
                print(f"MASER input shape: {maser_input.shape}")
                loss, nmse, ppc, output = model(maser_input, unmasked_list=channels, test_flag=True, return_sr_eeg=True)
            elif model_name == "bima":
                output = model(lr.to(device))  # Assuming inputs[0] is LR EEG
            elif model_name == "dibima":
                output = model.sample_from_lr(lr, pos=pos, label=label)  # Assuming inputs[0] is LR EEG
            outputs[model_name] = output[0].cpu()  # Store output for comparison
    return outputs
    
def plot_qualitative_comparison(idx, hr, outputs, multiplier=4):
    
    dibima_out = outputs['dibima']
    maser_out = outputs['maser']
    bima_out = outputs['bima']

    lr_bima = add_zero_channels(hr, dataset_name=dataset_name, multiplier=multiplier)
    lr_bima = lr_bima[0].cpu().detach().numpy().mean(axis=0)
    
    hr = hr.cpu().detach().numpy().mean(axis=0)
    maser_out = maser_out.cpu().detach().numpy().mean(axis=0)
    dibima_out = dibima_out.cpu().detach().numpy().mean(axis=0)
    bima_out = bima_out.cpu().detach().numpy().mean(axis=0)

    fig = plt.figure()
    plt.plot(lr_bima, color = "red", linestyle='--', linewidth=0.5, label='Masked EEG')
    plt.plot(hr, color = "green", linewidth=2, label='Original EEG')
    plt.plot(maser_out, color = "orange", linewidth=1, label=f'MASER x4 - mse {mse_loss(torch.tensor(maser_out), torch.tensor(hr)).item():.6f}')
    plt.plot(bima_out, color = "cyan", linewidth=1, label=f'BiMa x{multiplier} - mse {mse_loss(torch.tensor(bima_out), torch.tensor(hr)).item():.6f}')
    plt.plot(dibima_out, color = "purple", linewidth=1, label=f'DiBiMa x{multiplier} - mse {mse_loss(torch.tensor(dibima_out), torch.tensor(hr)).item():.6f}')
  
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Sample {idx} EEG Signal")
    plt.tight_layout()
    return fig

def plot_gradcam(inputs, bima_model):

    # Explainer 
    explainer = DiBiMaGradCam(bima_model)

    # Single explanation
    lr, hr, pos, label = inputs  # single sample
    heatmap, _ = explainer.generate_heatmap(lr, hr, pos, label=label)

    #Plot explanation
    fs_lr = fs_hr
    fig = explainer.plot_explanation(lr, heatmap, label, fs=fs_lr)
    return fig

def demo_fn(index, scale):

    global dataset 

    multiplier = int(scale[1:])  # Extract numeric part from "x4" -> 4
    target_channels_names = map
    target_channels = [map_mmi_channels[i] for i in case2_mmi['x4']]
    dataset.sr_type = sr_type
    dataset.multiplier = multiplier
    dataset.num_channels = len(unmask_channels[dataset_name][f"x{multiplier}"])
    print(f"Dataset configured for {dataset.sr_type} super-resolution with multiplier x{dataset.multiplier}")
    
    lr, hr, pos, label = load_eeg_sample(dataset, index)
    print(f"Loaded EEG sample with shapes - LR: {lr.shape}, HR: {hr.shape}, Pos: {pos.shape}, Label: {label.shape}")
    
    models = load_models(multiplier=multiplier)
    print("Models loaded and ready for inference.")
    #for name, model in models.items():
    #    print(f"{name} model summary:")
    #    print(model)
    print(models['bima'])
    
    inputs = [lr, hr, pos, label] 
    outputs = generate_sr_outputs(inputs, models, target_channels)
    print("Generated super-resolution outputs for all models.")

    qual = plot_qualitative_comparison(index, hr, outputs, multiplier=multiplier)
    grad = plot_gradcam(inputs, models['bima'])
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    target_channel_names = np.array(dataset.channel_names)
    our_models = {"bima": models['bima'], "dibima": models['dibima']}
    psd_topo = plot_topomap_comparison(
        our_models, 
        models['maser'],
        dataloader_test, 
        multiplier=multiplier,
        index=index, 
        fs_hr=fs_hr,
        dataset_name=dataset_name,
        target_channel_names=target_channel_names
    )
    return qual, grad, psd_topo



fs_hr = 160
seconds = 2 
dataset_name = "mmi"
sr_type = "spatial"
dataset = load_eeg_dataset()  

with gr.Blocks(title="BiMa-EEGSR Demo") as demo:
    gr.Markdown("# BiMa-EEGSR: EEG Spatial Super-Resolution Demo")
    gr.Markdown("Select dataset index and scale factor to compare models.")
    
    with gr.Row():
        index = gr.Slider(0, len(dataset) - 1, value=0, step=1, label="Dataset Index")
        scale = gr.Dropdown(["x2", "x4", "x8"], value="x4", label="Super-Resolution Scale")
    run = gr.Button("Generate Outputs")
    
    with gr.Tabs():
        with gr.TabItem("Qualitative Comparison"):
            qual_plot = gr.Plot(label="Model Outputs")
        with gr.TabItem("GradCAM Explanations"):
            grad_plot = gr.Plot(label="Explainability Heatmaps")
        with gr.TabItem("PSD Topomaps"):
            psd_plot = gr.Plot(label="Power Spectral Density Maps")

    run.click(demo_fn, inputs=[index, scale], outputs=[qual_plot, grad_plot, psd_plot])

if __name__ == "__main__":

    demo.launch()
    