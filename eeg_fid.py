from compare_to_maser import *
from downstream_task.resnet50 import ResNet50, ResNetPL
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

MULTIPLIER = 8
LR = 0.001


class EEGResNet50Extractor(nn.Module):
    def __init__(self, feature_extractor):  # Pass your self.feature_extractor
        super().__init__()
        self.feature_extractor = feature_extractor  # Assumes it has .model.features
        
    def forward(self, eeg_signals):
        """
        eeg_signals: (B, 64, 2000) or (B, 256, T) float32
        Returns: (B, 2048) double
        """
        with torch.no_grad():
            features = self.feature_extractor.model.features(eeg_signals)
            features = features.view(features.size(0), -1).double()  # (B, 2048)
        return features

class EEGFIDCalculator:
    
    def __init__(self, feature_extractor, device='cuda'):
        """
        feature_extractor: Your pretrained ResNet50 (frozen)
        """
        self.feature_extractor = feature_extractor.eval()
        self.device = device
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def calculate_fid(self, real_loader, generated_loader, device='cuda'):
       
        from scipy.linalg import sqrtm
        extractor = EEGResNet50Extractor(self.feature_extractor).eval().to(device)
        
        # Extract features for real and generated data
        real_feats = torch.cat([extractor(batch[1].to(device, dtype=torch.float32)).double().cpu() 
                            for batch in real_loader]).numpy()
        fake_feats = torch.cat([extractor(batch['eeg'].to(device, dtype=torch.float32)).double().cpu() 
                            for batch in generated_loader]).numpy()
        
        # FID computation 
        mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu_g, sigma_g = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
        
        ssdiff = np.sum((mu_r - mu_g)**2)
        
        # Handle tuple from sqrtm safely
        cov_sqrt_result = sqrtm(sigma_r.dot(sigma_g), disp=False)
        if isinstance(cov_sqrt_result, tuple):
            covmean = cov_sqrt_result[0]  # First element is sqrtm
        else:
            covmean = cov_sqrt_result
        
        # Take real part safely
        if np.iscomplexobj(covmean):
            covmean = np.real(covmean)
        
        fid_score = ssdiff + np.trace(sigma_r + sigma_g - 2 * covmean)
        
        print(f"FID: {fid_score:.4f}")
        return float(fid_score)

class GeneratedEEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, generator_model, device):
        """
        base_dataset: The original dataset to get inputs from
        generator_model: The model (e.g., Maser) to generate HR EEG signals
        device: Device to run the generator on
        """
        self.base_dataset = base_dataset
        self.generator_model = generator_model
        self.device = device

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        
        sample = self.base_dataset[idx]
        lr_eeg = sample[0].unsqueeze(0).to(self.device)  # Add batch dim
        hr_eeg = sample[1].unsqueeze(0).to(self.device)
        pos = sample[2].unsqueeze(0).to(self.device)
        label = sample[3].unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.generator_model.__class__.__name__ == 'DiBiMa_Diff':
                if self.generator_model.model.use_lr_conditioning:
                    hr_eeg_gen = self.generator_model.sample_from_lr(lr_eeg, pos=pos, label=label)
                else:
                    hr_eeg_gen = self.generator_model.sample(lr_eeg, pos=pos, label=label)
            elif type(self.generator_model).__name__ == 'Maser':
                channels = [map_mmi_channels[i] for i in case2_mmi['x4']]
                loss, nmse, ppc, hr_eeg_gen = maser(hr_eeg.permute(0, 2, 1).unsqueeze(1), unmasked_list=channels, test_flag=True, return_sr_eeg=True)
            else: # BiMa
                hr_eeg_gen = self.generator_model(lr_eeg)
            
        hr_eeg_gen = hr_eeg_gen.squeeze(0).cpu()  # Remove batch dim and move to CPU
        
        return {
            'eeg': hr_eeg_gen,
            'label': label.squeeze(0).cpu()
        }

if __name__ == "__main__":
    
    dataset_name = "mmi"
    target_channels = 64 if dataset_name == "mmi" else 62
    subject_ids = list(range(1, dict_n_patients[dataset_name] + 1))
    runs = map_runs_dataset[dataset_name]
    project_path = os.getcwd()
    data_path = project_path + os.sep + "eeg_data"
    sr_type = "spatial"
    data_folder = data_path + os.sep + dataset_name + os.sep
    batch_size = 32
    multiplier = 8
    seconds = 2

    dataset = EEGDataset(subject_ids=subject_ids, data_folder=data_folder, normalize=False, dataset_name=dataset_name, verbose=False, demo=True, is_classification=True, num_channels=target_channels, seconds=seconds)
    dataloader_train, dataloader_val, dataloader_test = prepare_dataloaders_windows(
            dataset_name, dataset, seconds=seconds, batch_size=batch_size, return_test=True
    )

    print(f"Loaded {len(dataloader_test.dataset)} samples from test data.")
    
    num_channels = len(unmask_channels[dataset_name][f'x{multiplier}'])
    dataloader_test.dataset.num_channels = num_channels
    dataloader_test.dataset.multiplier = multiplier
    dataloader_test.dataset.sr_type = sr_type   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    print("Loading Maser model...")
    maser_path = maser_project_path + sep + 'ckpt' + sep + 'last.ckpt'
    config_path = maser_project_path + sep + 'config' + sep + 'case2-4x-MM-state8-Mdep2.yml'
    maser, args = load_maser_model(config_path, maser_path)
    maser.to(device)
    maser.eval()

    print("Loading DiBiMa...")
    dibima = load_dibima_model(device)
    dibima.eval()

    print("Loading BiMa...")
    bima = load_bima_model(device)
    bima.eval()

    print("Starting FID computation...")
    # Usage
    NUM_CLASSES = len(np.unique(dataloader_test.dataset.labels.numpy()))
   
    y_true = dataloader_test.dataset.labels.numpy()
    weights = torch.load(os.path.join("downstream_task", "model_weights", f'class_weights_{dataset_name}.pt'))
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    in_channels = target_channels if sr_type == "temporal" else len(unmask_channels[dataset_name][f"x{MULTIPLIER}"])
    resnet50 = ResNet50(in_channels=target_channels, classes=NUM_CLASSES).float().to(device)
    optimizer = torch.optim.Adam(resnet50.parameters(), lr=LR)
    lit_model = ResNetPL(resnet50, optimizer, criterion)
    
    print("Loading pretrained ResNet50 for feature extraction...")
    #print(resnet50)
    resnet50_path = project_path + os.sep + "downstream_task" + os.sep + "model_weights" + os.sep + f"best_model_{dataset_name}_hr.pth"
    model_name = f"best_model_{dataset_name}_hr.pth"
    lit_model.load_state_dict(torch.load(os.path.join('downstream_task', 'model_weights', model_name)))

    fid_calculator = EEGFIDCalculator(lit_model, device='cuda')

    # Generate HR signals using Maser
    dataloader_test.dataset.multiplier = 4
    maser_generated_dataset = GeneratedEEGDataset(
        dataloader_test.dataset,
        generator_model=maser,  
        device=device
    )
    dataloader_test.dataset.multiplier = MULTIPLIER
    # Generate HR signals using BiMa
    bima_generated_dataset = GeneratedEEGDataset(
        dataloader_test.dataset,
        generator_model=bima,  
        device=device
    )
    # Generate HR signals using DiBiMa
    dibima_generated_dataset = GeneratedEEGDataset(
        dataloader_test.dataset,
        generator_model=dibima,  
        device=device
    )

    maser_generated_dataloader = torch.utils.data.DataLoader(maser_generated_dataset, batch_size=batch_size, shuffle=False)
    bima_generated_dataloader = torch.utils.data.DataLoader(bima_generated_dataset, batch_size=batch_size, shuffle=False)
    dibima_generated_dataloader = torch.utils.data.DataLoader(dibima_generated_dataset, batch_size=batch_size, shuffle=False)

    eeg_fid = fid_calculator.calculate_fid(dataloader_test, maser_generated_dataloader) 
    print(f"EEG FID Score Maser x4: {eeg_fid}")
    eeg_fid_bima = fid_calculator.calculate_fid(dataloader_test, bima_generated_dataloader) 
    print(f"EEG FID Score BiMa x{MULTIPLIER}: {eeg_fid_bima}")
    eeg_fid_dibima = fid_calculator.calculate_fid(dataloader_test, dibima_generated_dataloader) 
    print(f"EEG FID Score DiBiMa x{MULTIPLIER}: {eeg_fid_dibima}")