# DiBiMa-EEGSR

Code for the reproduction of results obtained in "A Bidirectional Mamba-2 Diffusion model for spatio-temporal EEG super-resolution"

## 🛠️ Third party softwares needed
- Anaconda
- Git

## 🚀 Environment Setup

To prepare the conda environment, open the terminal and navigate inside the EEGSR directory:


    conda create -n eegsr python=3.13
    conda activate eegsr
    pip install -r requirements.txt

## 🧪 Ablations

Ablations on Mamba Hyperparameters:
    
    conda activate eegsr
    python ablations_mambadim.py

Ablations: DDPM Diffusion & Mamba-2 To assess the impact of DDPM diffusion and Mamba-2 on EEG spatial and temporal super-resolution:

    conda activate eegsr
    python ablations_diff_mambadim.py

## 🎯 Training & Validation of the final model: 
    conda activate eegsr
    python train.py
    python test.py

## 📊 Downstream EEG Classification Task Available in downstream_task/ directory.

## 📋 To-Do Status

🧪 Ablations              
❌ Final model training
❌ Downstream-task

Legend:
🧪 Ongoing - ✅ Done - ❌ Todo