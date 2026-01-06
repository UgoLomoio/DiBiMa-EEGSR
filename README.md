Code for the reproduction of results obtained in "A Bidirectional Mamba-2 Diffusion model for spatio-temporal EEG super-resolution"

Third party softwares needed:
- Anaconda
- Git 

To prepare the conda enviroment, open the terminal and navigate inside the EEGSR directory

    conda create -n eegsr python=3.13
    conda activate eegsr
    pip install -r requirements.txt

To perform ablations on mamba hyperparams:

    conda activate eegsr 
    python ablations_mambadim.py

To perform ablations to assess the impact of DDPM diffusion and Mamba-2 on EEG spatial and temporal super-resolution:

    conda activate eegsr
    python ablations_diff_mamba.py


To train and validate the final model:

    conda activate eegsr
    python train.py

To test the model: 

    conda activate eegsr
    python test.py


Downstream classification task available  "downstream_task" directory.