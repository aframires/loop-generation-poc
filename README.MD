# Loop Generation POC

A barebones PyTorch Lightning + Hydra proof-of-concept for v-prediction diffusion and rectified flow.

## Installation

To run this code:
1. Clone the repository
2. Navigate to the root
3. If you want to make modifications to the code, run `pip install -e . `, otherwise just run `pip install . `

## Usage

To run the V-Prediction and Rectified Flow code with WandB logging, run:
 ``` bash
wandb login
python3 train.py +experiment=v_prediction
python3 train.py +experiment=rectified_flow
 ``` 

## Development
This code was developed using Python 3.11 and PyTorch 2.10

## Features
- Rectified Flow and V-Prediction training
- Basic Shape Testing
- Pure CLI sampling
- WandB logging
- Hydra parameterization
- Checkpointing and resume capability


## Simplifications
- Only the Transformer backbone is available. Needs further decoupling logic for allowing a different backbone
- Only selected conditioning approaches are present. Needs further decoupling for custom conditioning
- No masking support
- Only discrete euler (RF) and DDIM (v-prediction) areprovided. No custom noise schedule. No Distribution Shift
- No evaluation code
- No EMA
- No Autoencoder present
- No WandB Audio Logging