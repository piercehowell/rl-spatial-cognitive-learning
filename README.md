# rl-spatial-cognitive-learning

The repo hosts the code for experiments for analyzing spatial cognitive learning in  deep reinforcement learning.

# Installation
Install Anaconda/Miniconda via their website. Then create a conda env with
```bash
# create the environment
conda env create -f environment.yaml

# activate the environment
conda activate rl_spatial_cog
```

**Developers Note:** If you need to add or modify packages to the conda environment, first update/add to the `environment.yaml` file and then run
```bash
conda env update --file environment.yaml --prune
```

Install [gym-gridverse](https://github.com/abaisero/gym-gridverse) manually.
```bash
git clone https://github.com/abaisero/gym-gridverse.git
cd gym-gridverse
pip install -e .
```

# Usage Overview
## Quick Start
Both training and evaluations are run from the `run.py` script in the repo's home directory. For example:
```bash
# default training
python run.py mode=train

# default evaluation
python run.py mode=eval
```
## Setting Configurations with Hydra
You will notice that parameter (like `mode`) can be specified at the command line to control the functionality of experiments. This repository uses [Hydra](https://hydra.cc/docs/intro/), a python library for that enables hierarchical configuration (e.g. parameters) composition and override. This makes experimentation for different parameters and settings as simple as changing parameters in the defined configuration files or at the command line. **The configuration files for this project are located in the directory [`conf`](https://github.com/piercehowell/rl-spatial-cognitive-learning/tree/main/conf)**. 

You can view all of the default parameters without running the code by running
```bash
python run.py --cfg=job
```

## Training
To run a quick example training a simple navigation task using PPO without saving to Weights and Biases, run
```bash
python run.py device='cuda:0' mode=train wandb.mode='online' model.policy_type=RecurrentPPO model.name_prefix='ppo_model_9x9' model.save_freq=1000 model.total_timesteps=2000000 model.n_steps=2048
```

## Evaluation
Evaluations are performed on pre-trained models, where the model checkpoints are stored in a directory under `evaluation_models`. The folder should contain model checkpoints saved as `.zip` files (as is the standard format for saving model checkpoints in stable_baselines3. Here is an example of running the `CognitiveMapEvaluation` which tries to build the agent's internal representation of the environment from its hidden layers.

```bash
python run.py mode=eval wandb.mode=online environment='gv_four_rooms.9x9_eval.yaml' eval.evaluation_model=recurrent_ppo_test eval.hidden_layer_activation=mlp_extractor.policy_net.1 landmark_spec=all_landmarks
```
The above command will save the results from the evaluation to wandb.
 

## To set custom reset function
Custom reset functions can be written in the `custom_reset_functions.py` script. It is recommend to also refer to the gridverse documentation on [reset functions](https://gym-gridverse.readthedocs.io/en/latest/tutorial/customization/reset_functions.html?highlight=reset#reset-functions).
