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

# Training

TODO: Add instructions for training the policies for experimentation

To run a quick example training a simple navigation task using PPO without saving to Weights and Biases, run
```bash
python run.py device='cuda:0' mode=train wandb.mode='disabled'
```
 

# To set custom reset function
If using miniconda environment, replace the "reset_functions.py" located in:
/home/usr/miniconda/envs/hml_gridverse/lib/python 3.11/site-packages/gym-gridverse/utils

# To use custom functions in env
You can select the reset function in "gv_four_rooms.9x9.yaml" by changing the value of "name" under the "reset function" section, the custom function are:
- Agent_1_Goal_1
- Agent_1_Goal_3
- Agent_3_Goal_3

You can also change the distance measure used under the "reward function" section, you can choose:
- manhattan
- euclidean

You can either edit or replace with the yaml provided in the following directory:
/home/usr/miniconda/envs/hml_gridverse/lib/python 3.11/site-packages/gym-gridverse/registered_envs

After that, you can proceed to use the Jupyter NB for training!
