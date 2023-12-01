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

To run a quick example training a simple navigation task using PPO, run
```bash
python examples/stablebaselines_ppo.py
```



