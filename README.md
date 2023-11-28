# rl-spatial-cognitive-learning
The repo hosts the code for experiments for analyzing spatial cognitive learning in  deep reinforcement learning.

# Installation
Install Anaconda/Miniconda via their website. Then create a conda env with
```bash
conda env create -f environment.yaml
```

**Developers Note:** If you need to add or modify packages to the conda environment, first update/add to the `environment.yaml` file and then run
```bash
conda env update --file environment.yaml --prune
```