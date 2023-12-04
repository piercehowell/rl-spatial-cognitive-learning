# rl-spatial-cognitive-learning
# To use custom reset function
If using miniconda environment, replace the "reset_functions.py" located in:
/home/usr/miniconda/envs/hml_gridverse/lib/python 3.11/site-packages/gym-gridverse/utils

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
