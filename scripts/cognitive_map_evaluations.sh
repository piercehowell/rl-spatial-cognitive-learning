#!/bin/bash
proj_dir=$(find ~ -name "rl-spatial-cognitive-learning")
echo $proj_dir
cd $proj_dir

# python run.py mode=eval wandb.mode=online environment='gv_four_rooms.9x9_eval.yaml'\
#         landmark_spec=all_landmarks\
#         eval.hidden_layer_activation='mlp_extractor.policy_net.2'\
#         eval.evaluation_model=ppo_model_9x9_5_edges_euclidean wandb.mode='disabled'

# Example to run on all the LSTM recurrent models
python run.py -m mode=eval wandb.mode=online environment='gv_four_rooms.9x9_eval.yaml'\
        landmark_spec=all_landmarks\
        eval.hidden_layer_activation='mlp_extractor.policy_net.2'\
        eval.evaluation_model='ppo_model_9x9_5_edges_euclidean,ppo_model_9x9_All_edges_a_euclidean,ppo_model_9x9_All_edges_u_euclidean,ppo_model_9x9_A1_G1_euclidean'

# python run.py -m mode=eval wandb.mode=online environment='gv_four_rooms.9x9_eval.yaml'\
#         landmark_spec=all_landmarks model.policy_type="PPO" model.name_prefix="ppo_no_LSTM_model"\  
#         eval.hidden_layer_activation='mlp_extractor.policy_net.1'\
#         eval.evaluation_model=ppo_no_LSTM_model_9x9_5_edges_euclidean,ppo_no_LSTM_model_9x9_All_edges_a_euclidean,ppo_no_LSTM_model_9x9_All_edges_u_euclidean,ppo_no_LSTM_model_9x9_A1_G1_euclidean