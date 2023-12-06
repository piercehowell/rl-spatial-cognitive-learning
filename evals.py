"""
"""
from torch import nn
import torch
from typing import List, Dict, Tuple, Callable, Union
import gym
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import matplotlib.pyplot as plt
import os
from utils import make_env_from_data
from plotting import plot_cognitive_map
import yaml


# directory of run.py
script_path = os.path.dirname(os.path.abspath(__file__))


def make_env_for_landmark_setting(landmark_start, landmark_goal, cfg):
    """
    Make the environment with a specific landmark start and goal
    """
    path = os.path.join(script_path, 'environments', cfg.environment)
    with open(path) as f:
        data = yaml.safe_load(f)

    data['reset_function']['landmark_start'] = landmark_start
    data['reset_function']['landmark_goal'] = landmark_goal
    return make_env_from_data(data, cfg)

# TODO: Clean up imports
class HiddenLayerExtractor(nn.Module):
    """
    Wrapper for the neural network policy that when called
    on a forward pass, outputs both the action and the requested
    hidden layer activation.
    """
    def __init__(self, policy: nn.Module, hidden_layer_name: str):
        super(HiddenLayerExtractor, self).__init__()
        self.policy = policy
        self.hidden_layer_name = hidden_layer_name
        self.hidden_layer_output = None


        # register a hook to get the hidden layer
        for name, module in self.policy.policy.named_modules():
            print(f"Module name: {name}")
            if name == hidden_layer_name:
                print(f"HiddenLayerExtractor registering output of module {name}")
                module.register_forward_hook(self._hidden_layer_hook)
                break
        
    
    def _hidden_layer_hook(self, module, input, output):
        """
        Creates a hook that logs the intermediate output
        """
        # if module.__class__.__name__ == self.hidden_layer_name:
        self.hidden_layer_output = output.detach()
        # print(self.hidden_layer_output)

    def forward(self, obs, state=None, deterministic=True):
        """
        Forward pass
        """
        # with torch.no_grad():
        episode_starts = torch.tensor(np.ones((1,), dtype=bool))
        obs = torch.tensor(obs)
        action, lstm_states = self.policy.predict(obs, state, episode_starts)
        return action, self.hidden_layer_output, lstm_states
    

class CognitiveMapEvaluation(nn.Module):
    """
    Applies the MDS Cognitive Mapping algorithm to hidden layers
    of a policy at prespecified landmarks.
    """
    def __init__(self, cfg, env: gym.Env, landmarks: Dict[str, List], goal_ldmrk: Dict[str, List], policy: nn.Module, dist_func: Callable=cosine_distances, seed: int=1):
        """
        """
        super(CognitiveMapEvaluation, self).__init__()

        self.env = env
        self._landmarks = landmarks

        # wrap the policy in a module to extract a hidden layer from the policy
        self.policy = HiddenLayerExtractor(policy, 'mlp_extractor.policy_net')
        self._goal_ldmrk = goal_ldmrk
        self._dist_func = dist_func
        self.hidden_layer_activations = []
        self._seed = seed
        self._cfg = cfg

    def forward(self):
        """Run the evaluate an output the cognitive map and corresponding meta-data"""

        ldmrk_locs = []
        goal_ldmrk_name = list(self._goal_ldmrk.keys())[0]
        goal_ldmrk_loc = self._goal_ldmrk[goal_ldmrk_name]
        for ldmrk_name, ldmrk_loc in self._landmarks.items():
            ldmrk_locs.append(list(ldmrk_loc))
            # reset the environment so that the agent begins at the landmark location
            # pass in the starting point and goal ldmrk
            # TODO: Update the environment such that on reset we can set the agents
            # starting landmark and goal landmark

            # remake the environment
            env = make_env_for_landmark_setting(ldmrk_loc, goal_ldmrk_loc, self._cfg)

            obs_at_ldmrk = env.reset()

            lstm_states = None

            # take an action and get the requested hidden layer of that activation
            action, hidden_layer_activation, lstm_states = self.policy(obs_at_ldmrk, lstm_states)
            print(f"Hidden Layer activation: {hidden_layer_activation}")
            # save the hidden_layer_activation
            self.hidden_layer_activations.append(hidden_layer_activation)
        
        # compute the distance matrix for the hidden layer activations
        dist_matrix = hidden_layers_to_distance_matrix(self.hidden_layer_activations, dist_func=self._dist_func)


        # generate the cognitive map and return it (as a figure)
        cog_map, disparity = mds_cognitive_mapping(dist_matrix, np.array(ldmrk_locs), self._seed)

        # Save the cognitive map to wandb or something else.
        return cog_map,
    
    @property
    def goal_ldmrk(self):
        return self._goal_ldmrk

    @goal_ldmrk.setter
    def goal_ldmrk(self, value: Dict[str, Tuple]):
        self._goal_ldmrk = value

    @property
    def landmarks(self):
        return self._landmarks
    
    @landmarks.setter
    def landmarks(self, value: List[Dict[str, Tuple]]):
        self._landmarks = value

            
def hidden_layers_to_distance_matrix(hidden_layer_activations_struct: List[torch.tensor], dist_func: Callable=cosine_distances):
    """
    Builds a distance matrix between the pairwise distances of hidden layer activations.
    The output distance matrix is NxN, where N is the number of landmarks
    """
    # TODO: Make this function faster with tensorization.
    N = len(hidden_layer_activations_struct)
    dist_matrix = np.zeros((N, N))
    for i, hidden_layer_activation_i in enumerate(hidden_layer_activations_struct):
        for j, hidden_layer_activation_j in enumerate(hidden_layer_activations_struct):
            dist_matrix[i, j] = dist_func(hidden_layer_activation_i, hidden_layer_activation_j)
    return dist_matrix


def scale_data_min_max(embeddings, true_ldmrk_locs):
    """
    Scale the 2D embeddings such that the distances between the
    embeddings match the distances between the true landmark locations.
    """
    x_min_a, x_max_a = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    y_min_a, y_max_a = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    x_min_b, x_max_b = np.min(true_ldmrk_locs[:, 0]), np.max(true_ldmrk_locs[:, 0])
    y_min_b, y_max_b = np.min(true_ldmrk_locs[:, 1]), np.max(true_ldmrk_locs[:, 1])
    # scale the embeddings
    embeddings[:, 0] = (embeddings[:, 0] - x_min_a) / (x_max_a - x_min_a)
    embeddings[:, 1] = (embeddings[:, 1] - y_min_a) / (y_max_a - y_min_a)
    
    # scale the true landmark locations
    true_ldmrk_locs[:, 0] = (true_ldmrk_locs[:, 0] - x_min_b) / (x_max_b - x_min_b)
    true_ldmrk_locs[:, 1] = (true_ldmrk_locs[:, 1] - y_min_b) / (y_max_b - y_min_b)
    return embeddings, true_ldmrk_locs 

def shift_embedding_to_reference(embeddings, true_ldmrk_locs, reference_idx=0):
    """
    Scale the embeddings such that the embedding at reference idx is
    centered at zero, zero. Then scale the embeddings
    """
    x_ref_a, y_ref_a = embeddings[reference_idx, 0], embeddings[reference_idx, 1]
    x_ref_b, y_ref_b = true_ldmrk_locs[reference_idx, 0], true_ldmrk_locs[reference_idx, 1]
    embeddings += (-np.array((x_ref_a, y_ref_a)) + np.array((x_ref_b, y_ref_b)))

    return embeddings, true_ldmrk_locs

def mds_cognitive_mapping(cog_map_dist_matrix, true_ldmrk_locs, seed=1):
    """
    Perform multi-dimensional scaling on the 'cog_map_dist_matrix' 
    which is an NxN matrix containing distances between N landmark 
    vectors. Scale the outputs of MDS such that the distances match
    those of the true distance matrix.
    
    Cite: https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
    """

    # metric meds
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )

    pos = mds.fit(cog_map_dist_matrix).embedding_

    # Procrusted will normalize the true ldmrk locations
    # store this data so we can renormalized back to the 
    # scale of the map.
    mean = np.mean(true_ldmrk_locs, 0)
    norm = np.linalg.norm(true_ldmrk_locs-mean)

    # apply prcrusted (rotation, scaling, reflection)
    true_ldmrk_locs_normed, pos_normed, disparity = procrustes(true_ldmrk_locs, pos)
    
    # rescale the output to map it back into the environment/map size
    true_ldmrk_locs = (true_ldmrk_locs_normed * norm) + mean
    pos = (pos_normed * norm) + mean 

    # shift the data such that the reference points line up.
    pos, true_ldmrk_locs = shift_embedding_to_reference(pos, true_ldmrk_locs, reference_idx=0)
    
    # TODO: this is a measure of how dissimlar they are; log this!!!
    print(disparity)

    fig = plot_cognitive_map(true_ldmrk_locs, pos)
    return fig, disparity

        