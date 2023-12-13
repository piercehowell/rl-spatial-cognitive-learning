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
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import matplotlib.pyplot as plt
import os
from utils import make_env_from_data
from plotting import plot_cognitive_map
import yaml
from copy import deepcopy


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
            # print(f"Module name: {name}")
            if name == hidden_layer_name:
                # print(f"HiddenLayerExtractor registering output of module {name}")
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
        with torch.no_grad():
            episode_starts = torch.tensor(np.ones((1,), dtype=bool))
            obs = torch.tensor(obs)
            action, lstm_states = self.policy.predict(obs, state, episode_starts, deterministic=deterministic)
        return action, self.hidden_layer_output, lstm_states
    

class CognitiveMapEvaluation(nn.Module):
    """
    Applies the MDS Cognitive Mapping algorithm to hidden layers
    of a policy at prespecified landmarks.
    """
    def __init__(self, cfg, env: gym.Env, landmarks: Dict[str, List], A: np.array, policy: nn.Module, dist_func: Callable=cosine_distances, seed: int=1):
        """
        """
        super(CognitiveMapEvaluation, self).__init__()

        self.env = env
        self._landmarks = landmarks
        self._A = A # adjacency matrix

        # wrap the policy in a module to extract a hidden layer from the policy
        self._policy = HiddenLayerExtractor(policy, cfg.eval.hidden_layer_activation)
        self._dist_func = dist_func
        self.hidden_layer_activations = {}
        self._seed = seed
        self._cfg = cfg

    def forward(self):
        """Run the evaluate an output the cognitive map and corresponding meta-data"""

        # The evaluation is ran for all pairwise landmarks that are connected (see adjacency matrix)
        N = self._A.shape[0] # should be NxN
        # map the index to alphabettical value
        # index_2_ldmrk_key = lambda i,j : [chr(i+ord('A')), chr(j+ord('A'))]
        indicies = np.nonzero(self._A)
        landmark_graph = [(chr(i+ord('A')), chr(j+ord('A'))) for i, j in zip(*indicies)]

        for landmark_start_key, landmark_goal_key in landmark_graph:
            
            landmark_start_pos = self._landmarks[landmark_start_key]
            landmark_goal_pos = self._landmarks[landmark_goal_key]

            # remake the environment
            env = make_env_for_landmark_setting(landmark_start_pos, landmark_goal_pos, self._cfg)
            obs_at_landmark = env.reset()

            lstm_states = None

            # take an action and get the requested hidden layer of that activation
            action, hidden_layer_activation, lstm_states = self._policy(obs_at_landmark, lstm_states)
            # print(f"Hidden Layer activation: {hidden_layer_activation}")
            # save the hidden_layer_activation
            # self.hidden_layer_activations.append(hidden_layer_activation)
            self.hidden_layer_activations[(landmark_start_key, landmark_goal_key)] = hidden_layer_activation
        
        # compute the distance matrix for the hidden layer activations
        dist_matrix = self.hidden_layers_to_distance_matrix(self.hidden_layer_activations, dist_func=self._dist_func)

        # Save the cognitive map to wandb or something else.
        return mds_cognitive_mapping(dist_matrix, self._landmarks, self._seed)
    
    @property
    def landmarks(self):
        return self._landmarks
    
    @landmarks.setter
    def landmarks(self, value: List[Dict[str, Tuple]]):
        self._landmarks = value

    @property
    def policy(self):
        return self._policy
    
    @policy.setter
    def policy(self, policy):
        print("Resetting policy")
        self._policy = HiddenLayerExtractor(policy, self._cfg.eval.hidden_layer_activation)

            
    def hidden_layers_to_distance_matrix(self, hidden_layer_activations_struct: List[torch.tensor], dist_func: Callable=cosine_distances):
        """
        Builds a distance matrix between the pairwise distances of hidden layer activations.
        Note, each hidden layer is actually attached to a give landmark start/goal pair. Thus
        we compute the distances beween activations for different start postions over the same goal.
        We then average over all the goals.
        The output distance matrix is NxN, where N is the number of landmarks
        """
        # TODO: Make this function faster with tensorization.
        # N = len(hidden_layer_activations_struct)
        N = self._A.shape[0]
        key_to_int = lambda x : ord(x) - ord('A')
        dist_matrix_goal_conditioned = np.zeros((N, N, N, N)) # shape (goals_i,goals_j,landmark_start_i,landmark_start_j)
        for key_i, value_i in hidden_layer_activations_struct.items():
            landmark_start_key_i, landmark_goal_key_i = key_i[0], key_i[1]
            hidden_layer_activation_i = value_i
            for key_j, value_j in hidden_layer_activations_struct.items():
                landmark_start_key_j, landmark_goal_key_j = key_j[0], key_j[1]
                hidden_layer_activation_j = value_j

                a,b,c,d = (key_to_int(landmark_goal_key_i), 
                            key_to_int(landmark_goal_key_j), 
                            key_to_int(landmark_start_key_i), 
                            key_to_int(landmark_start_key_j))
                
                dist_matrix_goal_conditioned[a,b,c,d] = dist_func(hidden_layer_activation_i, hidden_layer_activation_j)
        # average over all the goal conditions
        dist_matrix = np.mean(dist_matrix_goal_conditioned, axis=(0,1))
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

def mds_cognitive_mapping(cog_map_dist_matrix: np.array, landmarks: Dict[str, List], seed: int=1):
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

    # get the landmarks into an array
    N = len(landmarks)
    true_ldmrk_locs = np.zeros((N,2))
    for i in range(N):
        true_ldmrk_locs[i,:] = landmarks[chr(ord('A')+i)]

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
    
    # TODO: Move to a separate evaluation type, but this is convient here for now
    pos_distances = euclidean_distances(pos)
    true_distances = euclidean_distances(true_ldmrk_locs)

    stat, pvalue = pearsonr(pos_distances.flatten(), true_distances.flatten())
    print(stat)

    fig = plot_cognitive_map(true_ldmrk_locs, pos, landmark_keys=list(landmarks.keys()))
    return fig, disparity, stat

        