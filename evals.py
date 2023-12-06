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
        for name, module in self.policy.named_modules():
            if name == hidden_layer_name:
                module.register_forward_hook(self._hidden_layer_hook)
                break
        
    
    def _hidden_layer_hook(self, module, input, output):
        """
        Creates a hook that logs the intermediate output
        """
        if module.__class__.__name__ == self.hidden_layer_name:
            self.hidden_layer_output = output.detach()

    def forward(self, x):
        """
        Forward pass
        """
        action = self.policy(x)
        return action, self.hidden_layer_output
    

class CognitiveMapEvaluation:
    """
    Applies the MDS Cognitive Mapping algorithm to hidden layers
    of a policy at prespecified landmarks.
    """
    def __init__(self, env: gym.Env, landmarks: List[Dict[str, Tuple]], goal_ldmrk: str, policy: nn.Module, dist_func: Callable=cosine_distances, seed: int=1):
        """
        """
        self.env = env
        self._landmarks = landmarks

        # wrap the policy in a module to extract a hidden layer from the policy
        self.policy = HiddenLayerExtractor(policy)
        self._goal_ldmrk = goal_ldmrk
        self._dist_func = dist_func
        self.hidden_layer_activations = []
        self._seed = seed

    def forward(self):
        """Run the evaluate an output the cognitive map and corresponding meta-data"""

        ldmrk_locs = []
        for ldmrk_name, ldmrk_loc in self._landmarks.items():
            ldmrk_locs.append(ldmrk_loc)
            # reset the environment so that the agent begins at the landmark location
            # pass in the starting point and goal ldmrk
            # TODO: Update the environment such that on reset we can set the agents
            # starting landmark and goal landmark
            obs_at_ldmrk = self.env.reset(landmark_start=ldmrk_loc, landmark_goal=self._goal_ldmrk)

            # take an action and get the requested hidden layer of that activation
            action, hidden_layer_activation = self.policy(obs_at_ldmrk)

            # save the hidden_layer_activation
            self.hidden_layer_activations.append({ldmrk_name: [ldmrk_loc, hidden_layer_activation]})
        
        # compute the distance matrix for the hidden layer activations
        dist_matrix = hidden_layers_to_distance_matrix(self.hidden_layer_activations, dist_func=self._dist_func)


        # generate the cognitive map and return it (as a figure)
        cog_map = mds_cognitive_mapping(dist_matrix, ldmrk_locs, self._seed)

        # Save the cognitive map to wandb or something else.
        return cog_map
    
    @property
    def goal_ldmrk(self):
        return self._goal_ldmrk

    @goal_ldmrk.setter
    def goal_ldmrk(self, value: str):
        self._goal_ldmrk = value

    @property
    def landmarks(self):
        return self._landmarks
    
    @landmarks.setter
    def landmarks(self, value: List[Dict[str, Tuple]]):
        self._landmarks = value

            
def hidden_layers_to_distance_matrix(hidden_layer_activations_struct: List[Dict[tuple, List[Union[str, torch.tensor]]]], dist_func: Callable=cosine_distances):
    """
    Builds a distance matrix between the pairwise distances of hidden layer activations.
    The output distance matrix is NxN, where N is the number of landmarks
    """
    # TODO: Make this function faster with tensorization.
    N = len(hidden_layer_activations_struct)
    dist_matrix = np.zeros((N, N))
    for i, ldmrk_dict_i in enumerate(hidden_layer_activations_struct):
        ldmrk_name_i = ldmrk_dict_i.keys()[0]
        ldmrk_loc_i, hidden_layer_activation_i = ldmrk_dict_i[ldmrk_name_i]
        for j, ldmrk_dict_j in enumerate(hidden_layer_activations_struct):
            ldmrk_name_j = ldmrk_dict_j.keys()[0]
            ldmrk_loc_j, hidden_layer_activation_j = ldmrk_dict_j[ldmrk_name_j]

            dist_matrix[i, j] = dist_func(hidden_layer_activation_i, hidden_layer_activation_j)
    return dist_matrix



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

    pos, _, _ = procrustes(true_ldmrk_locs, pos)
    pos, true_ldmrk_locs = shift_embedding_to_reference(pos, true_ldmrk_locs, reference_idx=0)

    # I want a matplotlib scatter plot for X_true, pos, and npos.
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.]) 
    s = 100

    plt.scatter(true_ldmrk_locs[:, 0], true_ldmrk_locs[:, 1], color='green', s=s, lw=0, label='True Positions')
    plt.scatter(pos[:, 0], pos[:, 1], color='red', s=s, lw=0, label='MDS')

    # plot the first point for true_ldmrk_locs and pos with a star marker (marker='*')
    plt.scatter(true_ldmrk_locs[0, 0], true_ldmrk_locs[0, 1], color='blue', marker='*', s=s, lw=0, label='True Positions')
    plt.scatter(pos[0, 0], pos[0, 1], color='black', marker='*', s=s, lw=0, label='MDS')

    # plt.scatter(npos[:, 0], npos[:, 1], color='turquoise', s=s, lw=0, label='NMDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)

    return fig

        