"""
"""
from torch import nn
from typing import List, Dict, Tuple
import gym

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
    def __init__(self, env: gym.Env, landmarks: List[Dict[str, Tuple]], goal_ldmrk: str, policy: nn.Module):
        """
        """
        self.env = env
        self.landmarks = landmarks

        # wrap the policy in a module to extract a hidden layer from the policy
        self.policy = HiddenLayerExtractor(policy)
        self.goal_ldmrk = goal_ldmrk
        self.hidden_layer_activations = []

    def forward(self):
        """Run the evaluate an output the cognitive map and corresponding meta-data"""

        for ldmrk_name, ldmrk_loc in self.landmarks.items():

            # reset the environment so that the agent begins at the landmark location
            # pass in the starting point and goal ldmrk
            # TODO
            obs_at_ldmrk = self.env.reset(ldmrk_loc, self.goal_ldmrk)

            # take an action
            action, hidden_layer_activation = self.policy(obs_at_ldmrk)

            # save the hidden_layer_activation
            self.hidden_layer_activations.append({'ldmrk_name': [ldmrk_loc, hidden_layer_activation]})

        # generate the cognitive map
        cog_map = mds_cognitive_mapping(self.hidden_layer_activations)
        return cog_map

            


        