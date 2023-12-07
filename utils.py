from torch import nn
import torch
from typing import List, Dict, Tuple, Callable, Union
import gym
import yaml
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import matplotlib.pyplot as plt
import os
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml, factory_env_from_data
from gym_gridverse.gym import outer_env_factory, GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)

# TODO: Clean up imports

# directory of run.py
script_path = os.path.dirname(os.path.abspath(__file__))

#Setup environment
class FlattenObservationWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		total_size = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces)
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)

	def observation(self, observation):
		# Flatten each part of the observation and then concatenate
		flattened_obs = np.concatenate([observation[key].flatten() for key in observation])
		return flattened_obs

def make_env_from_yaml(cfg):
    """
    Make the environment
    """
    # what are the registered reset functions
    # TODO: Edit this to look like https://github.com/abaisero/gym-gridverse/blob/07909d928595f44e152c8dd88bc38198d1a7f2a4/gym_gridverse/envs/yaml/factory.py#L242
    # but reset any parameter you desire.
    # inner_env = factory_env_from_yaml(os.path.join(script_path, 'environments', cfg.environment))
    path = os.path.join(script_path, 'environments', cfg.environment)
    with open(path) as f:
        data = yaml.safe_load(f)
    
    env = make_env_from_data(data, cfg)
    return env
        

def make_env_from_data(data, cfg):
    """
    Make the environment from data (not yaml)
    """
    inner_env = factory_env_from_data(data)
         
    state_representation = make_state_representation(
        'default',
        inner_env.state_space,
    )
    observation_representation = make_observation_representation(
        'default',
        inner_env.observation_space,
    )
    outer_env = OuterEnv(
        inner_env,
        state_representation=state_representation,
        observation_representation=observation_representation,
    )
    env = GymEnvironment(outer_env)
    env.seed(cfg.seed)
    # from gym_gridverse.envs.reset_functions import reset_function_registry
    # print(reset_function_registry.keys())


    # env= gym.make("GV-FourRooms-9x9-v0")
    env = FlattenObservationWrapper(env)
    return env

def visualize(model, env):
    """
    Visualize the environment
    """
    #Visualize agent
    obs = env.reset()
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)

        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        env.render()
        if dones:
            obs = env.reset()
     
