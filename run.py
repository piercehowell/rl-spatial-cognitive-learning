import gym
import gym_gridverse
import torch
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from evals import CognitiveMapEvaluation


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

def run_experiment(cfg):
    """
    Runs the experiment
    """
    # build the environment
    env = make_env()

    if(cfg.mode == "train"):

        # train model
        model = train_model(env, cfg)

        #Test and save full trained model
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        # model.save("ppo_gridworld_9x9_raytracing_1_8M")

    elif(cfg.mode == "eval"):
        evaluate_model(env, cfg)

def visualize(model):
    """
    Visualize the environment
    """
    #Visualize agent
    env= gym.make("GV-FourRooms-9x9-v0")
    env = FlattenObservationWrapper(env)
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
     
def make_env():
	env= gym.make("GV-FourRooms-9x9-v0")
	env = FlattenObservationWrapper(env)
	return env

#Training function

def load_policy(env, cfg):
    """
    Load the specified stable baselines model
    """

    if cfg.model.policy_type=="RecurrentPPO":
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, 
                    device=cfg.device, n_steps=cfg.model.n_steps,
                    tensorboard_log=f"./results/tb/{wandb.run.id}")
    
    return model

def evaluate_model(env, cfg):
    """
    Evaluates the desired model
    """

    # TODO: Get the landmarks and goal landmark from a specific configuration.

    # Load up the saved models to evaluate on.
    # TODO: Organize the saved modes in ascending order by step count
    # and iterate the evaluation for each model.
    evaluation_models_dir = "./evaluation_models/recurrent_ppo_test"
    file_names = os.listdir(evaluation_models_dir)

    # initialize the policy
    policy = load_policy(env, cfg)
    policy.load(file_names[0]) # TODO: This is just for testing, we will actually iterate throught file names later

    if(cfg.eval_type == "CognitiveMapping"):
        CognitiveMapEvaluation(env, landmarks, goal_landmark, policy)


def train_model(env, cfg):

    total_timesteps = cfg.model.total_timesteps
    model = load_policy(env, cfg)

    # Save the model every 100k steps
    checkpoint_callback = CheckpointCallback(save_freq=cfg.model.save_freq, save_path=f'./results/models/{wandb.run.id}', name_prefix='ppo_model_9x9')

    # Evaluation and logging
    #eval_callback = EvalCallback(env, best_model_save_path='./models/', log_path='./logs/', eval_freq=50000)

    model.learn(total_timesteps=total_timesteps, callback=[WandbCallback(), checkpoint_callback])

    return model

@hydra.main(version_base=None, config_path="conf", config_name="default")
def hydra_experiment(cfg: DictConfig) -> None:
    """
    Load parameter from config file using Hydra
    """
    print(OmegaConf.to_yaml(cfg))
    print(f"Working Directory : {os.getcwd()}")
    print(f"Output Directory : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    with wandb.init(project="rl_spatial_cognitive_learning", sync_tensorboard=True, 
                monitor_gym=True, config=dict(cfg), mode=cfg.wandb.mode):
        run_experiment(cfg)


if __name__ == "__main__":

    # TODO: Make better configuration
    hydra_experiment()
    