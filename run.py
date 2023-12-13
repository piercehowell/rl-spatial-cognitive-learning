import torch
import yaml
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from gym_gridverse.envs.yaml.factory import factory_env_from_yaml, factory_env_from_data
# from gym_gridverse.gym import outer_env_factory, GymEnvironment
# from gym_gridverse.outer_env import OuterEnv
# from gym_gridverse.representations.observation_representations import (
#     make_observation_representation,
# )
# from gym_gridverse.representations.state_representations import (
#     make_state_representation,
# )
from utils import make_env_from_yaml, visualize

from wandb.integration.sb3 import WandbCallback
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from evals import CognitiveMapEvaluation
from copy import deepcopy


# directory of run.py
script_path = os.path.dirname(os.path.abspath(__file__))

def run_experiment(cfg):
    """
    Runs the experiment
    """
    # build the environment
    env = make_env_from_yaml(cfg)

    # set the seed
    seed_value = cfg.seed
    np.random.seed(cfg.seed)
    # Set seed for CPU
    torch.manual_seed(seed_value)
    # Set seed for GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # training mode
    if(cfg.mode == "train"):

        # train model
        model = train_model(env, cfg)

        #Test and save full trained model
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        # model.save("ppo_gridworld_9x9_raytracing_1_8M")


    # evaluation mode
    elif(cfg.mode == "eval"):
        evaluate_model(env, cfg)

def load_landmark_specs(cfg):
    """
    Loads up the landmarks and the corresponding adjacency matrix
    """
    landmarks = cfg.landmark_spec.landmarks
    A = np.array(cfg.landmark_spec.adjacency_matrix)
    return landmarks, A


def load_policy(env, cfg):
	"""
	Load the specified stable baselines model
	"""

	if cfg.model.policy_type=="RecurrentPPO":
		model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, 
					device=cfg.device, n_steps=cfg.model.n_steps,
					tensorboard_log=f"./results/tb/{wandb.run.id}")
	elif cfg.model.policy_type=="PPO":
		model = PPO("MlpPolicy", env, verbose=2, 
					device=cfg.device, n_steps=cfg.model.n_steps,
					tensorboard_log=f"./results/tb/{wandb.run.id}")
	return model

def evaluate_model(env, cfg):
    """
    Evaluates the desired model
    """
    landmarks, A = load_landmark_specs(cfg)

    # Load up the saved models to evaluate on.
    # TODO: Organize the saved modes in ascending order by step count
    # and iterate the evaluation for each model.
    evaluation_models_dir = os.path.join(script_path, 'evaluation_models', cfg.eval.evaluation_model)
    file_names = os.listdir(evaluation_models_dir)
    model_checkpoints = [name.rstrip('.zip') for name in file_names] # remove the .zip extension (stable baselines doesn't expect it)
    get_step = lambda x: int(x.split('_')[-2])
    model_checkpoints.sort(key=get_step)

    if(cfg.eval.name == "CognitiveMapEvaluation"):

        # iterate through each of the trained models and run the evaluation to get the cognitive map 
        # eval_module = CognitiveMapEvaluation(cfg, env, landmarks, A, policy)
        # model_checkpoints = ['ppo_model_9x9_40_steps']
        for model_checkpoint in model_checkpoints:
            print(f"Running Cognitive Map evaluation for model checkpoint: {model_checkpoint}")
            policy = load_policy(env, cfg)
            policy = policy.load(os.path.join(evaluation_models_dir, model_checkpoint), device=cfg.device)
            # print(policy.policy)
            # eval_module.policy = policy
            eval_module = CognitiveMapEvaluation(cfg, env, landmarks, A, policy)
            cog_map_fig, disparity, dist_correlation_coeff = eval_module()
            
            step = get_step(model_checkpoint)
            print(f'step: {step}')
            wandb.log({'eval/cog_map':cog_map_fig,
                       'eval/disparity':disparity,
                       'eval/dist_correlation_coeff':dist_correlation_coeff}, step=step)
            # wandb.log({'eval/disparity':disparity}, step=step)

def train_model(env, cfg):

	total_timesteps = cfg.model.total_timesteps
	model = load_policy(env, cfg)

	# Save the model every 100k steps
	checkpoint_callback = CheckpointCallback(save_freq=cfg.model.save_freq, save_path=f'./results/models/{wandb.run.id}', name_prefix='ppo_model_9x9')

	# Evaluation and logging
	eval_callback = EvalCallback(env, best_model_save_path='./models/', log_path='./logs/', eval_freq=50000)

	model.learn(total_timesteps=total_timesteps, callback=[WandbCallback(), checkpoint_callback,eval_callback])

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
				monitor_gym=True, config=dict(cfg), mode=cfg.wandb.mode,
                group=cfg.wandb.group):
		run_experiment(cfg)


if __name__ == "__main__":

	# TODO: Make better configuration
	hydra_experiment()
	