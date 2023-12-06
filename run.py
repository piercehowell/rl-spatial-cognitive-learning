import torch
import yaml
import numpy as np
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
from utils import make_env_from_yaml

from wandb.integration.sb3 import WandbCallback
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from evals import CognitiveMapEvaluation


# directory of run.py
script_path = os.path.dirname(os.path.abspath(__file__))

def run_experiment(cfg):
    """
    Runs the experiment
    """
    # build the environment
    env = make_env_from_yaml(cfg)

    if(cfg.mode == "train"):

        # train model
        model = train_model(env, cfg)

        #Test and save full trained model
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        # model.save("ppo_gridworld_9x9_raytracing_1_8M")

    elif(cfg.mode == "eval"):
        evaluate_model(env, cfg)

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
     
# def make_env(cfg):
#     """
#     Make the environment
#     """
#     # what are the registered reset functions
#     # TODO: Edit this to look like https://github.com/abaisero/gym-gridverse/blob/07909d928595f44e152c8dd88bc38198d1a7f2a4/gym_gridverse/envs/yaml/factory.py#L242
#     # but reset any parameter you desire.
#     # inner_env = factory_env_from_yaml(os.path.join(script_path, 'environments', cfg.environment))
#     path = os.path.join(script_path, 'environments', cfg.environment)
#     with open(path) as f:
#         data = yaml.safe_load(f)
#         inner_env = factory_env_from_data(data)
         
#     state_representation = make_state_representation(
#         'default',
#         inner_env.state_space,
#     )
#     observation_representation = make_observation_representation(
#         'default',
#         inner_env.observation_space,
#     )
#     outer_env = OuterEnv(
#         inner_env,
#         state_representation=state_representation,
#         observation_representation=observation_representation,
#     )
#     env = GymEnvironment(outer_env)

#     # from gym_gridverse.envs.reset_functions import reset_function_registry
#     # print(reset_function_registry.keys())


#     # env= gym.make("GV-FourRooms-9x9-v0")
#     env = FlattenObservationWrapper(env)
#     return env

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
    seed_value = cfg.seed
    np.random.seed(cfg.seed)
    # Set seed for CPU
    torch.manual_seed(seed_value)

    # Set seed for GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # TODO: Get the landmarks and goal landmark from a specific configuration.
    landmarks = {'A':(1,1), 'B':(2,3), 'C':(3,6), 'D':(6,2),'E':(6,5),'F':(7,7)}
    goal_landmark = {'C':(3,6)}

    # Load up the saved models to evaluate on.
    # TODO: Organize the saved modes in ascending order by step count
    # and iterate the evaluation for each model.
    base_dir = os.path.join(script_path, 'evaluation_models', 'recurrent_ppo_test')
    evaluation_models_dir = base_dir
    file_names = os.listdir(evaluation_models_dir)
    file_names = [name.rstrip('.zip') for name in file_names] # remove the .zip extension (stable baselines doesn't expect it)

    # initialize the policy
    policy = load_policy(env, cfg)
    policy.load(os.path.join(base_dir, 'ppo_model_9x9_2040_steps')) # TODO: This is just for testing, we will actually iterate throught file names later

    # TODO: Specify the hidden layers in the configuration file
    print(policy.policy)

    if(cfg.eval_type == "CognitiveMapping"):
        eval_module = CognitiveMapEvaluation(cfg, env, landmarks, goal_landmark, policy)
        eval_module()

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
    