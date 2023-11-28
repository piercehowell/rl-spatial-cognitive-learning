import gym
import gym_gridverse
from stable_baselines3 import PPO, A2C
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import outer_env_factory, GymEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper


class SelectKeysWrapper(gym.Wrapper):
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys
        print(self.observation_space)
        self.observation_space = gym.spaces.Dict({key: env.observation_space[key] for key in self.keys})
        print(self.observation_space)
    
    def step_wait(self, action):
        obs, reward, done, info = self.env.step(action)
        return {key: obs[key] for key in self.keys}, reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return {key: obs[key] for key in self.keys}
    
env = gym.make('GV-FourRooms-7x7-v0')

obs_keys = ["agent_id_grid"]
obs = env.reset()
env = SelectKeysWrapper(env, obs_keys)
print(env.observation_space)
# env = ModifiedObservationWrapper(env, obs_keys)
# env = DummyVecEnv([lambda: env])


model = PPO("MultiInputPolicy", env, verbose=1)
obs = env.reset()
print(obs.keys())
model.learn(total_timesteps=1_000_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()