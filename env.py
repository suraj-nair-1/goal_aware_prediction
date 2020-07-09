import cv2
import numpy as np
import torch
import gap_env


def _images_to_observation(images):
  images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  return images.unsqueeze(dim=0)  # Add batch dimension


class GymEnv():
  def __init__(self, env, seed, max_episode_length, action_repeat, problem = "rand"):
    import gym
    self._env = gym.make(env)
    self._env.seed(seed)
    self._env.problem=problem
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    return _images_to_observation(self._env._get_obs())
  
  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state, reward_k, done, _ = self._env.step(action)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    observation = _images_to_observation(self._env._get_obs())
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return (6,64,64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())


def Env(env, seed, max_episode_length, action_repeat, problem = "rand"):
  return GymEnv(env, seed, max_episode_length, action_repeat, problem)
