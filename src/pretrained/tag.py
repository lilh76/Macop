import gym
from gym.spaces import Tuple
# from pretrained.ddpg import DDPG
import torch
import os

class RandomTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.n_agents = 2
        self.n_preys = 3

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:-1 * self.n_preys]

    def step(self, action):
        action = tuple(action) + tuple([self.pt_action_space.sample() for _ in range(self.n_preys)])
        obs, rew, done, info = super().step(action)
        return obs[:-1 * self.n_preys], rew[:-1 * self.n_preys], done[:-1 * self.n_preys], info

class HeuristicTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.n_agents = 2
        self.n_preys = 5

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs = obs[-1 * self.n_preys: ] # len = n_preys
        return obs[:-1 * self.n_preys] # len = n_agents
    
    def get_action(self, prey_obs):
        # np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + other_pos + other_vel)
        # x_self, y_self, vx_self, vy_self, x_predator1, y_predator1, x_predator2, y_predator2, ...
        self_pos = prey_obs[:2]
        predators_pos = prey_obs[4: 4 + 2 * self.n_agents].reshape(self.n_agents, 2)
        dist = ((predators_pos - self_pos) ** 2).sum(axis=-1)
        nearest_predator_pos = predators_pos[dist.argmax()]
        x, y = nearest_predator_pos
        if x >= 0 and abs(x) >= abs(y):
            prey_action = 2
        elif x < 0 and abs(x) >= abs(y):
            prey_action = 1
        elif y >= 0 and abs(y) >= abs(x):
            prey_action = 4
        elif y < 0 and abs(y) >= abs(x):
            prey_action = 3
        else:
            prey_action = 0
        return prey_action

    def step(self, action):
        action = tuple(action) + tuple([self.get_action(self.last_prey_obs[i]) for i in range(self.n_preys)])
        obs, rew, done, info = super().step(action)
        return obs[:-1 * self.n_preys], rew[:-1 * self.n_preys], done[:-1 * self.n_preys], info
        