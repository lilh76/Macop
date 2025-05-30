from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        elif max_episode_steps is None and getattr(self.env, "token", None) == "gym_cooking":
            # gym_cooking
            max_episode_steps = self.env.max_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        #print(done)
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]

        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []
        
        if getattr(env, "observation_spaces", None) is None:
            for sa_obs in env.observation_space:
                flatdim = spaces.flatdim(sa_obs)
                ma_spaces += [
                    spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(flatdim,),
                        dtype=np.float32,
                    )
                ]
            self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        #print(observation, type(observation))
        #if isinstance(observation, dict):
        #    observation = list(observation.values())
        return tuple(
                [
                    spaces.flatten(obs_space, obs)
                    for obs_space, obs in zip(self.env.observation_space, observation)
                ]
            )
        """if getattr(self.env, "observation_spaces", None) is None:
            return tuple(
                [
                    spaces.flatten(obs_space, obs)
                    for obs_space, obs in zip(self.env.observation_space, observation)
                ]
            )
        else:
            return tuple(
                [
                    spaces.flatten(obs_space, obs)
                    for obs_space, obs in zip(list(self.env.observation_spaces.values()), observation)
                ]
            )"""



import pretrained

class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit
        self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            if "Random" in key: # SimpleTagRandom-v0, SimpleTagRandom{i}-v0, ...
                self._env = pretrained.RandomTag(self._env)
            elif "Heuristic" in key:
                self._env = pretrained.HeuristicTag(self._env)
            else:
                assert 0

        self.n_agents = self._env.n_agents
        self._obs = None
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
                self._env.observation_space, key=lambda x: x.shape
            )
        """if getattr(self._env, "action_spaces", None) is None:    
            self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        else:
            self.longest_action_space = max(self._env.action_spaces.values(), key=lambda x: x.n)"""
        
        """if getattr(self._env, "observation_spaces", None) is None:
            self.longest_observation_space = max(
                self._env.observation_space, key=lambda x: x.shape
            )
        else:
            self.longest_observation_space = max(
                self._env.observation_spaces.values(), key=lambda x: x.shape
            )"""

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        #print(self._obs, reward, done)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)
        # return self._env.get_state()

    def get_state_size(self):
        return self.n_agents * flatdim(self.longest_observation_space)
        # return self._env.get_state_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        
        if hasattr(self._env, "get_agent_valid_actions"):
            valid = np.array(flatdim(self._env.action_space[agent_id]) * [0])
            valid_moves = self._env.get_agent_valid_actions(agent_id)
            valid[valid_moves] = 1
            valid = valid.tolist()
        else:
            valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        #print(self._env)
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)