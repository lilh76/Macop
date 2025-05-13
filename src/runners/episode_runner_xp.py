from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from copy import deepcopy

class EpisodeRunnerXP:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if getattr(self.env, "episode_limit", None) is None:
            self.episode_limit = self.env.get_env_info()["episode_limit"]
        else:
            self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, mac1=None, mac2=None, test_mode=False, test_mode_1=False, test_mode_2=False, 
            negative_reward=False, tm_id=None, iter=None, eps_greedy_t=0, head_id=None, few_shot=False, lipo_xptm_id=None):

        self.reset()

        run_info = {}
        terminated = False
        episode_return = 0
        mac1.init_hidden(batch_size=self.batch_size)
        if mac2 is not None:
            mac2.init_hidden(batch_size=self.batch_size)
    
        while not terminated:
                
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions1 = mac1.select_actions(self.batch, t_ep=self.t, t_env=eps_greedy_t, test_mode=test_mode_1, head_id=head_id)
            actions = actions1
            if mac2 is not None:
                actions2 = mac2.select_actions(self.batch, t_ep=self.t, t_env=eps_greedy_t, test_mode=test_mode_2)
                actions = self.merge_actions(actions1, actions2)
            
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            if negative_reward:
                reward *= -1
        
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions1 = mac1.select_actions(self.batch, t_ep=self.t, t_env=eps_greedy_t, test_mode=test_mode_1)
        actions = actions1
        if mac2 is not None:
            actions2 = mac2.select_actions(self.batch, t_ep=self.t, t_env=eps_greedy_t, test_mode=test_mode_2)
            actions = self.merge_actions(actions1, actions2)
        self.batch.update({"actions": actions}, ts=self.t)

        run_info = {"episode_return": episode_return} # maybe other info
        if few_shot:
            assert test_mode and test_mode_1 and test_mode_2
            return run_info    
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        if mac2 is None:
            if tm_id >= 0: # teammate selfplay
                log_prefix = f"test_tm_{tm_id}_" if test_mode else f"tm_{tm_id}_"
            elif tm_id == -1: # ego selfplay
                log_prefix = f"test_ego_" if test_mode else f"ego_"
        else:
            assert iter is not None
            log_prefix = f"test_xp_{iter}_{tm_id}_" if test_mode else f"xp_{iter}_{tm_id}_"
            if lipo_xptm_id is not None:
                log_prefix = f"test_xp_{tm_id}_{lipo_xptm_id}_" if test_mode else f"xp_{tm_id}_{lipo_xptm_id}_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # log head_id
        if mac2 is not None and test_mode and test_mode_1 and test_mode_2:
            if head_id is None:
                cur_stats["head_id_None"] = 1 + cur_stats.get("head_id_None", 0)
            else:
                cur_stats["head_id_None"] = 0 + cur_stats.get("head_id_None", 0)
                for head_id_ in range(len(mac1.agent.head_dict)):
                    if head_id_ == head_id:
                        cur_stats[f"head_id_{head_id_}"] = 1 + cur_stats.get(f"head_id_{head_id_}", 0)
                    else:
                        cur_stats[f"head_id_{head_id_}"] = 0 + cur_stats.get(f"head_id_{head_id_}", 0)
    
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            ret = np.mean(cur_returns)
            self._log(cur_returns, cur_stats, log_prefix)
            return ret
        elif (not test_mode) and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(mac1.action_selector, "epsilon"):
                self.logger.log_stat("epsilon1", mac1.action_selector.epsilon, self.t_env)
            if mac2 is not None and hasattr(mac2.action_selector, "epsilon"):
                self.logger.log_stat("epsilon2", mac2.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch
    
    def run_test_xp(self, mac1, mac2, head_id):
        self.reset()
        terminated = False
        episode_return = 0
        mac1.init_hidden(batch_size=self.batch_size)
        mac2.init_hidden(batch_size=self.batch_size)
        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)
            actions1 = mac1.select_actions(self.batch, t_ep=self.t, t_env=0, test_mode=True, head_id=head_id)
            actions2 = mac2.select_actions(self.batch, t_ep=self.t, t_env=0, test_mode=True)
            actions = self.merge_actions(actions1, actions2)
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
        return episode_return

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def merge_actions(self, actions1, actions2):
        actions = deepcopy(actions2)
        actions[0, : self.args.n_ego] = actions1[0, : self.args.n_ego]
        return actions