import copy
import torch as th
from torch.optim import RMSprop, Adam
from torch.nn.functional import kl_div

from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd


class QLearnerTM:
    def __init__(self, mac, scheme, logger, args, tm_index):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.tm_index = tm_index

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.optim_type.lower() == "rmsprop":
            self.optimiser = RMSprop(params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
        elif self.args.optim_type.lower() == "adam":
            self.optimiser = Adam(params=self.params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Invalid optimiser type", self.args.optim_type)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.sp_ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
            self.xp_ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.sp_rew_ms = RunningMeanStd(shape=(1,), device=device)
            self.xp_rew_ms = RunningMeanStd(shape=(1,), device=device)

    def get_loss(self, batch: EpisodeBatch, type="sp", get_diversity=False, tm_id=None, tm2mac=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        log_info = {}
        if self.args.standardise_rewards:
            if type == "sp":
                self.sp_rew_ms.update(rewards)
                rewards = (rewards - self.sp_rew_ms.mean) / th.sqrt(self.sp_rew_ms.var)
            elif type == "xp":
                self.xp_rew_ms.update(rewards)
                rewards = (rewards - self.xp_rew_ms.mean) / th.sqrt(self.xp_rew_ms.var)
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        div_loss = th.tensor(0)
        if get_diversity and self.args.diversity_coef > 0 and len(tm2mac) > 1:
            all_mac_out = [mac_out[:, :-1].unsqueeze(0)] # (bs, seq_len-1, n_agents, n_actions)
            for tm, mac in tm2mac.items():
                if tm == self.tm_index:
                    continue
                tm_mac_out = []
                mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    agent_outs = mac.forward(batch, t=t)
                    tm_mac_out.append(agent_outs)
                tm_mac_out = th.stack(tm_mac_out, dim=1)[:, :-1].unsqueeze(0) # [1, bs, seq_len, n_agents, n_actions]
                all_mac_out.append(tm_mac_out)
            mean_mac_out = th.softmax(th.cat(all_mac_out, dim=0).detach().mean(dim=0), dim=-1) # [bs, seq_len, n_agents, n_actions]
            mac_out_softmax = th.softmax(mac_out[:, :-1], dim=-1)
            kl = kl_div(mac_out_softmax.log(), mean_mac_out, reduction='none').mean(dim=-1).mean(dim=-1, keepdim=True) # [bs, seq_len, n_agents, n_actions] -> [bs, seq_len, 1]
            div_loss = -(kl * mask).mean()

        mask_elems = mask.sum().item()
        log_info["td_error_abs"] = (masked_td_error.abs().sum().item()/mask_elems)
        log_info["q_taken_mean"] = (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents)
        log_info["target_mean"] = (targets * mask).sum().item()/(mask_elems * self.args.n_agents)
        return td_loss, div_loss, log_info

    def train(self, sp_batch: EpisodeBatch, xp_batch: EpisodeBatch, t_env: int, episode_num: int, tm2mac):
        xp_td_loss = th.tensor(0)
        sp_td_loss, diveristy_loss, sp_log_info = self.get_loss(sp_batch, type="sp", get_diversity=True, tm2mac=tm2mac)
           
        xp_log_info = {}
        if self.args.xp_coef > 0 and xp_batch is not None:
            xp_td_loss, _, xp_log_info = self.get_loss(xp_batch, type="xp")
        loss = sp_td_loss + self.args.xp_coef * xp_td_loss + self.args.diversity_coef * diveristy_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(f"loss_{self.tm_index}", loss.item(), t_env)
            self.logger.log_stat(f"sp_loss_{self.tm_index}", sp_td_loss.item(), t_env)
            self.logger.log_stat(f"div_loss_{self.tm_index}", diveristy_loss.item(), t_env)
            self.logger.log_stat(f"xp_loss_{self.tm_index}", xp_td_loss.item(), t_env)
            self.logger.log_stat(f"grad_norm_{self.tm_index}", grad_norm, t_env)
            for sp_elem_key in sp_log_info:
                self.logger.log_stat("sp_"+sp_elem_key+f"_{self.tm_index}", sp_log_info[sp_elem_key], t_env)
            for xp_elem_key in xp_log_info:
                self.logger.log_stat("xp_"+xp_elem_key+f"_{self.tm_index}", xp_log_info[xp_elem_key], t_env)
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
