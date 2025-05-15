import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from copy import deepcopy
import h5py

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import json
import numpy as np
from collections import defaultdict

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        tb_exp_direc = os.path.join(results_save_dir, 'tb_logs')
        logger.setup_tb(tb_exp_direc)

        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(results_save_dir, "config.json"), "w") as f:
            f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def evaluate_xp_log(args, test_tm_id:int, iter:int, iter_plus_tm_2_head_id:dict, all_head_id_list:list,
                    runner, mac_ego, test_mac):
    few_shot_info = {head_id: [] for head_id in all_head_id_list+[None]}
    for test_head_id in all_head_id_list + [None]:
        tmp_info = runner.run(mac1=mac_ego,  mac2=test_mac, test_mode=True, test_mode_1=True, test_mode_2=True, 
                    tm_id=test_tm_id, iter=iter, head_id=test_head_id, few_shot=True)
        few_shot_info[test_head_id].append(tmp_info["episode_return"])
    best_head_id, best_ret = -1, -1e9 
    for head_id, ret_list in few_shot_info.items():
        if len(ret_list) == 0:
            continue
        ret = np.mean(ret_list)
        if ret > best_ret:
            best_ret = ret
            best_head_id = head_id
    for _ in range(args.test_nepisode): # xp
        runner.run(mac1=mac_ego,  mac2=test_mac, test_mode=True, test_mode_1=True, test_mode_2=True, tm_id=test_tm_id, iter=iter, head_id=best_head_id)

def run_sequential(args, logger):

    ## well designed xp runner, basic env info and init buffer
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess)
    empty_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    ## init ego
    mac_ego = mac_REGISTRY[args.mac](empty_buffer.scheme, groups, args)
    buffer_ego = deepcopy(empty_buffer)
    learner_ego = le_REGISTRY[args.learner](mac_ego, empty_buffer.scheme, logger, args)
    if args.use_cuda:
        learner_ego.cuda()

    ## init tm pop
    tm2buffer, tm2bufferxp, tm2learner, tm2mac, iter2tm2mac4testing = {}, {}, {}, {}, {}
    for tm_id_init in range(args.n_population):
        mac_tm = deepcopy(mac_ego)
        tm2buffer[tm_id_init] = deepcopy(empty_buffer)
        tm2bufferxp[tm_id_init] = deepcopy(empty_buffer)
        tm2learner[tm_id_init] = le_REGISTRY[args.learner_tm](mac_tm, tm2buffer[tm_id_init].scheme, logger, args, tm_index=tm_id_init)
        if args.use_cuda:
            tm2learner[tm_id_init].cuda()
        tm2mac[tm_id_init] = mac_tm

    ## start running
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    all_head_id_list = []
    iter_plus_tm_2_head_id = {}
    iter_plus_tm_2_return = {}

    model_save_time = 0

    for iter in range(args.max_iteration):

        ## evolve teammates
        start_time = time.time()
        last_time = start_time
        if iter == 0 or getattr(args, "t_train_tm_after", -1)==-1:
            iter_t_train_tm = args.t_train_tm
        else:
            iter_t_train_tm = args.t_train_tm_after
        logger.console_logger.info("Beginning training teammates for {} timesteps".format(iter_t_train_tm))

        ## cache parents before mutation, tm2mac will be offspring
        if iter > 0:
            parents = {}
            for tm_parent, mac_parent in tm2mac.items():
                tmp_mac = mac_REGISTRY[args.mac](empty_buffer.scheme, groups, args)
                if args.use_cuda:
                    tmp_mac.cuda()
                tmp_mac.agent.load_state_dict(mac_parent.agent.state_dict())
                parents[tm_parent] = tmp_mac

        t_env_start = runner.t_env
        while runner.t_env - t_env_start <= iter_t_train_tm:

            ## sample a tm from pop and collect sp+xp traj
            tm_id_train = np.random.randint(args.n_population)
            episode_batch = runner.run(mac1=tm2mac[tm_id_train], mac2=None, test_mode=False, test_mode_1=False, test_mode_2=False, tm_id=tm_id_train, eps_greedy_t=runner.t_env-t_env_start)
            tm2buffer[tm_id_train].insert_episode_batch(episode_batch)
            if args.xp_coef > 0 and iter > 0:
                if len(all_head_id_list) == 0 or np.random.uniform(0, 1) < 0.5:
                    sample_head_id = None
                else:
                    sample_head_id = np.random.choice(all_head_id_list) # past head id
                episode_batch = runner.run(mac1=mac_ego, mac2=tm2mac[tm_id_train], test_mode=False, test_mode_1=True, test_mode_2=False, negative_reward=True, tm_id=tm_id_train, eps_greedy_t=runner.t_env-t_env_start, iter=iter, head_id=sample_head_id)
                tm2bufferxp[tm_id_train].insert_episode_batch(episode_batch)

            ## call learner and train the tm network
            if tm2buffer[tm_id_train].can_sample(args.batch_size):
                sp_batch = tm2buffer[tm_id_train].sample(args.batch_size)
                max_ep_t = sp_batch.max_t_filled()
                sp_batch = sp_batch[:, :max_ep_t]
                if sp_batch.device != args.device:
                    sp_batch.to(args.device)
                xp_batch = None
                if args.xp_coef > 0 and tm2bufferxp[tm_id_train].can_sample(args.batch_size) and iter > 0:
                    xp_batch = tm2bufferxp[tm_id_train].sample(args.batch_size)
                    max_ep_t = xp_batch.max_t_filled()
                    xp_batch = xp_batch[:, :max_ep_t]
                    if xp_batch.device != args.device:
                        xp_batch.to(args.device)
                tm2learner[tm_id_train].train(sp_batch, xp_batch, runner.t_env, episode, tm2mac)

            ## testing
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                logger.console_logger.info("Iter {} teammate t_env: {} / {}".format(iter, runner.t_env - t_env_start, iter_t_train_tm))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env - t_env_start, iter_t_train_tm), time_str(time.time() - start_time)))
                last_time = time.time()
                last_test_T = runner.t_env
                for test_tm, test_mac in tm2mac.items():
                    for _ in range(args.test_nepisode): # tm sp
                        runner.run(mac1=test_mac, mac2=None,     test_mode=True, test_mode_1=True, test_mode_2=True, tm_id=test_tm)
                    evaluate_xp_log(args, test_tm, iter, iter_plus_tm_2_head_id, all_head_id_list, runner, mac_ego, test_mac)

            ## logging
            episode += args.batch_size_run
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.log_stat("n_head", len(all_head_id_list), runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

            ## save models
            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                for save_tm, save_mac in tm2mac.items():
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env), "tm" + str(save_tm))
                    os.makedirs(save_path, exist_ok=True)
                    save_mac.save_models(save_path)

        ## clear the tm buffers
        for tm_id_train in range(args.n_population):
            tm2buffer[tm_id_train]   = deepcopy(empty_buffer)
            tm2bufferxp[tm_id_train] = deepcopy(empty_buffer)

        ## final testing
        for test_tm, test_mac in tm2mac.items():
            for _ in range(args.test_nepisode): # tm sp
                runner.run(mac1=test_mac, mac2=None,     test_mode=True, test_mode_1=True, test_mode_2=True, tm_id=test_tm)
            evaluate_xp_log(args, test_tm, iter, iter_plus_tm_2_head_id, all_head_id_list, runner, mac_ego, test_mac)

        ## population selection
        if iter > 0:
            # select the tm that can sp well
            can_sp_tm2sp_ret = {}
            for id_test_stop, mac_test_stop in enumerate(list(parents.values()) + list(tm2mac.values())):
                ret_lst = []
                for _ in range(args.test_nepisode):
                    ret = runner.run_test_xp(mac1=mac_test_stop, mac2=mac_test_stop, head_id=None)
                    ret_lst.append(ret)
                can_sp_tm2sp_ret[id_test_stop] = np.mean(ret_lst)
            for id_select, ret_select in can_sp_tm2sp_ret.items():
                logger.log_stat(f"select_sp_return_{id_select}", ret_select, runner.t_env)
            offspring_cannot_sp = sorted(can_sp_tm2sp_ret.items(), key=lambda x:x[1], reverse=True)[ int(args.n_population * 2 * args.selfplay_threshold) : ] # discard cannot-sp-tm
            for tm_id, sp_ret in offspring_cannot_sp:
                del can_sp_tm2sp_ret[tm_id]
            select_sp_return_mean =  np.mean(list(can_sp_tm2sp_ret.values()))
            logger.log_stat(f"select_sp_return_mean", select_sp_return_mean, runner.t_env)
            xp_ret_lst = []
            for id_test_stop, mac_test_stop in enumerate(list(parents.values()) + list(tm2mac.values())):
                if hasattr(mac_ego.agent, 'head_dict'):
                    head_id_lst = list(mac_ego.agent.head_dict.keys())
                    best_head_id = None
                    if len(head_id_lst) > 0:
                        head_id2fewshot_ret = defaultdict(list)
                        for test_head_id in all_head_id_list + [None]:
                            ret = runner.run_test_xp(mac1=mac_ego, mac2=mac_test_stop, head_id=test_head_id)
                            head_id2fewshot_ret[test_head_id].append(ret)
                        best_head_id, best_ret = -1, -1e9 
                        for head_id, ret_list in head_id2fewshot_ret.items():
                            if len(ret_list) == 0:
                                continue
                            ret = np.mean(ret_list)
                            if ret > best_ret:
                                best_ret = ret
                                best_head_id = head_id
                else:
                    best_head_id = None
                ret_lst = []
                for _ in range(args.test_nepisode):
                    ret = runner.run_test_xp(mac1=mac_ego, mac2=mac_test_stop, head_id=best_head_id)
                    ret_lst.append(ret)
                xp_ret_lst.append((id_test_stop, mac_test_stop, np.mean(ret_lst)))
                
            offspring_cannot_xp = sorted(xp_ret_lst, key=lambda x:x[2])
            offspring = []
            for item in offspring_cannot_xp:
                if item[0] in can_sp_tm2sp_ret:
                    offspring.append(item)
            offspring = offspring[ : args.n_population]
            offspring_ids = [x[0] for x in offspring]
            # select new pop and load network
            for tm_new_pop, mac_new_pop in tm2mac.items():
                mac_new_pop.agent.load_state_dict(offspring[tm_new_pop][1].agent.state_dict())
            # log selection info
            for id_select, mac_select, ret_select in xp_ret_lst:
                logger.log_stat(f"select_result_{id_select}", int(id_select in offspring_ids), runner.t_env)
                logger.log_stat(f"select_xp_return_{id_select}", ret_select, runner.t_env)

        # save new pop network
        for save_tm, save_mac in tm2mac.items():
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env), "tm" + str(save_tm))
            os.makedirs(save_path, exist_ok=True)
            save_mac.save_models(save_path)

        ## test convergence
        if iter > 0:
            new_pop_min_xp_return = offspring_cannot_xp[0][2]
            logger.log_stat(f"min_xp_return", new_pop_min_xp_return, runner.t_env)
            convergence_delta = (select_sp_return_mean - new_pop_min_xp_return) / (abs(select_sp_return_mean) + 1e-3)
            logger.log_stat(f"convergence_delta", convergence_delta, runner.t_env)
            if iter >= args.min_iteration and convergence_delta < args.stop_threshold:
                logger.console_logger.info(f"Stop Running! n_iter: {iter}")
                runner.close_env()
                return

        ## save the current pop for following testing
        iter2tm2mac4testing[iter] = {}
        for tm_, mac_ in tm2mac.items():
            tmp_mac = mac_REGISTRY[args.mac](empty_buffer.scheme, groups, args)
            if args.use_cuda:
                tmp_mac.cuda()
            tmp_mac.agent.load_state_dict(mac_.agent.state_dict())
            iter2tm2mac4testing[iter][tm_] = tmp_mac

        ## train ego
        start_time = time.time()
        last_time = start_time
        logger.console_logger.info("Beginning training ego for {} timesteps".format(args.t_train_ego))

        for tm_id_train_ego, mac_tm_train_ego in tm2mac.items():
            
            ## create a new head
            if iter==0 and tm_id_train_ego==0:
                pass
            else:
                mac_ego.agent.reset_head()
            
            ## start training
            t_env_start = runner.t_env
            ego_train_steps = args.t_train_ego // args.n_population
            while runner.t_env - t_env_start <= ego_train_steps:

                ## collect xp traj and learn
                episode_batch = runner.run(mac1=mac_ego, mac2=mac_tm_train_ego, test_mode=False, test_mode_1=False, test_mode_2=True, tm_id=tm_id_train_ego, eps_greedy_t=runner.t_env-t_env_start, iter=iter)
                buffer_ego.insert_episode_batch(episode_batch)
                if buffer_ego.can_sample(args.batch_size):
                    episode_sample = buffer_ego.sample(args.batch_size)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner_ego.train(episode_sample, runner.t_env, episode)

                ## testing
                if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                    logger.console_logger.info("Iter {} ego t_env: {} / {}".format(iter, runner.t_env - t_env_start, ego_train_steps))
                    logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env - t_env_start, ego_train_steps), time_str(time.time() - start_time)))
                    last_time = time.time()
                    last_test_T = runner.t_env
                    for iter_, tm2mac_ in iter2tm2mac4testing.items():
                        for tm_, mac_ in tm2mac_.items():
                            evaluate_xp_log(args, tm_, iter_, iter_plus_tm_2_head_id, all_head_id_list, runner, mac_ego, mac_)

                ## logging
                episode += args.batch_size_run
                if (runner.t_env - last_log_T) >= args.log_interval:
                    logger.log_stat("episode", episode, runner.t_env)
                    logger.log_stat("n_head", len(all_head_id_list), runner.t_env)
                    logger.print_recent_stats()
                    last_log_T = runner.t_env

                ## save models
                if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env), "ego")
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    learner_ego.save_models(save_path)

            ## clear the buffer
            buffer_ego = deepcopy(empty_buffer)

            ## final testing
            for iter_, tm2mac_ in iter2tm2mac4testing.items():
                for tm_, mac_ in tm2mac_.items():
                    evaluate_xp_log(args, tm_, iter_, iter_plus_tm_2_head_id, all_head_id_list, runner, mac_ego, mac_)

            ## head expansion mechanism, decide if the new trained head should be saved
            tm2head_id = get_head_id(args, tm_id_train_ego, iter, iter_plus_tm_2_head_id)
            logger.console_logger.info("Start Macop expand head")
            # log performance based on the mac
            all_head_log_info = {head_id: [] for head_id in all_head_id_list + [None]}
            # for tm, mac in tm2mac.items():
            for test_head_id in all_head_log_info.keys():
                for _ in range(args.test_nepisode):
                    tmp_info = runner.run(mac1=mac_ego,  mac2=mac_tm_train_ego, test_mode=True, test_mode_1=True,
                                            test_mode_2=True, tm_id=tm_id_train_ego, iter=iter,
                                            head_id=test_head_id, few_shot=True)
                    all_head_log_info[test_head_id].append(tmp_info["episode_return"])
            new_head_return = np.mean(all_head_log_info[None])
            maximum_past_head_id, maximum_past_head_return = -1, -1e9
            for head_id, head_return in all_head_log_info.items():
                if head_id is None:
                    continue
                if np.mean(head_return) > maximum_past_head_return:
                    maximum_past_head_return = np.mean(head_return)
                    maximum_past_head_id = head_id
            # determine whether to save the new head
            
            if (head_id == -1) or \
                    (new_head_return - maximum_past_head_return) / (np.abs(maximum_past_head_return) + 1e-3) > args.macop_threshold: # > 0
                # cache directly
                logger.console_logger.info("Expand New Head {}, new head return: {}, maximum past head return: {}".format(tm2head_id, new_head_return, maximum_past_head_return))
                iter_plus_tm_2_head_id[(iter, tm_id_train_ego)] = tm2head_id
                iter_plus_tm_2_return[(iter, tm_id_train_ego)] = new_head_return
                assert tm2head_id not in all_head_id_list
                all_head_id_list.append(tm2head_id)
                mac_ego.agent.cache_head(tm2head_id)
                learner_ego.cache_feature_layer(tm2head_id)
            else:
                logger.console_logger.info("Use past head {}, new head return: {}, maximum past head return: {}".format(maximum_past_head_id, new_head_return, maximum_past_head_return))
                # use the past head
                iter_plus_tm_2_head_id[(iter, tm_id_train_ego)] = maximum_past_head_id
                iter_plus_tm_2_return[(iter, tm_id_train_ego)] = maximum_past_head_return
                # do not cache head, but cache feature_layer
                learner_ego.cache_feature_layer(maximum_past_head_id)

            # write iter_plus_tm_head_id to file
            dict_save_dir = os.path.join(args.results_save_dir, "iter_plus_tm_2_head_id.json")
            saved_iter_plus_tm_2_head_id = {}
            for (iter, tm_id_train_ego), head_id in iter_plus_tm_2_head_id.items():
                saved_iter_plus_tm_2_head_id[f"iter_{iter}_tm_{tm_id_train_ego}"] = head_id
            # save the dict with json form
            with open(dict_save_dir, "w") as f:
                json.dump(saved_iter_plus_tm_2_head_id, f, indent=4)
            dict_save_dir = os.path.join(args.results_save_dir, "iter_plus_tm_2_head_id_plus_return.json")
            # save the dict with json form
            saved_iter_plus_tm_2_return = {}
            for (iter, tm_id_train_ego),  return_ in iter_plus_tm_2_return.items():
                saved_iter_plus_tm_2_return[f"iter_{iter}_tm_{tm_id_train_ego}"] = return_
            with open(dict_save_dir, "w") as f:
                json.dump(saved_iter_plus_tm_2_return, f, indent=4)

            ## save ego network
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env), "ego")
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner_ego.save_models(save_path)

    runner.close_env()
    logger.console_logger.info("Finished Training")

def get_head_id(args, tm, iter, iter_plus_tm_2_head_id):
    
    if (iter, tm) in iter_plus_tm_2_head_id:
        # seen (iter, tm)
        return iter_plus_tm_2_head_id[(iter, tm)]
    return len(set(iter_plus_tm_2_head_id.values()))

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
