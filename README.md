# Learning to Coordinate with Anyone

This repository contains official implementation for Learning to Coordinate with Anyone.

## Environment Installation

Build the environment by running:

```
pip install -r requirements.txt
```

Install the Level Based Foraging (LBF) environment by running:

```
pip install -e src/envs/lb-foraging
```

Install the Predator-Prey (PP) environment by running:

```
pip install -e src/envs/mpe/multi_agent_particle
```

Install the StarCraft Multi-Agent Challenge (SMAC) environment by running:

```
pip install -e src/envs/smac
```

## Run an experiment

```
python3 src/main.py --config=[Algorithm name] --env-config=[Scenario name]
```

The config files act as defaults for an algorithm or scenario. They are all located in `src/config`. `--config` refers to the config files in `src/config/algs` including Macop-VDN and Macop-QMIX. `--env-config` refers to the config files in `src/config/envs`, including the LB-Foraging environment (https://github.com/semitable/lb-foraging), the Predator Prey and the Cooperative Navigation environments (https://github.com/openai/multiagent-particle-envs),  and the StarCraft Multi-Agent Challenge environment (https://github.com/oxwhirl/smac).

All results will be stored in the `results` folder.

For example, run Macop-VDN on LBF1 scenario:

```
python3 src/main.py --config=vdn --env-config=lbf1
```

## Publication

If you find this repository useful, please [cite our paper](https://dl.acm.org/doi/10.1145/3627676.3627678):

```
@inproceedings{macop,
  title     = {Learning to Coordinate with Anyone},
  author    = {Lei Yuan and Lihe Li and Ziqian Zhang and Feng Chen and Tianyi Zhang and Cong Guan and Yang Yu and Zhi-Hua Zhou},
  booktitle = {Proceedings of the Fifth International Conference on Distributed Artificial Intelligence},
  year      = {2023}
}
```

