from gym.envs.registration import register
import mpe.scenarios as scenarios

def _register(scenario_name, gymkey):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,

            "done_callback": scenario.done,
        },
    )

scenario_name = f"simple_tag_random"
gymkey = f"SimpleTagRandom-v0"
_register(scenario_name, gymkey)

scenario_name = f"simple_tag_heuristic"
gymkey = f"SimpleTagHeuristic-v0"
_register(scenario_name, gymkey)

scenario_name = f"simple_spread_2"
gymkey = f"SimpleSpread2-v0"
_register(scenario_name, gymkey)

scenario_name = f"simple_spread_3"
gymkey = f"SimpleSpread3-v0"
_register(scenario_name, gymkey)