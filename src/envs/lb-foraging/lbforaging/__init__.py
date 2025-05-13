from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = [5, 6]
players = [2]
coop = [True]
foods = [4]
_get_close = [False]
for s, p, f, c, close in product(sizes, players, foods, coop, _get_close):
    register(
        id="Foraging-{0}x{0}-{1}p-{2}f{3}{4}-v1".format(s, p, f, "-coop" if c else "", "-close" if close else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 500,
            "force_coop": c,
            "_get_close": close,
        },
    )

foods = [1]
_get_close = [True]
min_close_count_lst = [2]
for s, p, f, c, close, min_close_count in product(sizes, players, foods, coop, _get_close, min_close_count_lst):
    register(
        id="Foraging-{0}x{0}-{1}p-{2}f{3}{4}-{5}-v1".format(s, p, f, "-coop" if c else "", "-close" if close else "", min_close_count),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 500,
            "force_coop": c,
            "_get_close": close,
            "min_close_count": min_close_count,
        },
    )
