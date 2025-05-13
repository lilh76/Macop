from .tag import RandomTag, HeuristicTag

REGISTRY = {
    "random_tag": RandomTag,
    "heuristic_tag": HeuristicTag,
}