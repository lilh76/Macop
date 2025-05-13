from .q_learner_tm import QLearnerTM
from .q_learner_ego import QLearnerEgo

REGISTRY = {}

REGISTRY["q_learner_tm"] = QLearnerTM
REGISTRY["q_learner_ego"] = QLearnerEgo