import math
from src.utils.minmaxstats import MinMaxStats
from src.config import Config
from src.mcts.node import Node


def select_child(config: Config,  node: Node, min_max_stats: MinMaxStats):

    _, action, child = max((puct_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items())

    return action, child

    # The score for a node is based on its value, plus an exploration bonus based on the prior.

def puct_score(config: Config, node: Node, child: Node, min_max_stats: MinMaxStats):
    pb_c = math.log((node.visits + config.c2 + 1) / config.c1) + config.c1
    pb_c *= math.sqrt(node.visits) / (child.visits + 1)

    prior_score = pb_c * child.policy_value
    if child.visits > 0:
        value_score = min_max_stats.normalize(child.reward + config.discount * child.value())
    else:
        value_score = 0
    return prior_score + value_score        
    
    #  if node.visits == 0:
    #     Q = torch.tensor(0.0)  # Ensure Q is a tensor
    # else:
    #     Q = torch.tensor(node.value_sum / node.visits)  # Convert Q to tensor

    # if node.parent is None:
    #     P = torch.tensor(0.0)  # Ensure P is a tensor
    # else:
    #     P = torch.tensor(node.parent.visits)  # Convert P to tensor

    # # Ensure policy_value is a tensor
    # policy_value = node.policy_value
    # if not isinstance(policy_value, torch.Tensor):
    #     policy_value = torch.tensor(policy_value, dtype=torch.float32)

    # # Compute the score as a tensor
    # score = Q + policy_value * torch.sqrt((P) / (1 + Q)) * (config.c1 + torch.log((P + config.c2 + 1) / config.c2))
    # return torch.argmax(score)