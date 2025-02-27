from numpy import argmax
from src.config import Config
from src.mcts.node import Node
from math import sqrt, log


def select_child(config: Config,  node: Node):

    _, action, child = max((puct_score(config, node), action, child)
                           for action, child in node.children.items())
    return action, child

    # The score for a node is based on its value, plus an exploration bonus based on the prior.


def puct_score(config: Config, node: Node):
    if node.visits == 0:
        Q = 0
    else:
        Q = node.value_sum/node.visits
    
    if node.parent is None:
        P = 0
    else:
        P = node.parent.visits
    return argmax(Q + node.policy_value * sqrt((P)/(1+Q))*(config.c1 + log((P + config.c2 + 1) / config.c2)))
