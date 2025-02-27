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
    return argmax(node.value_sum/node.visits + node.policy_value * sqrt((node.parent.visits)/(1+node.visits))*(config.c1 + log((node.parent.visits + config.c2 + 1) / config.c2)))
