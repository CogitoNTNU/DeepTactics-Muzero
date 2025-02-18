from numpy import argmax
from src.config import Config
from src.mcts.node import Node
from math import sqrt, log

def puct_score(config: Config, node: Node):
    return argmax(node.value/node.visits + node.policy_value * sqrt((node.parent.visits)/(1+node.visits))*(config.c1 + log((node.parent.visits + config.c2 + 1) / config.c2)))