from src.config import Config
from src.mcts.node import Node
import numpy as np

def add_exploration_noise(config: Config, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_noise_alpha] * len(actions))
    frac = config.dirichlet_exploration_factor
    for a, n in zip(actions, noise):
        node.children[a].policy_value = node.children[a].policy_value * (1 - frac) + n * frac
