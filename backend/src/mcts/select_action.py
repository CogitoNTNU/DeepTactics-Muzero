from src.networks.network import Network
from src.config import Config
from src.mcts.node import Node
import numpy as np


def select_action(config: Config, num_moves: int, node: Node, network: Network) -> int:
    """
    Selects an action from the root node using a softmax policy over visit counts.

    Args:
        config (Config): Configuration containing temperature and other parameters.
        num_moves (int): The current number of moves made.
        node (Node): The root node from which actions are selected.
        network (Network): The network used to determine the training steps.

    Returns:
        int: The index of the selected action.
    """
    visit_counts = [(child.visits, action) for action, child in node.children.items()]
    t = config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=network.training_steps())
    action = softmax_sample(visit_counts, t)
    
    return action

def softmax_sample(distribution, temperature: float) -> int:
    """
    Samples an action index from a distribution of visit counts using a softmax function
    with temperature scaling.

    Args:
        distribution (list of tuples): A list where each tuple contains visit counts and corresponding actions.
        temperature (float): A temperature parameter to control the randomness of the sampling. 
                             Higher values result in more random sampling, while lower values make the sampling more deterministic.
    Returns:
        int: The index of the selected action.
    """

    visit_counts = np.array([visit_counts for visit_counts, _ in distribution])
    visit_counts_exp = np.exp(visit_counts)
    policy = visit_counts_exp / np.sum(visit_counts_exp)
    policy = (policy ** (1 / temperature)) / (policy ** (1 / temperature)).sum()
    action_index = np.random.choice(range(len(policy)), p=policy)

    return action_index
