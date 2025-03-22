from typing import List
from src.utils.minmaxstats import MinMaxStats
from src.mcts.node import Node
from src.game.player import Player


def backpropagate(search_path: List[Node], value: float, to_play: Player, discount: float, min_max_stats: MinMaxStats) -> None:
    """
    Propagates the value and updates statistics along the search path in the Monte Carlo Tree Search (MCTS).

    This function traverses the search path in reverse order, updating the value sum, visit count, 
    and min-max statistics for each node. It also applies a discount factor to the value as it moves 
    up the tree and adjusts the value based on the player's turn.

    Args:
        search_path (List[Node]): The list of nodes representing the path from the root to the current node.
        value (float): The value to propagate back through the tree.
        to_play (Player): The player whose turn it is, used to determine value adjustment.
        discount (float): The discount factor applied to the value as it propagates up the tree.
        min_max_stats (MinMaxStats): An object to track and normalize the min and max values of the tree.

    Returns:
        None
    """

    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play.get_turn_multiplier() else -value #//TODO: Uproblematisk for cartpole, men her må man kanskje endre i othello
        node.visits += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value
