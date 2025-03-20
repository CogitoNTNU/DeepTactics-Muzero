from typing import List
from src.utils.minmaxstats import MinMaxStats
from src.mcts.node import Node
from src.game.player import Player


def backpropagate(search_path: List[Node], value: float, to_play: Player, discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play.get_turn_multiplier() else -value #//TODO: Uproblematisk for cartpole, men her må man kanskje endre i othello
        node.visits += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value
