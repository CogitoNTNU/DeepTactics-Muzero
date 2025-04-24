from src.mcts.node import Node
from src.game.player import Player


def test_node_initialization():
    player = Player(is_board_game=False)
    node = Node(parent=None, policy_value=0.5, player=player, state=None)

    assert node.policy_value == 0.5
    assert node.to_play == player
    assert node.parent is None


def test_node_functions():
    node = Node(parent=None, policy_value=0.5, player=None, state=None)

    # Test expanded function
    node.children = {}  # No children
    assert not node.expanded()

    node.children = {1: "child"}  # One child
    assert node.expanded()

    # Test value function
    node.visits = 0
    node.value_sum = 0.0
    assert node.value() == 0.0

    node.visits = 10
    node.value_sum = 50.0
    assert node.value() == 5.0
