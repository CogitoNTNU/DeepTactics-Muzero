from typing import List, Union
from src.networks.network import NetworkOutput
from src.mcts.node import Node
from src.game.player import Player
from src.game.action import Action
from src.gameenv import ActionHistory
import math

# Feltene node.to_play, node.hidden_state, node.reward osv...
# må legges til Node-klassen eller behandles på en annen måte

# NetworkOutput får vi fra initial_inference fra network...


def expand_node(node: Node, to_play: Player, actions: List[Union[Action, int]], network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward

    actions = [Action(a) if isinstance(a, int) else a for a in actions]
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(node, policy_value = p / policy_sum, player=node.to_play, state=None)
