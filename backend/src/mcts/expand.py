from src.networks.network import NetworkOutput
from src.mcts.node import Node
from src.game.player import Player

def expand_node(node: Node, to_play: Player, network_output: NetworkOutput):
    node.to_play = to_play.get_turn_multiplier()
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    
    for action, policy_logit in enumerate(network_output.policy_logits[0]):
        node.children[action] = Node(parent=node, policy_value = policy_logit, player=node.to_play, state=None)
