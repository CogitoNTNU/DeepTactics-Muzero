from src.networks.network import NetworkOutput
from src.mcts.node import Node
from src.game.player import Player

def expand_node(node: Node, to_play: Player, network_output: NetworkOutput) -> None:
    """
    Expands a node in the Monte Carlo Tree Search (MCTS) by adding child nodes based on the network's output.

    This function sets the current node's player, hidden state, and reward, and creates child nodes
    for each possible action using the policy logits provided by the network output.

    Args:
        node (Node): The node to be expanded.
        to_play (Player): The player whose turn it is, used to set the node's player.
        network_output (NetworkOutput): The output of the network, containing the hidden state, reward,
                                        and policy logits for the current node.

    Returns:
        None
    """
    node.to_play = to_play.get_turn_multiplier()
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    
    for action, policy_logit in enumerate(network_output.policy_logits[0]):
        node.children[action] = Node(parent=node, policy_value = policy_logit, player=node.to_play, state=None)
