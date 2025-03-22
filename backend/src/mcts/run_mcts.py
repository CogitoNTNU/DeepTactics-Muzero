from src.mcts.backpropagate import backpropagate
from src.mcts.expand import expand_node
from src.mcts.puct import select_child
from src.networks.network import Network
from src.utils.minmaxstats import MinMaxStats
from src.config import Config
from src.mcts.node import Node
from src.game.player import Player

def run_mcts(config: Config, root: Node, to_play: Player, network: Network) -> None:
    """
    Runs Monte Carlo Tree Search (MCTS) from the given root node.

    For a fixed number of tree searches, this function:
      - Traverses the tree by selecting children based on PUCT scores.
      - Uses the network's recurrent inference to expand a leaf node.
      - Backpropagates the value estimate to update the statistics of nodes along the search path.

    Args:
        config (Config): Configuration parameters for the MCTS, including known bounds, discount factor,
                         exploration constants, and number of tree searches.
        root (Node): The root node from which the MCTS starts.
        to_play (Player): The player object that tracks whose turn it is.
        network (Network): The MuZero network used for inference (both initial and recurrent).

    Returns:
        None
    """
    min_max_stats = MinMaxStats(config.known_bounds)
    for _ in range(config.n_tree_searches):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            search_path.append(node)
            to_play.change_player()
            
        # Use the dynamics function to obtain the 
        # next hidden state given the action and previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state, action)

        expand_node(node, to_play, network_output)

        backpropagate(search_path,network_output.value, to_play, config.discount, min_max_stats)

