from backend.src.mcts import backpropagate
from backend.src.mcts.expand import expand_node
from backend.src.mcts.puct import select_child
from backend.src.utils.minmaxstats import MinMaxStats
from src.config import Config
from src.mcts.node import Node
from src.game.action_history import ActionHistory


def run_mcts(config: Config, 
             root: Node, 
             action_history: ActionHistory,
             network: Network):
    
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, 
                      network_output.value, 
                      history.to_play(),
                      config.discount, 
                      min_max_stats)