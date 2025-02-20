from src.config import Config
from src.mcts.node import Node
from src.mcts.add_dirichlet import add_exploration_noise
from src.mcts.expand import expand_node
from src.mcts.run_mcts import run_mcts
from src.mcts.select_action import select_action

def play_game(config: Config, network):
    
    game = config.new_game()
    
    
    # game.history should be a list of actions taken.
    while not game.terminal() and len(game.history) < config.max_moves:
        
        root = Node()
        current_observation = game.make_image(-1) # should get the observation (state) from the env.
        expand_node(root, game.to_play(), game.legal_actions(), network.inital_inference(current_observations)) # inital_inference should give the inital policy, value and hiddenstate (from representation network)
        
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network) # select action, but with respect to the temperature
        game.apply(action)
        game.store_search_statistics(root)
    
    return game
        