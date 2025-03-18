from src.networks.network import Network
from src.config import Config
from src.mcts.node import Node
from src.mcts.add_dirichlet import add_exploration_noise
from src.mcts.expand import expand_node
from src.mcts.run_mcts import run_mcts
from src.mcts.select_action import select_action
from src.gameenv import Game

def play_game(config: Config, network: Network) -> Game:
    game = Game(config.action_space_size, config.discount)
    print("Playing the game")
    # game.history should be a list of actions taken.
    
    while not game.terminal() and len(game.action_history().history) < config.max_moves:
        # Player is always 1 becuase cartpole only has one player
        root = Node(None, state = None, policy_value=0, player=1)
        
        current_observation = game.environment.obs
        
        # inital_inference should give the inital policy, value and hiddenstate (from representation network)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation))

        add_exploration_noise(config, root)

        run_mcts(config, root, game.action_history(), network)
        # select action, but with respect to the temperature
        action = select_action(config, len(game.action_history().history), root, network)

        game.apply(action)
        game.store_search_statistics(root)
        game.environment.env.render()

    return game