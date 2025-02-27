from src.networks.network import Network
from src.config import Config
from src.mcts.node import Node
from src.mcts.add_dirichlet import add_exploration_noise
from src.mcts.expand import expand_node
from src.mcts.run_mcts import run_mcts
from src.mcts.select_action import select_action
from src.gameenv import Game


def play_game(config: Config, network: Network):

    game = Game(config.action_space, config.discount)

    # game.history should be a list of actions taken.
    while not game.terminal() and len(game.action_history().history) < config.max_moves:

        root = Node(None, None, 0)
        # should get the observation (state) from the env.
        current_observation = game.make_image(-1)
        # inital_inference should give the inital policy, value and hiddenstate (from representation network)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation))

        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        # select action, but with respect to the temperature
        action = select_action(config, len(
            game.action_history().history), root, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game
