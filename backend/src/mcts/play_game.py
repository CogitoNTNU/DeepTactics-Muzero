from src.networks.network import Network
from src.config import Config
from src.mcts.node import Node
from src.mcts.add_dirichlet import add_exploration_noise
from src.mcts.expand import expand_node
from src.mcts.run_mcts import run_mcts
from src.mcts.select_action import select_action
from src.gameenv import Game

import torch


def self_made_tree(root: Node):
            print(root.visits, root.reward, root.value())
            for key, child in root.children.items():  # Iterate over dictionary
                base = "    "
                print(base+f"visits: {child.visits}, rewards: {child.reward}, value: {child.value()}")
                if child.children:  # Check if child has further children
                    self_made_children(child, base)

def self_made_children(node: Node, base):
    base += "    "
    for key, child in node.children.items():
        print(base+f"visits: {child.visits}, reward: {child.reward}, value: {child.value()}")
        if not child.children and key == max(node.children.keys()):  
            base = base[4:]  # Adjust indentation only at the last child
            print("")
        else:
            self_made_children(child, base)
def play_game(config: Config, network: Network) -> Game:
    """
    Plays a game using a the MCTS.

    Runs the game loop until the game terminates or the maximum number of moves is reached.
    At each step, it:
      - Creates a root node for MCTS.
      - Uses the network's initial inference to expand the node.
      - Adds exploration noise.
      - Runs MCTS and selects an action.
      - Applies the action and stores search statistics.

    Args:
        config (Config): Configuration parameters for the game and MuZero.
        network (Network): The MuZero network for inference.

    Returns:
        tuple: A tuple containing the completed Game instance and the number of steps taken.
    """
    with torch.no_grad():
        game = Game(config.action_space_size, config.discount, gamefile=config.game_name)
        # game.history should be a list of actions taken.
        steps = 0
        
        while not game.terminal() and len(game.get_action_history()) < config.max_moves:
            steps += 1
            # Player is always 1 becuase cartpole only has one player
            root = Node(parent=None, state = None, policy_value=0, player=1)
            
            current_observation = game.environment.obs
            
            # inital_inference should give the inital policy, value and hiddenstate (from representation network)
            expand_node(node=root, to_play=game.to_play(), network_output=network.initial_inference(current_observation))
            
            add_exploration_noise(config=config, node=root)

            run_mcts(config=config, root=root, to_play=game.to_play(), network=network)
            # select action, but with respect to the temperature
            action = select_action(config=config, num_moves=len(game.get_action_history()), node=root, network=network)
            game.apply(action=action)
            game.store_search_statistics(root=root)
            if config.render:
                game.environment.env.render()
        #self_made_tree(root)
        
        
        
        return game, steps