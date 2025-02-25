from src.game.player import Player
from src.game.action import Action
from src.mcts.play_game import play_game
from src.mcts.node import Node
from src.mcts.expand import expand_node
from src.mcts.puct import select_child
from src.mcts.backpropagate import backpropagate
from src.utils.minmaxstats import MinMaxStats
from src.config import Config
from src.game.action_history import ActionHistory
from backend.architecture.src.network import RepresentationNetwork, PredictionNetwork


player = Player()
config = Config()
network = RepresentationNetwork()

play_game(config, network)