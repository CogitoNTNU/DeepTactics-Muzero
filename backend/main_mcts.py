from src.game.player import Player
from src.mcts.play_game import play_game
from src.config import Config
from src.networks.network import Network
from src.networks.config import MuZeroConfig


config = MuZeroConfig()
network = Network(config)

play_game(Config(), network)
