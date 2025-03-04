from src.game.player import Player
from src.mcts.play_game import play_game
from src.config import Config
from src.networks.network import Network


config = Config()
network = Network(config)

play_game(Config(), network)