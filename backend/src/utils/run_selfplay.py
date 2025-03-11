


from src.config import Config
from src.mcts.play_game import play_game
from src.utils.replay_buffer import ReplayBuffer
from src.utils.shared_storage import SharedStorage


def run_selfplay(config: Config, 
                 storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    
    #while True: # in theory, this should be a job (i.e. thread) that runs continuously
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.update_buffer(game)