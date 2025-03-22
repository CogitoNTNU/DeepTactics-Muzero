from src.config import Config
from src.mcts.play_game import play_game
from src.networks.network import Network
from src.utils.replay_buffer import ReplayBuffer

def run_selfplay(config: Config, model: Network, replay_buffer: ReplayBuffer):
    #while True: # in theory, this should be a job (i.e. thread) that runs continuously
    tot_steps = 0
    for i in range(config.training_interval):
        #print(f"Starting game: {i}")
        game, steps = play_game(config=config, network=model)
        replay_buffer.update_buffer(game)
        tot_steps += steps
    
    print("avg steps survived: ", tot_steps/config.training_interval )