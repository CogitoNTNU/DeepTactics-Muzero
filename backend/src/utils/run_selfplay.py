from src.config import Config
from src.mcts.play_game import play_game
from src.networks.network import Network
from src.utils.replay_buffer import ReplayBuffer

def run_selfplay(config: Config, model: Network, replay_buffer: ReplayBuffer) -> None:
    """
    Runs self-play games and stores trajectories in the replay buffer.

    For a number of games defined by `config.training_interval`, this function:
      - Plays a game using the current model.
      - Updates the replay buffer with the resulting game trajectory.
      - Tracks the total number of steps survived across games.
      - Prints the average steps per game at the end.

    Args:
        config (Config): Configuration object with training parameters.
        model (Network): The network model used for self-play inference.
        replay_buffer (ReplayBuffer): The replay buffer for storing game trajectories.
    """
    #while True: # in theory, this should be a job (i.e. thread) that runs continuously
    tot_steps = 0
    for i in range(config.training_interval):
        #print(f"Starting game: {i}")
        game, steps = play_game(config, model)
        replay_buffer.update_buffer(game)
        tot_steps += steps
    
    print("avg steps survived: ", tot_steps/config.training_interval )