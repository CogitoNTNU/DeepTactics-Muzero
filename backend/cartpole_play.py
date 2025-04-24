from src.config import Config, get_cartpole_config
from src.utils.run_selfplay import run_selfplay
from src.utils.replay_buffer import ReplayBuffer
from src.networks.network import Network

# MuZero training is split into two independent parts:
# Network training and self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.


def muzero(config: Config):

    replay_buffer = ReplayBuffer(config=config)
    model = Network.load(config)

    # Watch model gradients
    # logger.watch_model(model)

    for i in range(config.training_episodes):
        run_selfplay(config, model, replay_buffer, None, i)

### Entry-point function
# muzero(get_cartpole_debug_config())
muzero(get_cartpole_config())
