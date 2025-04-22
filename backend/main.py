from src.config import Config, get_cartpole_config, get_cartpole_debug_config
from src.utils.train_network import train_network
from src.utils.run_selfplay import run_selfplay
from src.utils.replay_buffer import ReplayBuffer
from src.networks.network import Network
from src.utils.wandb_logger import WandbLogger

import time
import numpy as np
import wandb

# MuZero training is split into two independent parts:
# Network training and self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.


def muzero(config: Config):
    # Initialize wandb
    wandb_config = {
        "learning_rate": config.learning_rate,
        "learning_rate_decay": config.learning_rate_decay,
        "learning_rate_decay_steps": config.learning_rate_decay_steps,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "num_unroll_steps": config.num_unroll_steps,
        "td_steps": config.td_steps,
        "training_episodes": config.training_episodes,
        "epochs": config.epochs,
        "action_space_size": config.action_space_size,
        "hidden_layer_size": config.hidden_layer_size,
        "observation_space_size": config.observation_space_size,
    }

    logger = WandbLogger(project_name="muzero-cartpole", config=wandb_config)

    replay_buffer = ReplayBuffer(config=config)
    model = Network.load(config)

    # Watch model gradients
    logger.watch_model(model)

    rewards = []
    losses = []
    moving_averages = []

    t = time.time()
    try:
        for i in range(config.training_episodes):
            # self-play
            run_selfplay(config, model, replay_buffer, logger, i)

            # training
            loss = train_network(config, model, replay_buffer, i, logger)

            # Save model periodically
            if i % 100 == 0:
                model.save()

    except KeyboardInterrupt:
        model.save()
        logger.finish()
        return

    logger.finish()
    return model


### Entry-point function
# muzero(get_cartpole_debug_config())
muzero(get_cartpole_config())
