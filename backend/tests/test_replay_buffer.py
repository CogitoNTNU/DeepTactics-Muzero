from src.utils.replay_buffer import ReplayBuffer
import random
import numpy as np
from src.config import Config
from src.gameenv import Game
import torch


def generate_dummy_game(num_steps, state_shape, num_actions, game: Game):
    for step in range(num_steps):
        state = np.zeros(state_shape)
        reward = 1
        value = 1
        action = 1
        child_visits = 20
        
        game.observations.append(state)
        game.rewards.append(reward)
        game.root_values.append(value)
        game.action_history.append(action)
        game.child_visits.append(child_visits)
    return game

def test_replay_buffer():
    config = Config()
    num_games, num_steps, td_steps, state_shape, action_space_size = 1, 10, 5, (3,3,5), 4
    replay_buffer = ReplayBuffer(config=config)
    for i in range(num_games):
        game = Game(action_space_size=action_space_size, discount=config.discount)
        game = generate_dummy_game(num_steps, state_shape, action_space_size, game)
        replay_buffer.update_buffer(game)    
    assert len(replay_buffer.buffer) == num_games, "Length of buffer not equal to number of games"
    
    batch = replay_buffer.sample_batch(num_unroll_steps=num_steps, td_steps=td_steps, action_space_size=action_space_size)
    assert len(batch) == config.batch_size, "Length of batch is not equal to batch size"
    
    image = batch[0][0]
    actions = batch[0][1]
    targets = batch[0][2]
    value_target = targets[0]
    reward_target = targets[1]
    policy_target = targets[2]
    assert image.shape == state_shape, "Image shape not same as the defined state shape"
    # assert len(actions) == num_steps, "Number of steps should be equal to number of actions, but it is not"
    
    # How to test the target values when they depend on "state_index" in "make_target" function 
    # as "state_index" is sample randomly in "sample_position" function? The same applies for
    # checking the length of of actions.