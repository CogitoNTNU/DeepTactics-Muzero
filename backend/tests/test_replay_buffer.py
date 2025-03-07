from src.utils.replay_buffer import ReplayBuffer
import random
import numpy as np


def generate_dummy_game(num_steps, state_shape, num_actions):
    game_trajectory = []
    for step in range(num_steps):
        state = np.zeros(state_shape)
        reward = 1
        value = 1
        raw_policy = [random.random() for _ in range(num_actions)]
        total = sum(raw_policy)
        policy = [p / total for p in raw_policy]
        action = 1
        step_data = {
            "state": state,
            "value": value,
            "policy": policy,
            "action": action,
            "reward": reward
        }
        game_trajectory.append(step_data)
    return game_trajectory

def test_replay_buffer():
    num_games, num_steps, state_shape, num_actions = 5, 100, (3,3,5), 4
    replay_buffer = ReplayBuffer()
    for i in range(num_games):
        game = generate_dummy_game(num_steps, state_shape, num_actions)
        replay_buffer.update_buffer(game)    
    assert len(replay_buffer.buffer) == num_games, "Length of buffer not equal to number of games"
    
    game_1_state_1 = replay_buffer.buffer[0][0]
    assert game_1_state_1["reward"] == 1, "Reward should be 1"
    
    game_trajectory = replay_buffer.get_game_trajectory()
    assert len(game_trajectory) == num_steps, "Length of game trajectory not equal to number of steps"
    
    history_length = 10
    nr_of_next_states = 5
    history, next_states = replay_buffer.get_positions(buffer_entry=game_trajectory, history_length=history_length, nr_of_next_states=nr_of_next_states)
    assert history.shape == (state_shape[0], state_shape[1], state_shape[2]*(history_length+1)), "Current state concatenated with its history states not of correct shape"
    
    # Should the number of next state also be of fixed size?
    # assert len(next_states) == nr_of_next_states, "Length of next states not equal to 5"
    
    
    
    
    