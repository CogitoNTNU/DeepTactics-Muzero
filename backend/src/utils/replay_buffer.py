import random
import numpy as np
from src.config import Config
from src.gameenv import Game

class ReplayBuffer:
    def __init__(self):
        self.buffer_size: int = 10000
        self.batch_size = Config().batch_size
        self.buffer: list[list[dict]] = []  #Stores game trajectories
        
    def update_buffer(self, game: Game):
        if len(self.buffer) == self.buffer_size: #Buffer reached max cap
            self.buffer.pop(0)
        self.buffer.append(game)
       
    #Retrieves a random trajectory from the buffer 
    def get_game_trajectory(self) -> dict:
        entry = random.randint(0, len(self.buffer)-1)
        buffer_entry = self.buffer[entry]
        return buffer_entry
    
    def sample_batch(self, num_unroll_steps: int, td_steps: int, action_space_size: int):
        games: list[Game] = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), 
                    g.get_action_history()[i:i + num_unroll_steps],
                    g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]
    
    def sample_game(self) -> Game:
        
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.choice(range(len(self.buffer)))]

    def sample_position(self, game: Game) -> int:
        
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(range(len(game.rewards) - 1))
    
    #Returns the rest of the trajectory from a random postion in a game
    def get_positions(self, buffer_entry: list, history_length: int = 0, nr_of_next_states: int = 5) -> tuple:
        game_state = random.randint(0, len(buffer_entry)-1) #Selects a random state in the game
        # history_states = []
        if game_state+1 <= history_length:
            # We need to pad with (history_length + 1) - (game_state + 1) states.
            pad_count = history_length - game_state
            pad = [buffer_entry[0]["state"]] * pad_count  # Copy the initial state as many times as needed
            history_list = pad + [entry["state"] for entry in buffer_entry[:game_state + 1]]
            history_states = np.concatenate(history_list, axis=-1)
        else:
            # If enough states exist, take the last (history_length + 1) states.
            history_list = [entry["state"] for entry in buffer_entry[game_state - history_length: game_state+1]]
            history_states = np.concatenate(history_list, axis=-1)
            
        # Returns a tuple. First element is state concatenated with the last n states. Second element is the next n states.
        next_state = game_state+1

        return (history_states, buffer_entry[next_state:next_state+nr_of_next_states])
       
    #Retrieves a game trajectory from the replay buffer and returns the last n states.
    def retrieve_game(self, n_last_states: int = 0, nr_of_next_states: int = 5) -> tuple:
        buffer_entry = self.get_game_trajectory()
        return self.get_positions(buffer_entry, n_last_states, nr_of_next_states) #Returns the last n states of a trajectory from the replay buffer
    
    def last_game(self) -> Game:
        return self.buffer[-1]