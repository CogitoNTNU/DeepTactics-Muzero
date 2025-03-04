import random
import numpy as np
from src.gameenv import Game

class ReplayBuffer:
    def __init__(self):
        self.buffer_size: int = 10000
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
    
    #Returns the rest of the trajectory from a random postion in a game
    def get_positions(self, buffer_entry: list, history_length: int = 0, nr_of_next_states: int = 5) -> tuple:
        game_state = random.randint(0, len(buffer_entry)) #Selects a random state in the game
       
        # Returns a tuple. First element is state concatenated with the last n states. Second element is the next n states.
        return (np.concatenate(buffer_entry[game_state-history_length:game_state+1], axis=-1)
                , buffer_entry[game_state+1:game_state+nr_of_next_states+1])         
       
    #Retrieves a game trajectory from the replay buffer and returns the last n states.
    def retrieve_game(self, n_last_states: int = 0, nr_of_next_states: int = 5) -> tuple:
        buffer_entry = self.get_game_trajectory()
        return self.get_positions(buffer_entry, n_last_states, nr_of_next_states) #Returns the last n states of a trajectory from the replay buffer
    
    
    
        