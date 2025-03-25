import random
import numpy as np
from src.config import Config
from src.gameenv import Game

class ReplayBuffer:
    """
    A replay buffer for storing game trajectories and sampling training batches.

    Attributes:
        buffer_size (int): Maximum number of trajectories to store.
        batch_size (int): Number of games to sample per training batch.
        buffer (list[list[dict]]): List of game trajectories.
    """
    def __init__(self, config: Config):
        """
        Initializes the ReplayBuffer with a fixed buffer size and batch size.
        """
        self.buffer_size: int = config.buffer_size
        self.batch_size = config.batch_size
        self.buffer: list[list[dict]] = []  #Stores game trajectories
        
    def update_buffer(self, game: Game) -> None:
        """
        Adds a game trajectory to the buffer, removing the oldest entry if at capacity.

        Args:
            game (Game): The game trajectory to add.
        """
        if len(self.buffer) == self.buffer_size: #Buffer reached max cap
            print("Had to pop, buffer full")
            self.buffer.pop(0)
        self.buffer.append(game)
       
    #Retrieves a random trajectory from the buffer 
    def get_game_trajectory(self) -> dict:
        """
        Retrieves a random game trajectory from the buffer.

        Returns:
            dict: A randomly selected game trajectory.
        """
        entry = random.randint(0, len(self.buffer)-1)
        buffer_entry = self.buffer[entry]
        return buffer_entry
    
    def sample_batch(self, num_unroll_steps: int, td_steps: int, action_space_size: int) -> list:
        """
        Samples a batch of training data from the stored game trajectories.

        For each sampled game, a random position is chosen and the corresponding
        image, action history, and target values are prepared.

        Args:
            num_unroll_steps (int): Number of unroll steps for training.
            td_steps (int): Number of temporal difference steps.
            action_space_size (int): Size of the action space.

        Returns:
            list: A list of tuples, each containing:
                - The concatenated state image.
                - The action history for the unroll steps.
                - The target values for training.
        """
        games: list[Game] = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.get_action_history()[i:i + num_unroll_steps], 
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play())) for (g, i) in game_pos]
    
    def sample_game(self) -> Game:
        """
        Samples a game trajectory from the buffer uniformly.

        Returns:
            Game: A randomly selected game trajectory.
        """
        return self.buffer[np.random.choice(range(len(self.buffer)))]

    def sample_position(self, game: Game) -> int:
        """
        Samples a position within a game trajectory uniformly.

        Args:
            game (Game): The game trajectory to sample from.

        Returns:
            int: A randomly selected position index in the game trajectory.
        """
        return np.random.choice(range(len(game.rewards) - 1))
    
    #Returns the rest of the trajectory from a random postion in a game
    def get_positions(self, buffer_entry: list, history_length: int = 0, nr_of_next_states: int = 5) -> tuple:
        """
        Retrieves concatenated historical states and the following states from a trajectory.

        If the sampled position is less than the required history length, the initial state is
        repeated as needed for padding.

        Args:
            buffer_entry (list): A game trajectory (list of dictionaries with state information).
            history_length (int, optional): Number of previous states to include. Defaults to 0.
            nr_of_next_states (int, optional): Number of future states to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing:
                - history_states (np.ndarray): Concatenated states representing history.
                - next_states (list): The next states following the sampled position.
        """
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
       
    def retrieve_game(self, n_last_states: int = 0, nr_of_next_states: int = 5) -> tuple:
        """
        Retrieves a game trajectory from the buffer and returns its last n states and the next states.

        Args:
            n_last_states (int, optional): Number of last states to include in the history. Defaults to 0.
            nr_of_next_states (int, optional): Number of subsequent states to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing the concatenated history states and the subsequent states.
        """
        buffer_entry = self.get_game_trajectory()
        return self.get_positions(buffer_entry, n_last_states, nr_of_next_states) #Returns the last n states of a trajectory from the replay buffer
    
    def last_game(self) -> Game:
        """
        Returns the most recently added game trajectory.

        Returns:
            Game: The last game trajectory in the buffer.
        """
        return self.buffer[-1]