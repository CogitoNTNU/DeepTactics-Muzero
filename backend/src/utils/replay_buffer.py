import random

class ReplayBuffer:
    def __init__(self):
        self.buffer_size: int = 10000
        self.buffer: list[dict] = []  #Stores game trajectories
        
    def update_buffer(self, game: Game):
        buffer_entry: dict[list, list, list, list, list] = {
            "states": game.states,
            "actions": game.actions,
            "rewards": game.rewards,
            "policies": game.policies,
            "values": game.values 
        }
        
        if len(self.buffer) == self.buffer_size: #Buffer reached max cap
            self.buffer.pop(0)
            self.buffer.append(buffer_entry)
            
       
    #Retrieves a random trajectory from the buffer 
    def get_game_trajectory(self) -> dict:
        entry = random.randint(0, len(self.buffer)-1)
        buffer_entry = self.buffer[entry]
        return buffer_entry
    
    #Returns the rest of the trajectory from a random postion in a game
    def get_positions(self, buffer_entry: dict, n_last_states: int = -999) -> dict:
        state = random.randint(0, len(buffer_entry)) #Selects a random state in the game
        if n_last_states == -999: # Not set
            return buffer_entry[state:] #Returns all subsequent states after the randomly selected state
        else:
            #This probably generates an index error
            return {key: value[state-n_last_states:state+1] for key, value in buffer_entry.items()} # Returns the n last states 
        
        
    #Retrieves a game trajectory from the replay buffer and returns the last n states.
    #If n not specified, random selection is done.
    def retrieve_game(self, n_last_states: int = -999) -> dict:
        buffer_entry = self.get_game_trajectory()
        return self.get_positions(buffer_entry, n_last_states) #Returns the last n states of a trajectory from the replay buffer
    
    
    
        