class Player(object):
    def __init__(self, is_cartpole:bool = True):
        self.is_cartpole: bool = is_cartpole
        self.turn_multiplier = 1
    
    def change_player(self):
        if(self.is_cartpole):
            self.turn_multiplier = 1
        else:
            self.turn_multiplier *= -1

    def get_turn_multiplier(self): 
        return self.turn_multiplier
"""
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index
    
    def __str__(self):
        return str(self.index)
"""