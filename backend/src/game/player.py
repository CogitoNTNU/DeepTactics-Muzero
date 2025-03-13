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