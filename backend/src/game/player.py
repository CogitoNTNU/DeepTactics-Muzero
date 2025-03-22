class Player(object):
    """
    Represents a player in the game, managing whos turn it is to move.

    Attributes:
        is_cartpole (bool): Indicates if the player is in CartPole mode.
        turn_multiplier (int): A multiplier used to determine whose turn it is to move.
    """
    def __init__(self, is_cartpole: bool = True) -> None:
        """
        Constructor for the Player class.

        Args:
            is_cartpole (bool, optional): Indicates if the player is in CartPole mode. Defaults to True.
        """
        self.is_cartpole: bool = is_cartpole
        self.turn_multiplier = 1
    
    def change_player(self):
        """
        Changes the turn multiplier to indicate whose turn it is to move.
        """
        if(self.is_cartpole):
            self.turn_multiplier = 1
        else:
            self.turn_multiplier *= -1

    def get_turn_multiplier(self): 
        """
        Returns the turn multiplier used to determine whose turn it is to move.

        Returns:
            int: The turn multiplier.
        """
        return self.turn_multiplier