from typing import List

from backend.src.game.action import Action
from backend.src.game.player import Player

class ActionHistory(object):

    """Simple history container used inside the search.
       Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        
        self.history.append(action)

    def last_action(self) -> Action:
        
        return self.history[-1]

    def action_space(self) -> List[Action]:
        
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        
        return Player(1)