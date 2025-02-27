from typing import Dict, Optional

class Node:
    def __init__(self, parent: Optional['Node'], state: None, policy_value: float, player: int) -> None:
        self.children: Dict[int, Node] = {}
        self.parent = parent
        self.visits = 0
        self.value_sum = 0
        self.state = state
        self.policy_value = policy_value
        self.to_play = player
        self.hidden_state = None #the hidden state of the representation network
        self.reward = 0
        
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visits == 0:
            return 0
        return self.value / self.visits
    
