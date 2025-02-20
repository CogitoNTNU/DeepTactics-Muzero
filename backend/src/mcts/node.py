from typing import Dict, Optional


class Node:
    def __init__(self, parent: Optional['Node'], state: None, policy_value: float) -> None:
        self.children: Dict[int, Node] = {}
        self.parent = parent
        self.visits = 0
        self.value = 0
        self.state = state
        self.policy_value = policy_value

    def expanded(self) -> bool:
        return len(self.children) > 0
