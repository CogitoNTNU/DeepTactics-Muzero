from typing import Dict, Optional

class Node:
    """
    A node in the MCTS tree holding state, value statistics, and network priors.

    Attributes:
        children (Dict[int, Node]): Mapping from action indices to child nodes.
        parent (Optional[Node]): Parent node; None for root.
        visits (int): Number of visits.
        value_sum (float): Cumulative value.
        state (None): Game state (customize type as needed).
        policy_value (float): Prior probability or value from the policy network.
        to_play (int): Player identifier for this node.
        hidden_state: Hidden state from the representation network.
        reward (float): Reward from transitioning to this node.
    """
    def __init__(self, parent: Optional['Node'], state: None, policy_value: float, player: int) -> None:
        """
        Initializes an MCTS node.

        Args:
            parent (Optional[Node]): Parent node (None if root).
            state (None): Associated game state.
            policy_value (float): Policy network's prior for this node.
            player (int): The playes whose turn it is.
        """
        self.children: Dict[int, Node] = {}
        self.parent = parent
        self.visits = 0
        self.value_sum = 0
        self.state = state
        self.policy_value = policy_value
        self.to_play: int = player
        self.hidden_state = None # The hidden state of the representation network
        self.reward = 0
        
    def expanded(self) -> bool:
        """
        Checks if the node has been expanded.

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return len(self.children) > 0
    
    def value(self) -> float:
        """
        Computes the node's average value.

        Returns:
            float: Average value (value_sum / visits) or 0.0 if unvisited.
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
