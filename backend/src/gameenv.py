import gymnasium as gym
import ale_py
from src.mcts.node import Node
from src.game.player import Player

"""
class ActionHistory(object):
  def __init__(self, history: list[int], action_space_size: int, player: Player):
    self.history = list(history)
    self.action_space_size = action_space_size
    self.player = player

  def clone(self):
    return ActionHistory(self.history, self.action_space_size, self.player)

  def add_action(self, action: int):
    self.history.append(action)

  def last_action(self) -> int:
    return self.history[-1]

  def action_space(self) -> list[int]:
    return [i for i in range(self.action_space_size)]

  def to_play(self) -> int:
    return self.player
"""

class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self, gamefile: str): #'ALE/Breakout-v5'
        self.env = gym.make(gamefile, render_mode="human")
        self.obs, self.info = self.env.reset()
        self.episode_over: bool = False
        self.input_size = self.env.action_space

    def step(self, action):
        self.action = action
        self.obs, self.reward, terminal, truncated, info = self.env.step(action)        
        self.episode_over = terminal or truncated
        return self.reward, self.obs

    def close(self):
        self.env.close()
    
    def action_space(self):
        self.env.action_space

class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float, gamefile: str = 'CartPole-v1', is_cartpole: bool=True):
    self.environment = Environment(gamefile=gamefile)  # Game specific environment.
    self.action_history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.observations = []
    self.action_space_size = action_space_size
    self.discount = discount
    self.is_cartpole:bool = is_cartpole
    self.player = Player(is_cartpole=self.is_cartpole)

  def terminal(self) -> bool:
    return self.environment.episode_over

  def legal_actions(self) -> list[int]:
    if(self.is_cartpole):
      return [0, 1]
    else:
      return self.environment.get_possible_actions() #her må en othello env defienres på forhond
    
  def apply(self, action: int):
    reward, obs = self.environment.step(action)
    self.rewards.append(reward)
    print(self.action_history)
    self.action_history.append(int(action))
    print(self.action_history)
    self.observations.append(obs)
    self.player.change_player() #sjekk at denne ikke blir kaldt på før to_play men etter
    self.episode_over = self.terminal()

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.value() for child in root.children.values())
    action_space = (index for index in range(self.action_space_size))
    self.child_visits.append([root.children[a].value() / sum_visits if a in root.children else 0 for a in action_space])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return self.observations[state_index]


  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      # For simplicity the network always predicts the most recently received
      # reward, even for the initial representation network where we already
      # know this reward.
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      if current_index < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, last_reward, []))
    return targets

  def to_play(self) -> Player:
    return self.player

  def get_action_history(self) -> list[int]:
    return self.action_history
  
  def total_rewards(self):
    return sum(self.rewards)