import gymnasium as gym
import ale_py
from src.mcts.node import Node
from src.game.action import Action 

######################---Disse skal nok ligge i en annen fil, placeholders settes her---######################


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
  

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: list[Action], action_space_size: int, player: Player):
    self.history = list(history)
    self.action_space_size = action_space_size
    self.player = player

  def clone(self):
    return ActionHistory(self.history, self.action_space_size, self.player)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> list[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> int:
    return self.player
  
##############################################################################################################


class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self, gamefile: str): #'ALE/Breakout-v5'
        self.env = gym.make(gamefile, render_mode="human") 
        self.obs, self.info = self.env.reset()
        self.episode_over: bool = False
        self.input_size = self.env.action_space


    def step(self, action):
        action = self.env.action_space.sample() #remove line, now its random
        self.obs, self.reward, terminal, truncated, info = self.env.step(action)
        self.episode_over = terminal and truncated 

    def close(self):
        self.env.close()
    
    def action_space(self):
      self.env.action_space

class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float, gamefile: str = 'CartPole-v1', is_cartpole: bool=True):
    self.environment = Environment(gamefile=gamefile)  # Game specific environment.
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount
    self.is_cartpole:bool = is_cartpole
    self.player = Player(is_cartpole=self.is_cartpole)

  def terminal(self) -> bool:
    return self.environment.episode_over

  def legal_actions(self) -> list[Action]:
    if(self.is_cartpole):
      return [0, 1]
    else:
      return self.environment.get_possible_actions() #her må en othello env defienres på forhond
    
  def apply(self, action: Action):
    reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)
    self.player.change_player() #sjekk at denne ikke blir kaldt på før to_play men etter

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.value() for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].value() / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return [1,1,1,1] #self.environment.obs


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

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size, self.player)