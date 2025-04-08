from abc import ABC, abstractmethod
import gymnasium as gym
from src.mcts.node import Node
from src.game.player import Player
import numpy as np
from typing import List
import torch


class Environment(ABC):
    """The environment MuZero is interacting with."""
    @abstractmethod
    def __init__(self):
        pass

    def step(self, action):
        self.action = action
        obs, self.reward, terminal, truncated, info = self.env.step(action.item())
        self.obs = self._transform_obs(obs)
        self.episode_over = terminal or truncated
        return self.reward, self.obs

    def close(self):
        self.env.close()

    def action_space(self):
        self.env.action_space

    @abstractmethod
    def get_actions(self) -> list[int]:
        pass

    @abstractmethod
    def _transform_obs(self, obs) -> torch.Tensor:
        pass

class TicTacToe(Environment):
    def __init__(self):
        self.env = gym.make("tictactoe-v0")
        obs, self.info = self.env.reset()
        self.obs = self._transform_obs(obs)
        self.episode_over: bool = False
        self.input_size = self.env.action_space
        self.is_board_game = True
        self.player = Player(self.is_board_game)
        
    def get_actions(self) -> List[int]:
        return self.env.get_actions()
    
    def _transform_obs(self, obs: np.ndarray) -> torch.Tensor:
        obs = obs.flatten()
        torch.from_numpy(obs)

class CartPole(Environment):
    def __init__(self):  #'ALE/Breakout-v5'
        self.env = gym.make("CartPole-v1", sutton_barto_reward=True)  # """render_mode='human',"""
        obs, self.info = self.env.reset()
        self.obs = self._transform_obs(obs)
        self.episode_over: bool = False
        self.input_size = self.env.action_space
        self.is_board_game = False
        self.player = Player(self.is_board_game)

    def get_actions(self) -> list[int]:
        return [0,1]
    
    def _transform_obs(self, obs):
        return obs
    
    

# TODO: Hvilket othello env bruker vi? Finner det ikke i requirements.txt, det er fordi othello enviromentet er i othello.py
"""
class Othello(Environment):
    def __init__(self, gamefile: str):
        self.env = gym.make(gamefile)  # render_mode='human',
        self.obs, self.info = self.env.reset()
        self.episode_over: bool = False
        self.input_size = self.env.action_space
        self.is_continious = False
"""


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float, config):
        self.environment = config.game_class()  # Game specific environment.
        self.action_history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.observations = []
        self.action_space_size = action_space_size  # self.environment.action_space()#
        self.discount = discount

    def terminal(self) -> bool:
        return self.environment.episode_over

    def legal_actions(self) -> list[int]:
        return self.environment.get_actions()  # _is_valid_action()#get_possible_actions() #her må en othello env defienres på forhond
    
    def change_player(self):
        self.environment.player.change_player()

    def apply(self, action: int):
        # reward, obs observation, winner, game over indicator, truncated= self.environment.step(action)
        self.observations.append(self.environment.obs)
        reward, obs = self.environment.step(action)
        self.rewards.append(reward)
        self.action_history.append(int(action))
        self.change_player()  # sjekk at denne ikke blir kaldt på før to_play men etter

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visits for child in root.children.values())
        action_space = (index for index in range(self.action_space_size))
        self.child_visits.append(
            [
                root.children[a].visits / sum_visits
                if a in root.children
                else 0
                for a in action_space
            ]
        )

        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        # Game specific feature planes.
        return self.observations[state_index]

    def make_target(
        self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player
    ):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            if not self.environment.is_board_game:
                bootstrap_index = current_index + td_steps
                if bootstrap_index < len(self.root_values):
                    value = self.root_values[bootstrap_index] * self.discount**td_steps
                else:
                    value = 0.0
            else:
                is_even_length = len(self.rewards) % 2 == 0
                is_current_even = current_index % 2 == 0

                if is_even_length == is_current_even:
                    value = self.rewards[-1]
                else:
                    value = -self.rewards[-1]
                """
                s = [x,  o,  x,  o,  x, o,  x]
                r =   [0,  0,  0,  0, 0,  1]
                v = [-1, 1, -1,  1, -1, 1]
                p = v
                """

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += (
                    reward * self.discount**i
                )  # pytype: disable=unsupported-operands

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0.0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0.0, last_reward, []))

        return targets

    def to_play(self) -> Player:
        return self.environment.player

    def get_action_history(self) -> list[int]:
        return self.action_history

    def total_rewards(self):
        return sum(self.rewards)

