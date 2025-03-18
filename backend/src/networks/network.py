
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Dict
from src.networks.residualblock import ResBlock


class NetworkOutput(NamedTuple):
    """
    The output from initial_inference or recurrent_inference:
      - value: scalar value estimate
      - reward: scalar immediate reward
      - policy_logits: dictionary {Action(a): prob_a}
      - policy_tensor: raw tensor of shape [batch_size, action_space_size]
      - hidden_state: shape [batch_size, hidden_layer_size]
    """
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: list[float]
    hidden_state: torch.Tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Expects a config object with:
          - observation_space_size: int
          - action_space_size: int
          - hidden_layer_size: int
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.hidden_layer_size = config.hidden_layer_size
        self.action_space_size = config.action_space_size

        # Representation: from raw observation to hidden state.
        self.representation = nn.Sequential(
            nn.Linear(config.observation_space_size, config.hidden_layer_size),
            nn.ReLU(),
        )

        # Value head: predicts scalar value from hidden state.
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, 1),
            nn.ReLU()
        )

        # Policy head: predicts action probabilities from hidden state.
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.action_space_size),
            nn.Softmax(dim=-1)
        )

        # Dynamics: given [hidden_state, action one-hot] -> next hidden state
        self.dynamics = nn.Sequential(
            nn.Linear(config.hidden_layer_size +
                      config.action_space_size, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size)
        )

        # Reward head: same input as dynamics, but outputs a scalar reward.
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(config.hidden_layer_size, 1),
            nn.ReLU()
        )

        self.tot_training_steps = 0

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        """
        For the first step from an environment observation.
        The reward is set to zero because no action has yet been taken.
        """
        # Convert list to tensor if necessary
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)

        if observation.dim() > 2:  # If it's mistakenly shaped like an image
            observation = observation.flatten()

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        hidden_state = self.representation(observation) # TODO make this return correct values
        #hidden_state = torch.rand(
            #(observation.shape[0], self.hidden_layer_size), device=observation.device)
        value = self.value_head(hidden_state)
        policy = self.policy_head(hidden_state)

        # Reward is zero at the root.
        reward = torch.zeros(
            (observation.shape[0], 1), device=observation.device, dtype=observation.dtype)
        return NetworkOutput(value, reward, policy, hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: int) -> NetworkOutput:
        """
        Takes a hidden state plus an action (converted to one-hot),
        returns the next hidden state, reward, value, and policy.
        """
        # Convert the single integer action to a one-hot.
        # For simplicity, we assume batch size of 1 or hidden_state is [1, hidden_size].
        action_tensor = torch.tensor([action], device=hidden_state.device)
        action_one_hot = F.one_hot(action_tensor, num_classes=self.action_space_size).float()

        # Concatenate hidden state + action.
        nn_input = torch.cat([hidden_state, action_one_hot], dim=-1)

        # Next hidden state.
        next_hidden_state = self.dynamics(nn_input)
        # Reward from the same input.
        reward = self.reward_head(next_hidden_state)
        # Then compute value, policy from next hidden state.
        value = self.value_head(next_hidden_state)
        policy = self.policy_head(next_hidden_state)

        return NetworkOutput(value, reward, policy, next_hidden_state)

            
    def get_weights(self):
        # Returns the weights of this network.
        networks = (self.representation, 
                    self.value, 
                    self.policy,
                    self.dynamics, 
                    self.reward)
        
        return [variables
                for variables_list in map(lambda n: n.weights, networks)
                for variables in variables_list] 


    def training_steps(self) -> int:
        # How many steps/batches the network has been trained for.
        return self.tot_training_steps
