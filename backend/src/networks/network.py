import os
from pytest import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class NetworkOutput(NamedTuple):
    """
    Data structure for the output of initial_inference or recurrent_inference.

    Attributes:
        value (torch.Tensor): Scalar value estimate.
        reward (torch.Tensor): Scalar immediate reward.
        policy_logits (List[float]): List of probabilities for each action.
        hidden_state (torch.Tensor): Hidden state tensor of shape [batch_size, hidden_layer_size].
    """
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: list[float]
    hidden_state: torch.Tensor


class ResidualBlock(nn.Module):
    """
    A residual block module with two convolutional layers and batch normalization.

    Args:
        channels (int): The number of channels for the convolutional layers.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Perform a forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual connection.
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class Network(nn.Module):
    """
    Neural network architecture that performs initial and recurrent inferences.
    It includes representation, value head, policy head, dynamics, and reward head modules.

    Args:
        config: A configuration object with attributes:
            - observation_space_size (int): Size of the observation space.
            - hidden_layer_size (int): Size of the hidden layer.
            - action_space_size (int): Number of possible actions.
    """
    def __init__(self, config: Config) -> None:
        super(Network, self).__init__()
        self.config: Config = config
        self.hidden_layer_size = config.hidden_layer_size
        self.action_space_size = config.action_space_size

        # Representation: from raw observation to hidden state.
        self.representation = nn.Sequential(
            nn.Linear(config.observation_space_size, config.hidden_layer_size),
            nn.ELU(),
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size)
        )

        # Value head: predicts scalar value from hidden state.
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ELU(),
            nn.Linear(config.hidden_layer_size, 1, bias=True)
        )

        # Policy head: predicts action probabilities from hidden state.
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ELU(),
            nn.Linear(config.hidden_layer_size, config.action_space_size),
            nn.Softmax(dim=-1)
        )

        # Dynamics: given [hidden_state, action one-hot] -> next hidden state
        self.dynamics = nn.Sequential(
            nn.Linear(config.hidden_layer_size + config.action_space_size, config.hidden_layer_size),
            nn.ELU(),
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size)
        )

        # Reward head: same input as dynamics, but outputs a scalar reward.
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            nn.ELU(),
            nn.Linear(config.hidden_layer_size, 1, bias=True)
        )

        self.tot_training_steps = 0

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        """
        Performs the initial inference from an environment observation.

        The reward is set to zero because no action has been taken yet.

        Args:
            observation (torch.Tensor): The input observation. Can be a list or tensor.
                If a list is provided, it will be converted to a tensor.
                Expected shape is either 1D or higher; if 1D, it will be unsqueezed to form a batch.

        Returns:
            NetworkOutput: NamedTuple containing value, reward, policy logits, and hidden state.
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
        reward = torch.zeros((observation.shape[0], 1), device=observation.device, dtype=observation.dtype)
        return NetworkOutput(value, reward, policy, hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: int) -> NetworkOutput:
        """
        Performs recurrent inference by taking a hidden state and an action.

        Converts the action to a one-hot tensor, concatenates it with the hidden state,
        and computes the next hidden state, reward, value, and policy.

        Args:
            hidden_state (torch.Tensor): Current hidden state tensor of shape [batch_size, hidden_layer_size].
            action (int): The action taken.

        Returns:
            NetworkOutput: NamedTuple containing value, reward, policy logits, and the next hidden state.
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
        networks = (self.representation, self.value, self.policy, self.dynamics, self.reward)
        return [variables for variables_list in map(lambda n: n.weights, networks) for variables in variables_list] 


    def training_steps(self) -> int:
        # How many steps/batches the network has been trained for.
        return self.tot_training_steps
    
    def save(self) -> None:
        path = self.config.model_save_filepath
        directory = os.path.dirname(path)
        
        # Check if the directory exists
        if not os.path.exists(directory):
            # If it doesn't exist, create it
            os.makedirs(directory)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, config: Config):
        path = config.model_load_filepath

        if not os.path.exists(path):
            print("No models with this name exists")
            return Network(config)
        

        model = cls(config)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        print("Model found and loaded.")
        
        return model