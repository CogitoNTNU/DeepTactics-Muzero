import torch
import torch.nn as nn
import torch.nn.functional as F
from config import OBSERVATION_SHAPE, HIDDEN_DIM, ACTION_SIZE, NUM_RESIDUAL_BLOCKS


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class Network(nn.Module):
    def __init__(self, config):
        """
        Expects a config object with:
          - observation_space_size: int
          - action_space_size: int
          - hidden_layer_size: int
        """
        super(Network, self).__init__()
        self.hidden_layer_size = config.hidden_layer_size
        self.action_space_size = config.action_space_size

        # Representation: from raw observation to hidden state.
        self.representation = nn.Sequential(
            nn.Linear(config.observation_space_size, config.hidden_layer_size),
            nn.ReLU(),
            ResidualBlock(config.hidden_layer_size, config.hidden_layer_size)
        )


