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







class PredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, hidden_state):
        policy = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return policy, value



class DynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN_DIM + ACTION_SIZE, HIDDEN_DIM)
        self.reward_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, hidden_state, action):
        action_onehot = F.one_hot(action, num_classes=ACTION_SIZE).float()
        x = torch.cat([hidden_state, action_onehot], dim=-1)
        next_state = F.relu(self.fc(x))
        reward = self.reward_head(next_state)
        return next_state, reward



if __name__ == "__main__":
    obs = torch.randn(1, *OBSERVATION_SHAPE)
    action = torch.tensor([2])

    rep_net = RepresentationNetwork()
    pred_net = PredictionNetwork()
    dyn_net = DynamicsNetwork()

    # Run the networks
    hidden_state = rep_net(obs)
    policy, value = pred_net(hidden_state)
    next_state, reward = dyn_net(hidden_state, action)

    print("Hidden State Shape:", hidden_state.shape)
    print("Policy:", policy)
    print("Value:", value.item())
    print("Next State Shape:", next_state.shape)
    print("Reward:", reward.item())
