import torch
import torch.nn.functional as F
import pytest
from src.networks.network import Network, ResidualBlock, NetworkOutput


# Dummy configuration with necessary attributes
class DummyConfig:
    observation_space_size = 4
    action_space_size = 3
    hidden_layer_size = 8

# Dummy Action factory; assumes Action is initialized with an index

def dummy_action(index=0):
    return index


def test_initial_inference():
    config = DummyConfig()
    net = Network(config)
    # Create a dummy observation (1D tensor with size observation_space_size)
    dummy_obs = torch.rand(config.observation_space_size)

    output = net.initial_inference(dummy_obs)

    # Check that output is an instance of NetworkOutput
    assert isinstance(output, NetworkOutput)
    # value and reward should have shape [1, 1]
    assert output.value.shape == (1, 1)
    assert output.reward.shape == (1, 1)
    # hidden_state should have shape [1, hidden_layer_size]
    assert output.hidden_state.shape == (1, config.hidden_layer_size)
    # policy_logits should have the same length as action_space_size
    assert output.policy_logits.shape[1] == config.action_space_size



def test_recurrent_inference():
    config = DummyConfig()
    net = Network(config)
    dummy_obs = torch.rand(config.observation_space_size)
    init_output = net.initial_inference(dummy_obs)

    # Choose an action, e.g., the first one
    action = dummy_action(0)
    rec_output = net.recurrent_inference(init_output.hidden_state, action)

    assert isinstance(rec_output, NetworkOutput)
    # Check the shapes of outputs
    assert rec_output.value.shape == (1, 1)
    assert rec_output.reward.shape == (1, 1)
    assert rec_output.hidden_state.shape == (1, config.hidden_layer_size)
    print(rec_output.policy_logits)
    assert rec_output.policy_logits.shape[1] == config.action_space_size



def test_residual_block():
    channels = 16
    block = ResidualBlock(channels)
    # Dummy input: batch size 1, 16 channels, 8x8 "image"
    dummy_input = torch.rand(1, channels, 8, 8)
    output = block(dummy_input)

    # Check that output has the same shape as input
    assert output.shape == dummy_input.shape
