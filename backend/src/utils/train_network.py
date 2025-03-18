import torch
from src.config import Config
from src.utils.replay_buffer import ReplayBuffer
from src.utils.shared_storage import SharedStorage
from src.networks.network import Network
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def calculate_loss(batch_coll):
    loss = 0

    for zipped_pairs in batch_coll:
        for step_idx, (prediction, target) in enumerate(zipped_pairs):
    
            gradient_scale, value, reward, policy_t = prediction
            target_value, target_reward, target_policy = target

            l_a = F.mse_loss(value, torch.tensor([target_value]))
        
            if step_idx > 0:
                l_b = F.mse_loss(reward, torch.tensor([target_reward]))
            else:
                l_b = 0

            l_c = F.cross_entropy(policy_t.values(), target_policy)
            
            loss += l_a + l_b + l_c       
            # loss += scale_gradient(l, gradient_scale)                   
    return loss / len(batch_coll)


def update_weights(optimizer, network: Network, batch):
    optimizer.zero_grad()
    batch_coll = []
    for image, actions, targets in batch:
    
        # Initial step, from the real observation.
        value, reward, policy_t, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_t)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
    
            value, reward, policy_t, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_t))
        batch_coll.append(list(zip(predictions, targets)))
    loss = calculate_loss(batch_coll)
    loss.backward()
    optimizer.step()


def train_network(config: Config, storage: SharedStorage, replay_buffer: ReplayBuffer, iterations: int):
    network = storage.latest_network()
    network.train()

    # learning_rate = config.learning_rate * config.lr_decay_rate**(iterations / config.lr_decay_steps)
    optimizer = optim.SGD(network.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    # Sample batch from replay buffer
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)

    # Compute loss
    loss = update_weights(optimizer, network, batch)

    # Update training steps counter
    network.tot_training_steps += 1

    network.train(False)
    
    return loss

# def scalar_loss(prediction, target) -> float:
#     # MSE in board games, cross entropy between categorical values in Atari.
#     return F.mse_loss(prediction, target)
#
# def scale_gradient(tensor, scale: float):
#     # Scales the gradient for the backward pass.
#     return tensor * scale + tensor.detach() * (1 - scale)