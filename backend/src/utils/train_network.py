import torch
from src.config import Config
from src.utils.replay_buffer import ReplayBuffer
from src.networks.network import Network
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def calculate_loss(batch_coll: list) -> torch.Tensor:
    """
    Calculates the aggregated loss over a batch of predictions and targets.

    The loss is computed as the average of three components for each prediction-target pair:
      - Mean squared error (MSE) loss for the value prediction.
      - MSE loss for the reward prediction (only for recurrent steps).
      - Cross entropy loss for the policy prediction if a target policy is provided.

    Args:
        batch_coll (list): A collection of lists, where each inner list contains tuples of
            (prediction, target) for each step. Each prediction is a tuple of
            (gradient_scale, value, reward, policy_t) and each target is a tuple of
            (target_value, target_reward, target_policy).

    Returns:
        torch.Tensor: The averaged loss over the batch.
    """
    loss = torch.tensor(0.0, dtype=torch.float32)

    for zipped_pairs in batch_coll:
        for step_idx, (prediction, target) in enumerate(zipped_pairs):
    
            gradient_scale, value, reward, policy_t = prediction
            target_value, target_reward, target_policy = target


            l_a = F.mse_loss(value, torch.tensor([[target_value]]))
        
            if step_idx > 0:
                l_b = F.mse_loss(reward, torch.tensor([[target_reward]]))
            else:
                l_b = torch.tensor(0.0)

            if target_policy != []:
                l_c = F.cross_entropy(policy_t, torch.tensor([target_policy]))
            else:
                l_c = torch.tensor(0.0)
            #print(f"Pred reward: {reward}, actual reward: {target_reward}, Loss: {l_b}")
            #print(f"Pred value: {value}, actual value: {target_value}, Loss: {l_a}")
            #print(f"Pred policy: {policy_t}, actual policy: {target_policy}, Loss: {l_c}\n")
            
            #print("L_c:", l_c, "L_b:", l_b, "L_a:", l_a)
            loss += (l_c + l_b + l_a)/len(zipped_pairs)
            
            

    return loss / len(batch_coll)


def update_weights(optimizer: torch.optim.Optimizer, network: Network, batch: list) -> torch.Tensor:
    """
    Performs a weight update on the network using a sampled batch.

    The function computes predictions for the initial step and subsequent recurrent steps,
    aggregates them with corresponding targets, calculates the loss, and performs backpropagation.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer used for updating the network.
        network (Network): The network to be trained.
        batch (list): A batch of training data, where each element is a tuple of 
            (image, actions, targets).

    Returns:
        torch.Tensor: The computed loss for the batch.
    """
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
    return loss


def train_network(config: Config, network: Network, replay_buffer: ReplayBuffer, iterations: int) -> torch.Tensor:
    """
    Trains the network for one iteration using a batch sampled from the replay buffer.

    The function sets the network to training mode, selects an optimizer based on the iteration count,
    samples a batch of trajectories from the replay buffer, computes the loss, updates the network weights,
    increments the training step counter, and then sets the network to evaluation mode.

    Args:
        config (Config): Configuration object containing training parameters.
        network (Network): The network to be trained.
        replay_buffer (ReplayBuffer): The replay buffer from which training batches are sampled.
        iterations (int): The current iteration count used for learning rate decay.

    Returns:
        torch.Tensor: The loss computed during the training update.
    """
    network.train()

    # learning_rate = config.learning_rate * config.lr_decay_rate**(iterations / config.lr_decay_steps)
    if(iterations>=config.learning_rate_decay_steps):
        optimizer = optim.SGD(network.parameters(), lr=config.learning_rate*(iterations+1)*config.learning_rate_decay, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(network.parameters(), lr=config.learning_rate*config.learning_rate_decay_steps*config.learning_rate_decay, momentum=config.momentum, weight_decay=config.weight_decay)
    
    # Sample batch from replay buffer
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)

    # Compute loss
    loss = update_weights(optimizer, network, batch)

    # Update training steps counter
    network.tot_training_steps += 1

    network.train(False)
    return loss