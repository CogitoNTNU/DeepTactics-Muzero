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
    
            value, reward, policy_t = prediction
            target_value, target_reward, target_policy = target


            value_loss = F.mse_loss(value, torch.tensor([[target_value]]))
        
            if step_idx > 0:
                reward_loss = F.mse_loss(reward, torch.tensor([[target_reward]]))
            else:
                reward_loss = torch.tensor(0.0, requires_grad=True)

            if target_policy != []:
                policy_loss = F.cross_entropy(policy_t.log(), torch.tensor([target_policy]))
            else:
                policy_loss = torch.tensor(0.0, requires_grad=True)
            
            
            gradient_scale = len(zipped_pairs)

            value_loss.register_hook(lambda gradient: gradient / gradient_scale)
            reward_loss.register_hook(lambda gradient: gradient / gradient_scale)
            policy_loss.register_hook(lambda gradient: gradient / gradient_scale)
            """
            print(f"Pred reward: {reward}, actual reward: {target_reward}, Loss: {reward_loss}")
            print(f"Pred value: {value}, actual value: {target_value}, Loss: {value_loss}")
            print(f"Pred policy: {policy_t}, actual policy: {target_policy}, Loss: {policy_loss}\n")
            """
            # 0.25 from reanalize appendix
            loss += (policy_loss * 0.25 + reward_loss + value_loss)
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
        value, reward, policy_t, hidden_state = network.initial_inference(observation=image)
        predictions = [(value, reward, policy_t)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_t, hidden_state = network.recurrent_inference(hidden_state=hidden_state, action=action)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_t))

        batch_coll.append(list(zip(predictions, targets)))
    loss = calculate_loss(batch_coll=batch_coll)
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

    for e in range(config.epochs):
        # learning_rate = config.learning_rate * config.lr_decay_rate**(iterations / config.lr_decay_steps)
        if(iterations>=config.learning_rate_decay_steps):
            lr=config.learning_rate*config.learning_rate_decay**config.learning_rate_decay_steps
            optimizer = optim.SGD(network.parameters(), lr=lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            lr = config.learning_rate*config.learning_rate_decay**(network.tot_training_steps+1)
            optimizer = optim.SGD(network.parameters(), lr=lr, momentum=config.momentum, weight_decay=config.weight_decay)
            
        # Sample batch from replay buffer
        batch = replay_buffer.sample_batch(num_unroll_steps=config.num_unroll_steps, td_steps=config.td_steps, action_space_size=config.action_space_size)

        # Compute loss
        loss = update_weights(optimizer=optimizer, network=network, batch=batch)
        #print(batch)
        #if e % 5 == 0:
        #    print(f"Loss on epoch: {e}: {loss}. LR: {lr}, tot_training_steps: {network.tot_training_steps}")

        # Update training steps counter
        network.tot_training_steps += 1

    network.train(False)
    return loss