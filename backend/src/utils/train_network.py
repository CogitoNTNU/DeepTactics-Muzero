﻿import torch
from src.config import Config
from src.utils.replay_buffer import ReplayBuffer
from src.networks.network import Network
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def calculate_loss(batch_coll: list) -> tuple[torch.Tensor, dict]:
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
        tuple[torch.Tensor, dict]: The averaged loss over the batch and a dictionary of individual losses.
    """
    loss = torch.tensor(0.0, dtype=torch.float32)

    # Initialize total losses for logging/debugging
    tot_value_loss = torch.tensor(0.0, dtype=torch.float32)
    tot_reward_loss = torch.tensor(0.0, dtype=torch.float32)
    tot_policy_loss = torch.tensor(0.0, dtype=torch.float32)

    for zipped_pairs in batch_coll:
        for step_idx, (prediction, target) in enumerate(zipped_pairs):
            value, reward, policy_t = prediction
            target_value, target_reward, target_policy = target

            value_loss = F.mse_loss(value, torch.tensor([[target_value]]))
            # value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)

            if step_idx > 0:
                reward_loss = F.mse_loss(reward, torch.tensor([[target_reward]]))
                # reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum( 1)
            else:
                reward_loss = torch.tensor(0.0, requires_grad=True)

            if target_policy != []:
                # print(f"Policy target: {target_policy}, predicted policy: {policy_t}")
                policy_loss = F.cross_entropy(policy_t, torch.tensor([target_policy]))
                # policy_loss = ( -target_policy * torch.nn.LogSoftmax(dim=1)(policy_t)).sum(1)
            else:
                policy_loss = torch.tensor(0.0, requires_grad=True)

            gradient_scale = len(zipped_pairs)

            value_loss.register_hook(lambda gradient: gradient / gradient_scale)
            reward_loss.register_hook(lambda gradient: gradient / gradient_scale)
            policy_loss.register_hook(lambda gradient: gradient / gradient_scale)

            # 0.25 from reanalize appendix
            # print(loss, policy_loss, reward_loss.squeeze(), value_loss.squeeze())
            loss += policy_loss * 0.25 + reward_loss.squeeze() + value_loss.squeeze()

            # Accumulate losses for logging/debugging
            tot_policy_loss += policy_loss
            tot_reward_loss += reward_loss.squeeze()
            tot_value_loss += value_loss.squeeze()

    # Calculate average losses
    avg_policy_loss = tot_policy_loss / len(batch_coll)
    avg_reward_loss = tot_reward_loss / len(batch_coll)
    avg_value_loss = tot_value_loss / len(batch_coll)
    avg_total_loss = loss / len(batch_coll)

    loss_dict = {
        "total_loss": avg_total_loss.item(),
        "value_loss": avg_value_loss.item(),
        "reward_loss": avg_reward_loss.item(),
        "policy_loss": avg_policy_loss.item(),
    }

    return loss / len(batch_coll), loss_dict


def update_weights(
    optimizer: torch.optim.Optimizer, network: Network, batch: list
) -> torch.Tensor:
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
        tuple[torch.Tensor, dict]: The computed loss for the batch and a dictionary of individual losses.
    """
    optimizer.zero_grad()
    batch_coll = []
    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_t, hidden_state = network.initial_inference(
            observation=image
        )
        predictions = [(value, reward, policy_t)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_t, hidden_state = network.recurrent_inference(
                hidden_state=hidden_state, action=action
            )
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_t))

        batch_coll.append(list(zip(predictions, targets)))
    loss, loss_dict = calculate_loss(batch_coll=batch_coll)
    loss.backward()
    optimizer.step()
    return loss, loss_dict


def train_network(
    config: Config,
    network: Network,
    replay_buffer: ReplayBuffer,
    iterations: int,
    logger,
) -> torch.Tensor:
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
        logger (WandbLogger, optional): Logger for tracking metrics.

    Returns:
        torch.Tensor: The loss computed during the training update.
    """
    print("Starting training")
    network.train()

    for e in range(config.epochs):
        # learning_rate = config.learning_rate * config.lr_decay_rate**(iterations / config.lr_decay_steps)
        if iterations >= config.learning_rate_decay_steps:
            lr = (
                config.learning_rate
                * config.learning_rate_decay**config.learning_rate_decay_steps
            )
            optimizer = optim.SGD(
                network.parameters(),
                lr=lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            lr = config.learning_rate * config.learning_rate_decay ** (
                network.tot_training_steps + 1
            )
            optimizer = optim.SGD(
                network.parameters(),
                lr=lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )

        # Sample batch from replay buffer
        batch = replay_buffer.sample_batch(
            num_unroll_steps=config.num_unroll_steps,
            td_steps=config.td_steps,
            action_space_size=config.action_space_size,
        )

        # Compute loss
        loss, loss_dict = update_weights(
            optimizer=optimizer, network=network, batch=batch
        )

        # print(batch)
        # if e % 5 == 0:
        #    print(f"Loss on epoch: {e}: {loss}. LR: {lr}, tot_training_steps: {network.tot_training_steps}")
        logger.log_losses(loss_dict, network.tot_training_steps)

        # Update training steps counter
        network.tot_training_steps += 1
    network.train(False)
    return loss
