from src.config import Config
from src.game.action import Action
from src.utils.replay_buffer import ReplayBuffer
from src.utils.shared_storage import SharedStorage
from src.networks.network import Network
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TorchLoss(nn.Module):
    def __init__(self):
        super(TorchLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, batch):
        loss = 0
        print(len(batch))
        print(batch[0])
        for predictions, targets in batch:
            for k, (prediction, target) in enumerate(zip(predictions, targets)):
        
                gradient_scale, value, reward, policy_t = prediction
                target_value, target_reward, target_policy = target

                l_a = self.mse(value, [target_value])
            
                if k > 0:
                    l_b = self.mse(reward, [target_reward])
                else:
                    l_b = 0
            
                l_c = self.cross_entropy(policy_t, target_policy)
                
                loss += l_a + l_b + l_c       
                # loss += scale_gradient(l, gradient_scale)                   
        return loss / len(batch)


def update_weights(optimizer, network, batch, lossfunc):
    optimizer.zero_grad()
    batch_coll = []
    for image, actions, targets in batch:
    
        # Initial step, from the real observation.
        value, reward, policy_t, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_t)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
    
            value, reward, policy_t, hidden_state = network.recurrent_inference(hidden_state, Action(action))
            predictions.append((1.0 / len(actions), value, reward, policy_t))
        batch_coll.append(zip(predictions, targets))
    loss = lossfunc(batch)
    loss.backward()
    optimizer.step()



#
# def update_weights(optimizer, network: Network, batch, weight_decay: float):
#     loss = 0
#     for image, actions, targets in batch:
#
#         # Initial step, from the real observation.
#         value, reward, _, policy_t, hidden_state = network.initial_inference(image)
#         predictions = [(1.0, value, reward, policy_t)]
#
#         # Recurrent steps, from action and previous hidden state.
#         for action in actions:
#
#             value, reward, _, policy_t, hidden_state = network.recurrent_inference(hidden_state, Action(action))
#             predictions.append((1.0 / len(actions), value, reward, policy_t))
#
#             hidden_state = scale_gradient(hidden_state, 0.5)
#
#         for k, (prediction, target) in enumerate(zip(predictions, targets)):
#
#             gradient_scale, value, reward, policy_t = prediction
#             target_value, target_reward, target_policy = target
#
#             l_a = scalar_loss(value, [target_value])
#
#             if k > 0:
#                 l_b = tf.dtypes.cast(scalar_loss(reward, [target_reward]), tf.float32)
#             else:
#                 l_b = 0
#
#             if target_policy == []:
#                 l_c = 0
#             else:
#                 #l_c = tf.nn.softmax_cross_entropy_with_logits(logits=policy_t, labels=target_policy)
#                 cce = keras.losses.CategoricalCrossentropy()
#                 l_c = cce([target_policy], policy_t)
#
#             l =  l_a + l_b + l_c       
#
#             loss += scale_gradient(l, gradient_scale)                   
#
#         loss /= len(batch)
#
#     return loss

def train_network(config: Config, storage: SharedStorage, replay_buffer: ReplayBuffer, iterations: int):
    lossfunc = TorchLoss()
    network = storage.latest_network()  # Ensure this is a PyTorch model
    network.train()  # Set the model to training mode
    network = storage.latest_network()
    # learning_rate = config.learning_rate * config.lr_decay_rate**(iterations / config.lr_decay_steps)
    learning_rate = config.learning_rate
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    # Initialize optimizer
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    # Sample batch from replay buffer
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)

    # Compute loss
    loss = update_weights(optimizer, network, batch, lossfunc=lossfunc)

    # Update training steps counter
    network.tot_training_steps += 1
    
    return loss

# def scalar_loss(prediction, target) -> float:
#     # MSE in board games, cross entropy between categorical values in Atari.
#     return F.mse_loss(prediction, target)
#
# def scale_gradient(tensor, scale: float):
#     # Scales the gradient for the backward pass.
#     return tensor * scale + tensor.detach() * (1 - scale)