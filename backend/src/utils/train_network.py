from src.config import Config
from src.game.action import Action
from src.utils.replay_buffer import ReplayBuffer
from src.utils.shared_storage import SharedStorage
from src.networks.network import Network
import torch as torch
import torch.optim as optim
import torch.nn.functional as F

def update_weights(optimizer, network: Network, batch, weight_decay: float):
    
    with tf.GradientTape() as tape:
    
        loss = 0
    
        for image, actions, targets in batch:
        
            # Initial step, from the real observation.
            value, reward, _, policy_t, hidden_state = network.initial_inference(image)
            predictions = [(1.0, value, reward, policy_t)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
        
                value, reward, _, policy_t, hidden_state = network.recurrent_inference(hidden_state, Action(action))
                predictions.append((1.0 / len(actions), value, reward, policy_t))

                hidden_state = scale_gradient(hidden_state, 0.5)

            for k, (prediction, target) in enumerate(zip(predictions, targets)):
        
                gradient_scale, value, reward, policy_t = prediction
                target_value, target_reward, target_policy = target

                l_a = scalar_loss(value, [target_value])
            
                if k > 0:
                    l_b = tf.dtypes.cast(scalar_loss(reward, [target_reward]), tf.float32)
                else:
                    l_b = 0
            
                if target_policy == []:
                    l_c = 0
                else:
                    #l_c = tf.nn.softmax_cross_entropy_with_logits(logits=policy_t, labels=target_policy)
                    cce = keras.losses.CategoricalCrossentropy()
                    l_c = cce([target_policy], policy_t)
                
                l =  l_a + l_b + l_c       
            
                loss += scale_gradient(l, gradient_scale)                   
    
        loss /= len(batch)
    
        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)
             
    #optimizer.minimize(loss) # this is old Tensorflow API, we use GradientTape
    
    gradients = tape.gradient(loss, [network.representation.trainable_variables,
                                     network.dynamics.trainable_variables,
                                     network.policy.trainable_variables,
                                     network.value.trainable_variables,
                                     network.reward.trainable_variables])
    
    optimizer.apply_gradients(zip(gradients[0], network.representation.trainable_variables))
    optimizer.apply_gradients(zip(gradients[1], network.dynamics.trainable_variables))
    optimizer.apply_gradients(zip(gradients[2], network.policy.trainable_variables))
    optimizer.apply_gradients(zip(gradients[3], network.value.trainable_variables))
    optimizer.apply_gradients(zip(gradients[4], network.reward.trainable_variables))

    return loss

def train_network(config: Config, storage: SharedStorage, replay_buffer: ReplayBuffer, iterations: int):
    
    network = storage.latest_network()
    # learning_rate = config.learning_rate * config.lr_decay_rate**(iterations / config.lr_decay_steps)
    learning_rate = config.learning_rate
    optimizer = tf.keras.optimizers.SGD(learning_rate, config.momentum)

    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.action_space_size)
    loss = update_weights(optimizer, network, batch, config.weight_decay)
    
    network.tot_training_steps += 1
    
    return loss

def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return F.mse_loss(prediction, target)

def scale_gradient(tensor, scale: float):
    # Scales the gradient for the backward pass.
    return tensor * scale + tensor.detach() * (1 - scale)
