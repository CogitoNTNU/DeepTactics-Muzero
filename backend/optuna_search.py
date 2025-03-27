import optuna
from src.config import Config
from src.utils.train_network import train_network
from src.utils.run_selfplay import run_selfplay
from src.utils.replay_buffer import ReplayBuffer
from src.networks.network import Network
import time

def objective(trial: optuna.Trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.25, log=True)
    n_tree_searches = trial.suggest_int("n_tree_searches", 20, 100)
    batch_size = trial.suggest_int("batch_size", 1, 32)
    hidden_layer_size = trial.suggest_int("hidden_layer", 32, 256)
    
    
    # Create a configuration with the sampled hyperparameters.
    # You can add more parameters as needed.
    config = Config(
        learning_rate=learning_rate,
        n_tree_searches=n_tree_searches,
        batch_size=batch_size,
        hidden_layer_size=hidden_layer_size,
        training_episodes=15  # use a small number for quick evaluation
    )
    
    # Initialize your replay buffer and network.
    replay_buffer = ReplayBuffer(config=config)
    model = Network.load(config)
    
    cumulative_loss = 0.0
    start_time = time.time()
    
    # Run a reduced number of episodes to evaluate the performance.
    for episode in range(config.training_episodes):
        run_selfplay(config, model, replay_buffer)
        loss_tensor = train_network(config, model, replay_buffer, episode)
        loss = loss_tensor.item()  # assuming loss is a tensor
        cumulative_loss += loss
        
        # Report intermediate result to Optuna.
        trial.report(cumulative_loss, episode)
        
        # Optionally prune unpromising trials early.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    elapsed = time.time() - start_time
    print(f"Trial finished in {elapsed:.2f} seconds with cumulative loss: {cumulative_loss}")
    return cumulative_loss  # assuming lower loss is better

if __name__ == "__main__":
    # Create an Optuna study. Adjust 'direction' as needed (minimize or maximize).
    study = optuna.create_study(direction="minimize")
    
    # Run the hyperparameter optimization for a defined number of trials.
    study.optimize(objective, n_trials=50, n_jobs=16)
    
    # Print the best trial results.
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
