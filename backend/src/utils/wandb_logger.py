import wandb
import torch
from typing import Dict, Any
from src.networks.network import Network


class WandbLogger:
    def __init__(self, project_name: str, config: Dict[str, Any]):
        """Initialize wandb logger with project name and config."""
        self.run = wandb.init(
            project=project_name, config=config, sync_tensorboard=True
        )

    def watch_model(self, model: Network):
        """Watch model gradients and parameters."""
        wandb.watch(model, log="all", log_freq=1, log_graph=True)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)

    def log_losses(
        self,
        loss_dict: dict,
        step: int,
    ):
        """Log all loss components."""
        self.log_metrics(
            {
                "loss/total": loss_dict["total_loss"],
                "loss/value": loss_dict["value_loss"],
                "loss/reward": loss_dict["reward_loss"],
                "loss/policy": loss_dict["policy_loss"],
            },
            step=step,
        )

    def log_game_stats(self, episode_length: int, step: int):
        """Log game statistics."""
        self.log_metrics(
            {
                # "game/reward": episode_reward,
                "game/length": episode_length
            },
            step=step,
        )

    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        self.log_metrics({"learning_rate": lr}, step=step)

    def finish(self):
        """Finish the wandb run."""
        wandb.finish()

