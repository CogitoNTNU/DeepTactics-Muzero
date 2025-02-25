
class MuZeroConfig:
    def __init__(
        self,
        observation_space_size: int = 4,  # CartPole defaults: 4
        action_space_size: int = 2,       # CartPole defaults: 2
        hidden_layer_size: int = 32,
        fine_tune: bool = False,
        learning_rate: float = 0.001
    ):
        #"made for cartpole right now"
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.hidden_layer_size = hidden_layer_size
        self.fine_tune = fine_tune
        self.learning_rate = learning_rate

    def finetune(self):
        if self.fine_tune:
            self.learning_rate = 0.0001
            # Add further modifications as needed
