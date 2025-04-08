from typing import Optional
from src.utils.minmaxstats import KnownBounds
from src.gameenv import Environment, TicTacToe, CartPole

def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 100:
        return 1.0
    elif training_steps < 250:
        return 0.5
    else:
        return 0.25

class Config:
    def __init__(
        self,
        render = False,
        visit_softmax_temperature_fn = visit_softmax_temperature,
        known_bounds: Optional[KnownBounds] = None,
        action_space_size: int = 9,  # 9 in tic-tac-toe, 2 legal actions in cartpole
        input_planes: int = 128,  # 3 rbg planes * 32 last states + 32 last actions (othello)
        height: int = 8,  # Pixel height and with (othello)
        width: int = 8, #othello
        # Number of moves that is used as input to representation model
        max_moves: float = 500,  # Max moves before game ends
        game_class: type[Environment] = CartPole,
        n_tree_searches=50,
        training_episodes=100_000, #how many training loops
        epsilon: float = 0.001,
        discount: float = 0.997,
        c1: float = 1.25,
        c2: float = 19652,
        diriclet_noise=0.25,
        # Set this to 0 for deterministic prior probabilites
        dirichlet_exploaration_factor=0.25,
        batch_size=2,
        epochs=25,
        training_interval=100,
        learning_rate: float = 0.0277,
        learning_rate_decay: float = 0.995,
        learning_rate_decay_steps: float = 1000,
        hidden_layer_size: int = 188,
        observation_space_size: int = 4,
        buffer_size = 750, 
        model_load_filename="test2",
        model_save_filename="test",
    ):
        # Environment
        self.action_space_size = action_space_size
        self.max_moves = max_moves
        self.game_name = game_class.__name__
        self.game_class = game_class
        self.input_planes = input_planes
        self.height = height
        self.width = width
        self.render = render

        # Selfplay
        self.n_tree_searches = n_tree_searches

        self.dirichlet_noise_alpha = diriclet_noise
        # e = 0.25 as seen in Alphago Zero paper
        self.dirichlet_exploration_factor = dirichlet_exploaration_factor
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn

        # Training
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_interval = training_interval
        self.model_load_filepath = "models/" + self.game_name + "/" + model_load_filename
        self.model_save_filepath = "models/" + self.game_name + "/" + model_save_filename
        self.training_episodes = training_episodes
        self.td_steps = 50 # ????
        self.num_unroll_steps = 500 # ????
        self.epochs = epochs
        self.buffer_size = buffer_size
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps

        # Network 
        self.observation_space_size = observation_space_size
        self.hidden_layer_size = hidden_layer_size
        
        # PUCT parameters
        self.c1 = c1
        self.c2 = c2
        self.eps = epsilon
        self.discount = discount

        self.known_bounds = known_bounds

def get_cartpole_config() -> Config:
    return Config(
        render = False,
        visit_softmax_temperature_fn = visit_softmax_temperature,
        known_bounds = None,
        action_space_size = 2,  # 9 in tic-tac-toe, 2 legal actions in cartpole
        # Number of moves that is used as input to representation model
        max_moves = 10000,  # Max moves before game ends
        game_class = CartPole,
        n_tree_searches = 25,
        training_episodes = 100_000, #how many training loops
        epsilon = 0.001,
        discount = 0.997,
        c1 = 1.25,
        c2 = 19652,
        diriclet_noise = 0.25,
        # Set this to 0 for deterministic prior probabilites
        dirichlet_exploaration_factor = 0.25,
        batch_size = 64,
        epochs=1,
        training_interval = 100,
        learning_rate = 0.0277,
        learning_rate_decay = 0.995,
        learning_rate_decay_steps = 1000,
        observation_space_size = 4,
        buffer_size = 750, 
        model_load_filename="test2",
        model_save_filename="test",
    )

def get_cartpole_debug_config() -> Config:
    return Config(
        render = False,
        visit_softmax_temperature_fn = visit_softmax_temperature,
        known_bounds = None,
        action_space_size = 2,  # 9 in tic-tac-toe, 2 legal actions in cartpole
        # Number of moves that is used as input to representation model
        max_moves = 10000,  # Max moves before game ends
        game_class = CartPole,
        n_tree_searches = 25,
        training_episodes = 100_000, #how many training loops
        epsilon = 0.001,
        discount = 0.997,
        c1 = 1.25,
        c2 = 19652,
        diriclet_noise = 0.25,
        # Set this to 0 for deterministic prior probabilites
        dirichlet_exploaration_factor = 0.25,
        batch_size = 64,
        epochs=1,
        training_interval = 1,
        learning_rate = 0.0277,
        learning_rate_decay = 0.995,
        learning_rate_decay_steps = 1000,
        observation_space_size = 4,
        buffer_size = 750, 
        model_load_filename="test2",
        model_save_filename="test",
    )

def get_tictactoe_config() -> Config:
    return Config(
        render = False,
        visit_softmax_temperature_fn = visit_softmax_temperature,
        known_bounds = None,
        action_space_size = 9,  # 9 in tic-tac-toe, 2 legal actions in cartpole
        # Number of moves that is used as input to representation model
        max_moves = 10000,  # Max moves before game ends
        game_class = TicTacToe,
        n_tree_searches = 25,
        training_episodes = 100_000, #how many training loops
        hidden_layer_size = 32,
        epsilon = 0.001,
        discount = 1.0,
        c1 = 1.25,
        c2 = 19652,
        diriclet_noise = 0.25,
        # Set this to 0 for deterministic prior probabilites
        dirichlet_exploaration_factor = 0.25,
        batch_size = 64,
        epochs=1,
        training_interval = 100,
        learning_rate = 0.0277,
        learning_rate_decay = 0.995,
        learning_rate_decay_steps = 1000,
        observation_space_size = 4,
        buffer_size = 750, 
        model_load_filename="test2",
        model_save_filename="test",
    )