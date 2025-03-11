from typing import Optional
from src.utils.minmaxstats import KnownBounds


def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 100:
        return 3
    elif training_steps < 125:
        return 2
    elif training_steps < 150:
        return 1
    elif training_steps < 175:
        return 0.5
    elif training_steps < 200:
        return 0.250
    elif training_steps < 225:
        return 0.125
    elif training_steps < 250:
        return 0.075
    else:
        return 0.001


class Config:
    def __init__(
        self,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds: Optional[KnownBounds] = None,
        action_space_size: int = 2,  # 2 legal actions in cartpole
        input_planes: int = 128,  # 3 rbg planes * 32 last states + 32 last actions
        height: int = 96,  # Pixel height and with
        width: int = 96,
        # Number of moves that is used as input to representation model
        num_input_moves: int = 32,
        max_moves: float = 50_000,  # Max moves before game ends
        game_name: str = "ALE/Breakout-v5",
        num_selfplay_games=1_000_000,
        max_replay_games=125_000,  # Replay buffer size
        n_tree_searches=50,
        training_episodes=200,
        epsilon: float = 0.001,
        discount: float = 0.997,
        c1: float = 1.25,
        c2: float = 19652,
        diriclet_noise=0.25,
        # Set this to 0 for deterministic prior probabilites
        dirichlet_exploaration_factor=0.25,
        batch_size=2048,
        # encode_game_state_fn = encode_state_atari,
        # softmax_policy_fn = softmax_policy_atari_train,
        info_print_rate=10,
        training_interval=1_000,
        learning_rate: float = 0.001,
        hidden_layer_size: int = 32,
        observation_space_size: int = 4,
        fine_tune: bool = False,
        num_training_rolluts=5,
        model_load_filename="test",
        model_save_filename="test",
    ):
        # Only to keep the type checker happy
        # gym.register_envs(ale_py)

        # Environment
        self.action_space_size = action_space_size
        self.max_moves = max_moves
        self.game_name = game_name
        self.input_planes = input_planes
        self.height = height
        self.width = width
        self.num_input_moves = num_input_moves

        # Selfplay
        self.num_selfplay_games = num_selfplay_games
        self.max_replay_games = max_replay_games
        self.n_tree_searches = n_tree_searches

        self.dirichlet_noise_alpha = diriclet_noise
        # e = 0.25 as seen in Alphago Zero paper
        self.dirichlet_exploration_factor = dirichlet_exploaration_factor
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn

        # Training
        self.learning_rate = learning_rate
        self.lr_decay_steps = 20
        self.lr_decay_rate = 0.1
        self.batch_size = batch_size
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.training_interval = training_interval
        self.model_load_filepath = "models/" + game_name + "/" + model_load_filename
        self.model_save_filepath = "models/" + game_name + "/" + model_save_filename
        self.training_episodes = training_episodes
        self.num_training_rolluts = num_training_rolluts
        self.td_steps = 7 # ????
        self.num_unroll_steps = 500 # ????

        # Network 
        self.observation_space_size = observation_space_size
        self.hidden_layer_size = hidden_layer_size
        self.fine_tune = fine_tune
        
        # PUCT parameters
        self.c1 = c1
        self.c2 = c2
        self.eps = epsilon
        self.discount = discount

        # Functions that you might want to customize for your enviroment
        # self.softmax_policy_fn = softmax_policy_fn
        # self.encode_game_state_fn = encode_game_state_fn

        # Logging
        self.info_print_rate = info_print_rate

        self.known_bounds = known_bounds

    # def init_game(self) -> gym.Env:
    #     pass
    def finetune(self):
        if self.fine_tune:
            self.learning_rate = 0.0001
            # Add further modifications as needed