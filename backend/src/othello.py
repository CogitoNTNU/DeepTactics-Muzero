"""Othello environments for reinforcement learning."""

import gym
from gym import spaces
from gym.envs.classic_control import rendering
import pyglet
from pyglet import gl
import numpy as np

BLACK_DISK = -1
NO_DISK = 0
WHITE_DISK = 1

IMAGE_SIZE = 96
MAX_INT = (1 << 31)

WINDOW_H = 480
WINDOW_W = 480
BOARDFIELD = 480


class OthelloEnv(gym.Env):
    """Wrapper of OthelloBaseEnv."""

    metadata = {'render.modes': ['np_array', 'human']}

    def __init__(self,
                 white_policy=None,
                 black_policy=None,
                 protagonist=WHITE_DISK,
                 board_size=8,
                 initial_rand_steps=0,
                 seed=0,
                 sudden_death_on_invalid_move=True,
                 render_in_step=False,
                 num_disk_as_reward=False,
                 possible_actions_in_obs=False):

        # Create the inner environment.
        self.board_size = board_size
        self.num_disk_as_reward = num_disk_as_reward
        self.env = OthelloBaseEnv(
            board_size=board_size,
            num_disk_as_reward=self.num_disk_as_reward,
            sudden_death_on_invalid_move=sudden_death_on_invalid_move,
            possible_actions_in_obs=possible_actions_in_obs,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_in_step = render_in_step
        self.initial_rand_steps = initial_rand_steps
        self.rand_seed = seed
        self.rnd = np.random.RandomState(seed=self.rand_seed)
        self.max_rand_steps = 0
        self.rand_step_cnt = 0

        # Initialize policies.
        self.protagonist = protagonist
        if self.protagonist == BLACK_DISK:
            self.opponent = white_policy
        else:
            self.opponent = black_policy

    def seed(self, seed=None):
        if seed is not None:
            self.rand_seed = seed
            self.rnd = np.random.RandomState(seed=self.rand_seed)
            if self.opponent is not None and hasattr(self.opponent, 'seed'):
                self.opponent.seed(self.rand_seed)

    def reset(self):
        obs = self.env.reset()
        self.max_rand_steps = self.rnd.randint(
            low=0, high=self.initial_rand_steps // 2 + 1) * 2
        self.rand_step_cnt = 0
        print('The initial {} steps will be random'.format(self.max_rand_steps))

        # This provides the opponent a chance to get env.possible_moves.
        if hasattr(self.opponent, 'reset'):
            try:
                self.opponent.reset(self)
            except TypeError:
                pass

        if self.env.player_turn == self.protagonist:
            return obs
        else:
            action = self.opponent.get_action(obs)
            obs, _, done, _ = self.env.step(action)
            if done:
                print('done==True in reset(), do it again.')
                return self.reset()
            else:
                return obs

    def step(self, action):
        assert self.env.player_turn == self.protagonist

        if self.rand_step_cnt < self.max_rand_steps:
            ix = self.rnd.randint(0, len(self.possible_moves))
            action = self.possible_moves[ix]
            self.rand_step_cnt += 1

        obs, reward, done, _ = self.env.step(action)  # My move.
        if self.render_in_step:
            self.render()
        if done:
            return obs, reward, done, None

        while not done and self.env.player_turn != self.protagonist:
            if self.rand_step_cnt < self.max_rand_steps:
                ix = self.rnd.randint(0, len(self.possible_moves))
                opponent_move = self.possible_moves[ix]
                self.rand_step_cnt += 1
            else:
                opponent_move = self.opponent.get_action(obs)
            obs, reward, done, _ = self.env.step(opponent_move)
            if self.render_in_step:
                self.render()
        return obs, -reward, done, None

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def close(self):
        self.env.close()

    @property
    def player_turn(self):
        return self.env.player_turn

    @property
    def possible_moves(self):
        return self.env.possible_moves


class OthelloBaseEnv(gym.Env):
    """Othello base environment."""

    metadata = {'render.modes': ['np_array', 'human']}

    def __init__(self,
                 board_size=8,
                 sudden_death_on_invalid_move=True,
                 num_disk_as_reward=False,
                 possible_actions_in_obs=False,
                 mute=False):

        # Initialize members from configs.
        self.board_size = max(4, board_size)
        self.sudden_death_on_invalid_move = sudden_death_on_invalid_move
        self.board_state = self._reset_board()
        self.viewer = None
        self.num_disk_as_reward = num_disk_as_reward
        self.mute = mute  # Log msgs can be misleading when planning with model.
        self.possible_actions_in_obs = possible_actions_in_obs

        # Initialize internal states.
        self.player_turn = BLACK_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = []

        # Initialize action space: one action for each board position.
        self.action_space = spaces.Discrete(self.board_size ** 2)

        # Initialize observation space.
        if self.possible_actions_in_obs:
            self.observation_space = spaces.Box(
                np.zeros([2, ] + [self.board_size] * 2),
                np.ones([2, ] + [self.board_size] * 2))
        else:
            self.observation_space = spaces.Box(
                np.zeros([self.board_size] * 2), np.ones([self.board_size] * 2))

    def _reset_board(self):
        board_state = np.zeros([self.board_size] * 2, dtype=int)
        center_row_ix = center_col_ix = self.board_size // 2
        board_state[center_row_ix - 1][center_col_ix - 1] = WHITE_DISK
        board_state[center_row_ix][center_col_ix] = WHITE_DISK
        board_state[center_row_ix][center_col_ix - 1] = BLACK_DISK
        board_state[center_row_ix - 1][center_col_ix] = BLACK_DISK
        return board_state

    def reset(self):
        self.board_state = self._reset_board()
        self.player_turn = BLACK_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = self.get_possible_actions()
        return self.get_observation()

    def get_num_killed_enemy(self, board, x, y, delta_x, delta_y):
        # We overload WHITE_DISK to be our disk, and BLACK_DISK to be enemies.
        # (x, y) is a valid position if the following pattern exists:
        #    "(x, y), BLACK_DISK, ..., BLACK_DISK, WHITE_DISK"

        next_x = x + delta_x
        next_y = y + delta_y

        # The neighbor must be an enemy.
        if (
                next_x < 0 or
                next_x >= self.board_size or
                next_y < 0 or
                next_y >= self.board_size or
                board[next_x][next_y] != BLACK_DISK
        ):
            return 0

        # Keep scanning in the direction.
        cnt = 0
        while (
                0 <= next_x < self.board_size and
                0 <= next_y < self.board_size and
                board[next_x][next_y] == BLACK_DISK
        ):
            next_x += delta_x
            next_y += delta_y
            cnt += 1

        if (
                next_x < 0 or
                next_x >= self.board_size or
                next_y < 0 or
                next_y >= self.board_size or
                board[next_x][next_y] != WHITE_DISK
        ):
            return 0
        else:
            return cnt

    def get_possible_actions(self, board=None):
        actions = []
        if board is None:
            if self.player_turn == WHITE_DISK:
                board = self.board_state
            else:
                board = -self.board_state

        for row_ix in range(self.board_size):
            for col_ix in range(self.board_size):
                if board[row_ix][col_ix] == NO_DISK:
                    if (
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, -1)
                    ):
                        actions.append(row_ix * self.board_size + col_ix)
        return actions

    def print_board(self, print_valid_moves=True):
        valid_actions = self.get_possible_actions()

        if print_valid_moves:
            board = self.board_state.copy().ravel()
            for p in valid_actions:
                board[p] = 2
            board = board.reshape(*self.board_state.shape)
        else:
            board = self.board_state

        print('Turn: {}'.format(
            'WHITE' if self.player_turn == WHITE_DISK else 'BLACK'))
        print('Valid actions: {}'.format(valid_actions))
        for row in board:
            print(' '.join(map(lambda x: ['B', 'O', 'W', 'V'][x + 1], row)))
        print('-' * 10)

    def get_observation(self):
        if self.player_turn == WHITE_DISK:
            # White turn, we don't negate state since white=1.
            state = self.board_state
        else:
            # Black turn, we negate board state such that black=1.
            state = -self.board_state
        if self.possible_actions_in_obs:
            grid_of_possible_moves = np.zeros(self.board_size ** 2, dtype=bool)
            grid_of_possible_moves[self.possible_moves] = True
            return np.concatenate([np.expand_dims(state, axis=0),
                                   grid_of_possible_moves.reshape(
                                       [1, self.board_size, self.board_size])],
                                  axis=0)
        else:
            return state

    def set_board_state(self, board_state, perspective=WHITE_DISK):
        """Force setting the board state, necessary in model-based RL."""
        if np.ndim(board_state) > 2:
            state = board_state[0]
        else:
            state = board_state
        if perspective == WHITE_DISK:
            self.board_state = np.array(state)
        else:
            self.board_state = -np.array(state)

    def update_board(self, action):
        x = action // self.board_size
        y = action % self.board_size

        if self.player_turn == BLACK_DISK:
            self.board_state = -self.board_state

        for delta_x in [-1, 0, 1]:
            for delta_y in [-1, 0, 1]:
                if not (delta_x == 0 and delta_y == 0):
                    kill_cnt = self.get_num_killed_enemy(
                        self.board_state, x, y, delta_x, delta_y)
                    for i in range(kill_cnt):
                        dx = (i + 1) * delta_x
                        dy = (i + 1) * delta_y
                        self.board_state[x + dx][y + dy] = WHITE_DISK
        self.board_state[x][y] = WHITE_DISK

        if self.player_turn == BLACK_DISK:
            self.board_state = -self.board_state

    def step(self, action):

        # Apply action.
        if self.terminated:
            raise ValueError('Game has terminated!')
        if action not in self.possible_moves:
            invalid_action = True
        else:
            invalid_action = False
        if not invalid_action:
            self.update_board(action)

        # Determine if game should terminate.
        num_vacant_positions = (self.board_state == NO_DISK).sum()
        no_more_vacant_places = num_vacant_positions == 0
        sudden_death = invalid_action and self.sudden_death_on_invalid_move
        done = sudden_death or no_more_vacant_places

        current_player = self.player_turn
        if done:
            # If game has terminated, determine winner.
            self.winner = self.determine_winner(sudden_death=sudden_death)
        else:
            # If game continues, determine who moves next.
            self.set_player_turn(-self.player_turn)
            if len(self.possible_moves) == 0:
                self.set_player_turn(-self.player_turn)
                if len(self.possible_moves) == 0:
                    if not self.mute:
                        print('No possible moves for either party.')
                    self.winner = self.determine_winner()

        reward = 0
        if self.terminated:
            if self.num_disk_as_reward:
                if sudden_death:
                    # Strongly discourage invalid actions.
                    reward = -(self.board_size ** 2)
                else:
                    white_cnt, black_cnt = self.count_disks()
                    if current_player == WHITE_DISK:
                        reward = white_cnt - black_cnt
                        if black_cnt == 0:
                            reward = self.board_size ** 2
                    else:
                        reward = black_cnt - white_cnt
                        if white_cnt == 0:
                            reward = self.board_size ** 2
            else:
                reward = self.winner * current_player
        return self.get_observation(), reward, self.terminated, None

    def set_player_turn(self, turn):
        self.player_turn = turn
        self.possible_moves = self.get_possible_actions()

    def count_disks(self):
        white_cnt = (self.board_state == WHITE_DISK).sum()
        black_cnt = (self.board_state == BLACK_DISK).sum()
        return white_cnt, black_cnt

    def determine_winner(self, sudden_death=False):
        self.terminated = True
        if sudden_death:
            if not self.mute:
                print('sudden death due to rule violation')
            if self.player_turn == WHITE_DISK:
                if not self.mute:
                    print('BLACK wins')
                return BLACK_DISK
            else:
                if not self.mute:
                    print('WHITE wins')
                return WHITE_DISK
        else:
            white_cnt, black_cnt = self.count_disks()
            if not self.mute:
                print('white: {}, black: {}'.format(white_cnt, black_cnt))
            if white_cnt > black_cnt:
                if not self.mute:
                    print('WHITE wins')
                return WHITE_DISK
            elif black_cnt > white_cnt:
                if not self.mute:
                    print('BLACK wins')
                return BLACK_DISK
            else:
                if not self.mute:
                    print('DRAW')
                return NO_DISK

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'np_array':
            self.print_board()
        else:
            self.show_gui_board()

    def show_gui_board(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)

        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        self.draw_board()

        win.flip()
        return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def draw_board(self):
        # Draw the green background.
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-BOARDFIELD, +BOARDFIELD, 0)
        gl.glVertex3f(+BOARDFIELD, +BOARDFIELD, 0)
        gl.glVertex3f(+BOARDFIELD, -BOARDFIELD, 0)
        gl.glVertex3f(-BOARDFIELD, -BOARDFIELD, 0)
        gl.glEnd()

        # Draw the grid.
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0, 0, 0, 1.0)
        grid_width = BOARDFIELD / self.board_size
        for i in range(1, self.board_size):
            offset = grid_width * i
            gl.glVertex2f(offset, 0)
            gl.glVertex2f(offset, BOARDFIELD)
            gl.glVertex2f(0, offset)
            gl.glVertex2f(BOARDFIELD, offset)
        gl.glEnd()

        # Draw disks.
        half_grid_width = grid_width / 2
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board_state[i][j] == WHITE_DISK:
                    gl.glColor4f(1, 1, 1, 1.)
                elif self.board_state[i][j] == BLACK_DISK:
                    gl.glColor4f(0, 0, 0, 1.)
                if self.board_state[i][j] != NO_DISK:
                    disk = make_disk_at(
                        x=i * grid_width + half_grid_width,
                        y=j * grid_width + half_grid_width,
                        radius=half_grid_width - 3)
                    disk.render1()

        # Draw possible positions.
        if self.player_turn == WHITE_DISK:
            gl.glColor4f(1, 1, 1, 1.)
            font_color = (255, 255, 255, 255)
        else:
            gl.glColor4f(0, 0, 0, 1.)
            font_color = (0, 0, 0, 255)
        for p in self.possible_moves:
            i = p // self.board_size
            j = p % self.board_size
            x = i * grid_width + half_grid_width
            y = j * grid_width + half_grid_width
            disk = make_disk_at(
                x=x, y=y, radius=half_grid_width - 3, filled=False)
            disk.render1()
            label = pyglet.text.Label('{}'.format(i * self.board_size + j),
                                      font_name='Times New Roman', font_size=10,
                                      color=font_color,
                                      x=x, y=y, anchor_x='center',
                                      anchor_y='center')
            label.draw()


def make_disk_at(x, y, radius, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * np.pi * i / res
        points.append((np.cos(ang) * radius + x, np.sin(ang) * radius + y))
    if filled:
        return rendering.FilledPolygon(points)
    else:
        return rendering.PolyLine(points, True)
