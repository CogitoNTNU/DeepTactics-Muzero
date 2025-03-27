import numpy as np
from typing import Tuple


class tic_tac_toe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.player = 1
        self.game_over = False

    def step(self, action: int) -> Tuple[float, bool]:
        """
        returns a tuple of (reward, game_over)
        """
        row = action // 3
        col = action % 3
        if self.board[row][col] != 0:
            raise Exception("Sorry, no numbers below zero")
        self.board[row][col] = self.player
        if self.check_win():
            self.game_over = True
            return 1.0, self.game_over
        if self.check_tie():
            self.game_over = True
            return 0.0, self.game_over
        self.player *= -1
        return 0.0, self.game_over

    def get_state(self) -> np.ndarray:
        output = np.zeros((3, 3, 3))
        output[self.board[0] == 1] = self.player
        # Assign 'O' where o_board is 1
        output[self.board[1] == 1] = self.player * -1
        return output

    def check_win(self):
        """Check if the player has won the game."""
        # Check rows and columns
        for i in range(3):
            if np.all(self.board[i, :] == self.player) or np.all(
                self.board[:, i] == self.player
            ):
                return True

        # Check diagonals
        if np.all(np.diag(self.board) == self.player) or np.all(
            np.diag(np.fliplr(self.board)) == self.player
        ):
            return True

        return False

    def check_tie(self):
        """Check if the board is full, meaning a tie if no winner is found."""
        return np.all(self.board > 0)  # No empty spaces (0s) left


def play_tui():
    game = tic_tac_toe()

    def render_board():
        symbols = {0: ".", 1: "X", -1: "O"}
        print("\n".join(" ".join(symbols[cell] for cell in row) for row in game.board))

    print("Welcome to Tic-Tac-Toe! Use numbers 0-8 to place your move.")

    while not game.game_over:
        render_board()
        try:
            move = int(
                input(
                    f"Player {'X' if game.player == 1 else 'O'}, enter your move (0-8): "
                )
            )
            if move not in range(9):
                raise ValueError
            _, game_over = game.step(move)
        except Exception as e:
            print(f"Invalid move: {e}")
            continue

    render_board()
    if game.check_win():
        print(f"Player {'X' if game.player == 1 else 'O'} wins!")
    else:
        print("It's a tie!")


if __name__ == "__main__":
    play_tui()
