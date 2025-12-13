import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from copy import deepcopy


class four_in_a_row:
    """Four in a Row environment class"""
    def __init__(self):
        self.init_game()

    def init_game(self):
        """Initialize the game"""
        # Game parameters
        self.name = 'four_in_a_row'
        self.win_length = 4  # number of pieces in a row needed to win
        self.win_reward = 1  # reward for winning the game
        self.lose_reward = -1  # reward for losing the game
        self.draw_reward = 0  # reward for a draw

        # Colors for players, there are two players
        self.player1_color = 0
        self.player2_color = 1
        self.not_occupied = .75  # the available grid, meaning empty cell

        # Board rendering
        self.cell_size = 100
        self.rows = 4
        self.cols = 9
        self.center = np.array([1.5, 4])

    # ------- game methods --------
    def get_valid_actions(self, board):
        """
        Outputs all valid actions for the current board state.
        Inputs:
            board (np.ndarray): current board state, with unoccupied cells marked by not_occupied
        Process:
            1. Check for draw (means no empty spaces), if so return empty list.
            2. Check if there is a winner on the board, if so return empty list.
            3. Else, iterate through the board to find all unoccupied cells and return their coordinates as valid actions.
        Outputs:
            actions (list of tuple): A list of tuples representing all valid actions (row, col) on the board.
        """
        if self.check_draw(board):
            return []
        elif self.check_win(board):
            return []
        else:
            l = np.vstack(np.where(board == self.not_occupied)).T
            return [(i[0], i[1]) for i in l]

    def check_draw(self, board):
        """
        draw: all grids are occupied
        """
        return np.all(board != self.not_occupied)

    def check_win(self, board):
        """
        Wining condition checker, checking if there are
            1) 4 consecutive pieces in a row
            2) 4 consecutive pieces in a column
            3) 4 consecutive pieces in a diagonal
        on the board.

        The lru_cache is used to cache the result of the function. This
        can be used to speed up the function.

        Inputs:
            board (np.ndarray): the board of the game
            not_occupied (float): the value of the not occupied grid
        Outputs:
            done (bool): whether the game is win or not finished
        """
        rows, cols = board.shape
        # check horizontal
        for i in range(rows):
            for j in range(cols - 3):
                # for each unoccupied grid
                q1 = board[i, j] != self.not_occupied
                q2 = board[i, j] == board[i, j + 1] == board[i, j + 2] == board[i, j + 3]
                if q1 and q2: return True

        # check vertical
        for i in range(rows - 3):
            for j in range(cols):
                q1 = board[i, j] != self.not_occupied
                q2 = board[i, j] == board[i + 1, j] == board[i + 2, j] == board[i + 3, j]
                if q1 and q2: return True

        # check diagonal1 (top-left to bottom-right)
        for i in range(rows - 3):
            for j in range(cols - 3):
                q1 = board[i, j] != self.not_occupied
                q2 = board[i, j] == board[i + 1, j + 1] == board[i + 2, j + 2] == board[i + 3, j + 3]
                if q1 and q2: return True

        # check diagonal2 (top-right to bottom-left)
        for i in range(rows - 3):
            for j in range(3, cols):
                q1 = board[i, j] != self.not_occupied
                q2 = board[i, j] == board[i + 1, j - 1] == board[i + 2, j - 2] == board[i + 3, j - 3]
                if q1 and q2: return True

        return False

    def check_win_action(self, board, action, player_id=0):
        """
        Check if the last action cause a win.

        Check if there are
            1) 4 consecutive pieces in a row
            2) 4 consecutive pieces in a column
            3) 4 consecutive pieces in a diagonal
        on the board.

        Inputs:
            board (np.ndarray): the board of the game
            action (tuple): the action to take
            not_occupied (float): the value of the not occupied grid
        """
        rows, cols = board.shape
        x, y = action

        # Check horizontal
        count = 0
        for c in range(max(0, y - 3), min(cols, y + 4)):
            if board[x, c] == player_id:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Check vertical
        count = 0
        for r in range(max(0, x - 3), min(rows, x + 4)):
            if board[r, y] == player_id:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Check diagonal1 (top-left to bottom-right)
        count = 0
        for i in range(-3, 4):
            r, c = x + i, y + i
            if 0 <= r < rows and 0 <= c < cols:
                if board[r, c] == player_id:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        # Check diagonal2 (top-right to bottom-left)
        count = 0
        for i in range(-3, 4):
            r, c = x + i, y - i
            if 0 <= r < rows and 0 <= c < cols:
                if board[r, c] == player_id:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        return False

    def determine(self, board):
        """
        Determine the game is over or not
        Inputs:
            board (np.ndarray): the board of the game
        Outputs:
            done (bool): whether the game is over, i.e., draw or win
        """
        if self.check_draw(board):
            return True
        elif self.check_win(board):
            return True
        return False

    # ------- state methods --------
    def transit(self, state, action):
        """
        Define the transition and reward functions (rules of the game)

        1) The piece can be dropped when the gird is not occupied.
        2) after one player drop a piece, it will be the other player's turn.
        3) the game is over when one player has 4 consecutive pieces
            in a row, column, or diagonal.
        4) the reward is 1 for the winner, -1 for the loser,
            and 0 for the draw and unfinised game.

        Inputs:
            state (tuple): (board, curr_player) the current state
                * board: np.ndarray, the board of the game
                * curr_player: int, black-player-0, white-player-1
            action (tuple): (row, col) the action to take
                * row: int, the row to drop the piece
                * col: int, the column to drop the piece

        Outputs:
            tuple: (next_state, reward, done, info)
                * next_state: (board, curr_player), the next state of the game
                * reward: int, 1/-1/0 for the winner/loser/draw and unfinised game
                * done: bool, whether the game is over
                * info: dict, the info of the game
        """
        # the board and the current player
        board, curr_player = state
        x, y = action
        reward, done, info = 0, False, {}

        # check if the action is valid
        valid = board[x, y]==self.not_occupied

        if valid:
            board_next = deepcopy(board)
            # update the board
            board_next[x, y] = curr_player
            # switch the player
            curr_player_next = 1-curr_player  # 0->1, 1->0

            # check if the game is finished
            draw = self.check_draw(board_next)
            if not draw:
                done = self.check_win_action(board_next, action, curr_player)
                if done:
                    reward = self.win_reward
                    winner = 'black' if curr_player == 0 else 'white'
                    info = {'winner': winner}
                else:
                    reward = 0
                    info = {}
            else:
                done = True
                info = {'winner': 'draw'}

            next_state = (board_next, curr_player_next)
            return next_state, reward, done, info

        return state, reward, done, info

    def reset(self):
        """
        Reset the environment to initial state.
        Outputs:
            state (tuple): the initial state of the game
                * board: np.ndarray, the board of the game
                    0.75 - not occupied, 0 - black piece, 1 - white piece
                * curr_player: int, 1-black player, 2-white player
        """
        self.board = np.ones([self.rows, self.cols])*self.not_occupied
        self.curr_player = self.player1_color
        self.state = (self.board.copy(), self.curr_player)
        return self.state

    def step(self, action):
        """Take a step in the environment"""
        next_state, reward, done, info = self.transit(self.state, action)
        self.state = next_state
        self.board = next_state[0]
        self.curr_player = next_state[1]
        return next_state, reward, done, info

    def render(self):
        """Render the board using matplotlib"""
        fig, ax = plt.subplots(figsize=(self.cols, self.rows))
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)

        for i in range(self.rows):
            for j in range(self.cols):
                val = self.board[i, j]
                if val == self.player1_color:
                    color = 'red'
                elif val == self.player2_color:
                    color = 'yellow'
                else:
                    color = 'white'
                circle = plt.Circle((j, i), 0.4, color=color, ec='black')
                ax.add_patch(circle)

        plt.title(f"Player {self.curr_player + 1}'s turn", fontsize=16)
        plt.show()


if __name__ == '__main__':
    env = four_in_a_row()
    state = env.reset()
    env.render()
