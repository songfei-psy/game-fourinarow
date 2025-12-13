import numpy as np
import random
from copy import deepcopy


class RandomAgent:
    """A simple agent that plays randomly"""
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id

    def select_action(self, board):
        valid_actions = self.env.get_valid_actions(board)
        return random.choice(valid_actions)


class MinimaxAgent:
    def __init__(self, env, player_id=1, max_depth=3):
        """
        Minimax agent with alpha-beta pruning
        :param env: game environment (four_in_a_row)
        :param player_id: 0 (black) or 1 (white)
        :param max_depth: search depth
        """
        self.env = env
        self.player_id = player_id
        self.opponent_id = 1 - player_id
        self.max_depth = max_depth

    def select_action(self, board):
        """
        Select the best action using Minimax with Alpha-Beta pruning
        """
        valid_actions = self.env.get_valid_actions(board)
        if not valid_actions:
            return None

        best_score = float('-inf')
        best_action = None

        for action in valid_actions:
            next_board = deepcopy(board)
            next_board[action[0], action[1]] = self.player_id

            score = self.minimax(next_board, depth=1, maximizing=False,
                                 alpha=float('-inf'), beta=float('inf'))

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def minimax(self, board, depth, maximizing, alpha, beta):
        """
        Recursive Minimax with Alpha-Beta pruning
        """
        if self.env.check_win(board):
            return 1000 if not maximizing else -1000  # previous move wins

        if self.env.check_draw(board) or depth >= self.max_depth:
            return self.evaluate(board)

        valid_actions = self.env.get_valid_actions(board)
        if not valid_actions:
            return 0

        if maximizing:
            max_eval = float('-inf')
            for action in valid_actions:
                next_board = deepcopy(board)
                next_board[action[0], action[1]] = self.player_id
                eval = self.minimax(next_board, depth + 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                next_board = deepcopy(board)
                next_board[action[0], action[1]] = self.opponent_id
                eval = self.minimax(next_board, depth + 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # alpha cut-off
            return min_eval

    def evaluate(self, board):
        """
        启发式评分函数：
        - 对AI有利的棋形加分
        - 对对手有利的棋形扣分
        - 只考虑长度为4的窗口（因为四子连线）
        """
        score = 0
        player = self.player_id
        opponent = self.opponent_id

        def evaluate_window(window):
            count_player = np.sum(window == player)
            count_opponent = np.sum(window == opponent)
            count_empty = np.sum(window == self.env.not_occupied)

            if count_player == 4:
                return 10000  # 已经胜利（不常出现）
            elif count_player == 3 and count_empty == 1:
                return 100
            elif count_player == 2 and count_empty == 2:
                return 10
            elif count_opponent == 3 and count_empty == 1:
                return -80
            elif count_opponent == 2 and count_empty == 2:
                return -5
            else:
                return 0

        rows, cols = board.shape

        # 横向检查
        for r in range(rows):
            for c in range(cols - 3):
                window = board[r, c:c + 4]
                score += evaluate_window(window)

        # 纵向检查
        for c in range(cols):
            for r in range(rows - 3):
                window = board[r:r + 4, c]
                score += evaluate_window(window)

        # 正对角线检查 \
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                score += evaluate_window(window)

        # 反对角线检查 /
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r - i, c + i] for i in range(4)])
                score += evaluate_window(window)

        # 中心控制（中心列更有价值）
        center_col = cols // 2
        center_array = board[:, center_col]
        center_count = np.sum(center_array == player)
        score += center_count * 5

        return score


