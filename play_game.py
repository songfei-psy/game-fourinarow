import pygame
import sys
import os
import json
import time
from datetime import datetime
from env.chess import four_in_a_row
from env.agent import *

# ===== 配置参数 =====
CELL_SIZE = 100
TOP_BAR_HEIGHT = 70
INFO_BAR_HEIGHT = 40
BOTTOM_BAR_HEIGHT = 70
SCREEN_WIDTH = 9 * CELL_SIZE
SCREEN_HEIGHT = TOP_BAR_HEIGHT + INFO_BAR_HEIGHT + 4 * CELL_SIZE + BOTTOM_BAR_HEIGHT

# ===== 颜色定义 =====
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG_COLOR = (245, 245, 245)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (50, 100, 200)
DARK_BLUE = (30, 80, 160)
SEMI_GRAY = (220, 220, 220)

# ===== 初始化 Pygame =====
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Four in a Row")
font = pygame.font.SysFont(None, 30)
big_font = pygame.font.SysFont(None, 56)

# ===== 全局变量 =====
selected_mode = "PVP"  # default mode
player1_moves = []
player2_moves = []
start_time = None


def current_ms():
    return int((time.perf_counter() - start_time) * 1000)


def log_move(player_id, coord):
    clean_coord = [int(coord[0]), int(coord[1])]
    move = {"coord": clean_coord, "time": current_ms()}
    if player_id == "player1":
        player1_moves.append(move)
    else:
        player2_moves.append(move)


def save_data(result):
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/{selected_mode}_{timestamp}.json"
    data = {
        "mode": selected_mode,
        "player1": player1_moves,
        "player2": player2_moves,
        "winner": result
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Game saved to {filename}")


def draw_board(game, winner_text=None, ai_thinking=False):
    screen.fill(BG_COLOR)

    # 玩家提示
    if not winner_text:
        if selected_mode == "PVP":
            text = "Player 1's Turn" if game.curr_player == 0 else "Player 2's Turn"
        elif selected_mode == "PVE":
            if game.curr_player == 0:
                text = "Player's Turn"
            elif ai_thinking:
                text = "AI is thinking..."
            else:
                text = ""
        else:
            text = ""
        player_text = font.render(text, True, BLACK)
        screen.blit(player_text, (SCREEN_WIDTH // 2 - 80, TOP_BAR_HEIGHT + 5))

    # 绘制棋盘
    for i in range(game.rows):
        for j in range(game.cols):
            x = j * CELL_SIZE
            y = TOP_BAR_HEIGHT + INFO_BAR_HEIGHT + i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, width=1)
            val = game.board[i][j]
            if val == game.player1_color:
                pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 2 - 10)
            elif val == game.player2_color:
                pygame.draw.circle(screen, WHITE, rect.center, CELL_SIZE // 2 - 10)
                pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 2 - 10, width=2)

    # 结局提示
    if winner_text:
        box_rect = pygame.Rect((SCREEN_WIDTH - 400) // 2, SCREEN_HEIGHT // 2 - 50, 400, 100)
        pygame.draw.rect(screen, SEMI_GRAY, box_rect, border_radius=10)
        pygame.draw.rect(screen, BLACK, box_rect, 2, border_radius=10)
        result_text = big_font.render(winner_text, True, DARK_BLUE)
        text_rect = result_text.get_rect(center=box_rect.center)
        screen.blit(result_text, text_rect)


def draw_buttons():
    half_width = SCREEN_WIDTH // 2
    pvp_rect = pygame.Rect(0, 0, half_width, TOP_BAR_HEIGHT)
    pve_rect = pygame.Rect(half_width, 0, half_width, TOP_BAR_HEIGHT)

    pygame.draw.rect(screen, BLUE if selected_mode == "PVP" else GRAY, pvp_rect)
    pygame.draw.rect(screen, BLUE if selected_mode == "PVE" else GRAY, pve_rect)

    screen.blit(font.render("Players vs Players", True, WHITE), (half_width // 2 - 80, 22))
    screen.blit(font.render("Player vs AI", True, WHITE), (half_width + half_width // 2 - 60, 22))

    # Bottom buttons
    spacing = 20
    button_width = 150
    button_height = 40
    total_width = 2 * button_width + spacing
    start_x = (SCREEN_WIDTH - total_width) // 2
    restart_rect = pygame.Rect(start_x, SCREEN_HEIGHT - BOTTOM_BAR_HEIGHT + 15, button_width, button_height)
    exit_rect = pygame.Rect(start_x + button_width + spacing, SCREEN_HEIGHT - BOTTOM_BAR_HEIGHT + 15, button_width, button_height)

    pygame.draw.rect(screen, DARK_BLUE, restart_rect, border_radius=8)
    pygame.draw.rect(screen, DARK_BLUE, exit_rect, border_radius=8)

    screen.blit(font.render("Restart", True, WHITE), (restart_rect.x + 35, restart_rect.y + 8))
    screen.blit(font.render("Exit", True, WHITE), (exit_rect.x + 50, exit_rect.y + 8))

    return pvp_rect, pve_rect, restart_rect, exit_rect


def get_cell_from_mouse(pos):
    x, y = pos
    row = (y - TOP_BAR_HEIGHT - INFO_BAR_HEIGHT) // CELL_SIZE
    col = x // CELL_SIZE
    return row, col


def reset_game(mode, agent_type="random"):
    global player1_moves, player2_moves, start_time
    player1_moves = []
    player2_moves = []
    start_time = time.perf_counter()
    game = four_in_a_row()
    if agent_type == "heuristic":
        params = default_params().to_list()
        agent = HeuristicAgent(game, params=params)
    elif agent_type == "bfs":
        params = default_params().to_list()
        agent = BFSAgent(game, params=params)
    elif agent_type == "random":
        agent = RandomAgent(game, player_id=1)
    else:  # error
        agent = None
        Exception("Unknown agent type")
    state = game.reset()
    return game, agent, state, False, None


def main():
    global selected_mode
    clock = pygame.time.Clock()
    game, agent, state, game_over, winner_text = reset_game(selected_mode, agent_type="bfs")

    while True:
        clock.tick(30)

        # 主循环中 AI 自动落子部分
        if not game_over and selected_mode == "PVE" and game.curr_player == 1:
            valid_actions = game.get_valid_actions(game.board)
            if valid_actions:
                draw_board(game, winner_text if game_over else None, ai_thinking=True)
                draw_buttons()
                pygame.display.flip()

                pygame.time.wait(300)  # 可调节延迟
                action = agent.select_action(game.board)
                log_move("player2", list(action))
                state, reward, done, info = game.step(action)

                if done:
                    game_over = True
                    winner = info.get("winner", "draw").lower()
                    result = "player1" if winner == "black" else "player2" if winner == "white" else "draw"
                    winner_text = "Player 1 wins!" if winner == "black" else "Player 2 wins!" if winner == "white" else "It's a draw!"
                    save_data(result)

        draw_board(game, winner_text if game_over else None)
        pvp_btn, pve_btn, restart_btn, exit_btn = draw_buttons()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                if pvp_btn.collidepoint(pos):
                    selected_mode = "PVP"
                    game, agent, state, game_over, winner_text = reset_game(selected_mode)
                    continue

                if pve_btn.collidepoint(pos):
                    selected_mode = "PVE"
                    game, agent, state, game_over, winner_text = reset_game(selected_mode)
                    continue

                if restart_btn.collidepoint(pos):
                    game, agent, state, game_over, winner_text = reset_game(selected_mode)
                    continue

                if exit_btn.collidepoint(pos):
                    pygame.quit()
                    sys.exit()

                if game_over:
                    continue

                row, col = get_cell_from_mouse(pos)
                if row < 0 or row >= game.rows or col >= game.cols:
                    continue
                if (row, col) not in game.get_valid_actions(game.board):
                    continue

                player = "player1" if game.curr_player == 0 else "player2"
                log_move(player, [row, col])
                state, reward, done, info = game.step((row, col))

                if done:
                    game_over = True
                    winner = info.get("winner", "draw").lower()
                    winner_text = "Player 1 wins!" if winner == "black" else "Player 2 wins!" if winner == "white" else "It's a draw!"
                    save_data("player1" if winner == "black" else "player2" if winner == "white" else "draw")


if __name__ == "__main__":
    main()
