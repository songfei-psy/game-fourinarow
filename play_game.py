import pygame
import sys
import os
import json
import time
import inspect
import pandas as pd
from datetime import datetime
# Import from my-rep local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.chess import four_in_a_row
import env.agent as agent_module

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
BG_COLOR = (230, 230, 230)
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

# ===== 动态获取可用智能体 =====
def get_available_agents():
    """从 agent 模块中自动发现所有 Agent 类"""
    agents = []
    for name, obj in inspect.getmembers(agent_module, inspect.isclass):
        if name.endswith('Agent') and name != 'basic_agent':
            # 提取简化名称
            simple_name = name.replace('Agent', '').lower()
            if simple_name == 'random':
                agents.append('random')
            elif simple_name == 'heuristic':
                agents.append('heuristic')
            elif simple_name == 'bfs':
                agents.append('bfs')
            elif simple_name == 'openendheuristic':
                agents.append('open_end')
    return sorted(agents)

available_agents = get_available_agents()

# ===== 全局变量 =====
selected_mode = "PVP"  # default mode
player1_moves = []
player2_moves = []
start_time = None
selected_agent_type = available_agents[0] if available_agents else "random"  # default AI for PVE
block_ids = {"PVP": 0, "PVE": 0}
player1_id = "player1"
player2_id = "player2"
ai_label = selected_agent_type
input_text1 = ""
input_text2 = ""
active_input1 = False
active_input2 = False
ids_confirmed = False
moves_started = False
agent_dropdown_open = False


def board2layout(board):
    return [''.join(['.' if x == 0.75 else '0' if x == 0 else '1' for x in row]) for row in board]


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
    filename = os.path.join("data", f"{selected_mode}_{timestamp}.json")
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
            text = f"{player1_id} to move (Black)" if game.curr_player == 0 else f"{player2_id} to move (White)"
        elif selected_mode == "PVE":
            if game.curr_player == 0:
                text = f"{player1_id} to move (Black)"
            elif ai_thinking:
                text = f"{ai_label} is thinking..."
            else:
                text = f"{ai_label} to move (White)"
        else:
            text = ""
        player_text = font.render(text, True, BLACK)
        screen.blit(player_text, (SCREEN_WIDTH // 2 - 150, TOP_BAR_HEIGHT + 5))

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

    # Bottom buttons (right aligned)
    spacing = 20
    button_width = 150
    button_height = 40
    total_width = 2 * button_width + spacing
    start_x = SCREEN_WIDTH - total_width - 20
    restart_rect = pygame.Rect(start_x, SCREEN_HEIGHT - BOTTOM_BAR_HEIGHT + 15, button_width, button_height)
    exit_rect = pygame.Rect(start_x + button_width + spacing, SCREEN_HEIGHT - BOTTOM_BAR_HEIGHT + 15, button_width, button_height)

    pygame.draw.rect(screen, DARK_BLUE, restart_rect, border_radius=8)
    pygame.draw.rect(screen, DARK_BLUE, exit_rect, border_radius=8)

    screen.blit(font.render("Restart", True, WHITE), (restart_rect.x + 35, restart_rect.y + 8))
    screen.blit(font.render("Exit", True, WHITE), (exit_rect.x + 50, exit_rect.y + 8))

    return pvp_rect, pve_rect, restart_rect, exit_rect


def draw_id_inputs(allow_edit):
    # Input panel on bottom-left side (kept compact to avoid covering board)
    panel_width = 320
    panel_height = 60
    panel_rect = pygame.Rect(10, SCREEN_HEIGHT - BOTTOM_BAR_HEIGHT + 5, panel_width, panel_height)
    pygame.draw.rect(screen, SEMI_GRAY, panel_rect, border_radius=8)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect, 2, border_radius=8)

    label1 = font.render("Player 1 ID:", True, BLACK)
    screen.blit(label1, (panel_rect.x + 10, panel_rect.y + 8))
    input_width = panel_width // 2 - 30
    input1_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 30, input_width, 24)
    fill1 = WHITE if (active_input1 and allow_edit) else (GRAY if allow_edit else DARK_GRAY)
    pygame.draw.rect(screen, fill1, input1_rect, border_radius=4)
    txt1 = font.render(input_text1 if input_text1 else player1_id, True, BLACK)
    screen.blit(txt1, (input1_rect.x + 5, input1_rect.y + 5))

    # PVE模式：显示AI智能体选择器；PVP模式：显示Player 2 ID输入框
    input2_rect = None
    agent_selector_rect = None
    agent_dropdown_items = []
    
    if selected_mode == "PVP":
        # Player 2 ID 输入框
        label2 = font.render("Player 2 ID:", True, BLACK)
        screen.blit(label2, (panel_rect.x + panel_width / 2 + 5, panel_rect.y + 8))
        input2_rect = pygame.Rect(panel_rect.x + panel_width / 2 + 5, panel_rect.y + 30, input_width, 24)
        fill2 = WHITE if (active_input2 and allow_edit) else (GRAY if allow_edit else DARK_GRAY)
        pygame.draw.rect(screen, fill2, input2_rect, border_radius=4)
        shown_text2 = input_text2 if input_text2 else player2_id
        txt2 = font.render(shown_text2, True, BLACK)
        screen.blit(txt2, (input2_rect.x + 5, input2_rect.y + 5))
    else:
        # AI智能体选择器
        label2 = font.render("AI Agent:", True, BLACK)
        screen.blit(label2, (panel_rect.x + panel_width / 2 + 5, panel_rect.y + 8))
        agent_selector_rect = pygame.Rect(panel_rect.x + panel_width / 2 + 5, panel_rect.y + 30, input_width, 24)
        
        # 选择器背景（锁定时变浅灰）
        selector_fill = WHITE if allow_edit else DARK_GRAY
        pygame.draw.rect(screen, selector_fill, agent_selector_rect, border_radius=4)
        pygame.draw.rect(screen, DARK_GRAY, agent_selector_rect, 1, border_radius=4)
        
        # 显示当前选择的智能体
        agent_text = font.render(selected_agent_type, True, BLACK)
        screen.blit(agent_text, (agent_selector_rect.x + 5, agent_selector_rect.y + 5))
        
        # 下拉箭头
        arrow_text = font.render("▼" if not agent_dropdown_open else "▲", True, BLACK)  
        screen.blit(arrow_text, (agent_selector_rect.right - 20, agent_selector_rect.y + 5))
        
        # 下拉菜单（向上展开以避免遮挡棋盘）
        if agent_dropdown_open and allow_edit:
            dropdown_height = len(available_agents) * 25
            for i, agent in enumerate(available_agents):
                # 向上展开
                item_y = agent_selector_rect.y - dropdown_height + i * 25
                item_rect = pygame.Rect(agent_selector_rect.x, item_y, input_width, 25)
                color = SEMI_GRAY if agent == selected_agent_type else WHITE
                pygame.draw.rect(screen, color, item_rect, border_radius=3)
                pygame.draw.rect(screen, DARK_GRAY, item_rect, 1, border_radius=3)
                item_text = font.render(agent, True, BLACK)
                screen.blit(item_text, (item_rect.x + 5, item_rect.y + 3))
                agent_dropdown_items.append((agent, item_rect))

    # Confirm button
    confirm_width = 70
    confirm_height = 30
    confirm_rect = pygame.Rect(panel_rect.right + 12, panel_rect.y + (panel_height - confirm_height) // 2, confirm_width, confirm_height)
    pygame.draw.rect(screen, BLUE if allow_edit else DARK_GRAY, confirm_rect, border_radius=6)
    screen.blit(font.render("OK", True, WHITE), (confirm_rect.x + 18, confirm_rect.y + 4))

    return input1_rect, input2_rect, confirm_rect, agent_selector_rect, agent_dropdown_items


def get_cell_from_mouse(pos):
    x, y = pos
    row = (y - TOP_BAR_HEIGHT - INFO_BAR_HEIGHT) // CELL_SIZE
    col = x // CELL_SIZE
    return row, col


def reset_game(mode, agent_type=None):
    global player1_moves, player2_moves, start_time
    player1_moves = []
    player2_moves = []
    start_time = time.perf_counter()
    game = four_in_a_row()
    # Default agent by mode
    if agent_type is None:
        agent_type = selected_agent_type if mode == "PVE" else "random"

    if agent_type == "heuristic":
        agent = agent_module.HeuristicAgent(game)
    elif agent_type == "bfs":
        agent = agent_module.BFSAgent(game)
    elif agent_type == "open_end":
        agent = agent_module.OpenEndHeuristicAgent(game)
    elif agent_type == "random":
        agent = agent_module.RandomAgent(game, player_id=1)
    else:  # error
        agent = None
        Exception("Unknown agent type")

    state = game.reset()
    game_data = {c: [] for c in ['board', 'play_to_move', 'action', 'done', 'winner', 'trial', 'time_elapsed', 'rt', 'player1_id', 'player2_id']}

    def make_block(pid):
        return {
            'player_id': pid,
            'trial': [],
            'board': [],
            'board_idx': [],
            'action': [],
            'action_idx': [],
            'time_elapsed': [],
            'rt': [],
            'env_player_id': []
        }

    block_data = {
        'player1': make_block(player1_id),
        'player2': make_block(player2_id if mode == "PVP" else ai_label)
    }

    block_start_time = time.perf_counter()
    state_present_time = block_start_time
    trial = 0
    return game, agent, state, False, None, game_data, block_data, block_start_time, state_present_time, trial


def main():
    global selected_mode, player1_id, player2_id, ai_label, input_text1, input_text2, active_input1, active_input2, ids_confirmed, moves_started, selected_agent_type, agent_dropdown_open
    ai_label = selected_agent_type
    clock = pygame.time.Clock()
    game, agent, state, game_over, winner_text, game_data, block_data, block_start_time, state_present_time, trial = reset_game(selected_mode, agent_type=selected_agent_type)
    moves_started = False

    while True:
        clock.tick(30)

        allow_edit = (not moves_started) or game_over

        # 主循环中 AI 自动落子部分
        if not game_over and selected_mode == "PVE" and game.curr_player == 1:
            valid_actions = game.get_valid_actions(game.board)
            if valid_actions:
                draw_board(game, winner_text if game_over else None, ai_thinking=True)
                draw_buttons()
                draw_id_inputs(allow_edit)  # keep input panel rendered to avoid flicker during AI thinking
                pygame.display.flip()

                pygame.time.wait(300)  # 可调节延迟
                # Support both RandomAgent.select_action(board) and planning agents' get_action(state)
                if hasattr(agent, "get_action"):
                    board_before = game.board.copy()
                    player_to_move = game.curr_player
                    action = agent.get_action((board_before, player_to_move))
                else:
                    board_before = game.board.copy()
                    player_to_move = game.curr_player
                    action = agent.select_action(board_before)

                response_time = time.perf_counter()
                log_move("player2", list(action))
                moves_started = True
                state, reward, done, info = game.step(action)

                # collect data
                winner = 'None' if len(info) == 0 else info.get('winner', 'None')
                game_data['board'].append(board2layout(board_before))
                game_data['play_to_move'].append(player_to_move)
                game_data['action'].append(str(action))
                game_data['done'].append(done)
                game_data['winner'].append(winner)
                game_data['trial'].append(trial)
                elapsed_seconds = response_time - block_start_time
                game_data['time_elapsed'].append(elapsed_seconds)
                rt = response_time - state_present_time
                game_data['rt'].append(rt)
                game_data['player1_id'].append(player1_id)
                game_data['player2_id'].append(ai_label)

                board_layout_str = ''.join(board2layout(board_before))
                block_data['player2']['trial'].append(trial)
                block_data['player2']['board'].append(board_layout_str)
                block_data['player2']['board_idx'].append(four_in_a_row.board2idx(board_before))
                block_data['player2']['action'].append(str(action))
                block_data['player2']['action_idx'].append(four_in_a_row.action2idx(action))
                block_data['player2']['time_elapsed'].append(elapsed_seconds)
                block_data['player2']['rt'].append(rt)
                block_data['player2']['env_player_id'].append(player_to_move)
                state_present_time = response_time
                trial += 1

                if done:
                    game_over = True
                    winner = info.get("winner", "draw").lower()
                    result = "player1" if winner == "black" else "player2" if winner == "white" else "draw"
                    winner_text = "Player 1 wins!" if winner == "black" else "Player 2 wins!" if winner == "white" else "It's a draw!"
                    mode_lower = selected_mode.lower()
                    mode_dir = os.path.join("data", mode_lower)
                    os.makedirs(mode_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = os.path.join(mode_dir, f"{mode_lower}-{timestamp}.csv")
                    pd.DataFrame(game_data).to_csv(fname, index=False)
                    block_fname = os.path.join(mode_dir, f"{mode_lower}-blocks-{timestamp}.json")
                    with open(block_fname, "w") as f:
                        json.dump({"mode": selected_mode, "player1": block_data['player1'], "player2": block_data['player2']}, f, indent=2)
                    block_ids['PVE'] += 1

        draw_board(game, winner_text if game_over else None)
        pvp_btn, pve_btn, restart_btn, exit_btn = draw_buttons()
        allow_edit = (not moves_started) or game_over
        input1_rect, input2_rect, confirm_rect, agent_selector_rect, agent_dropdown_items = draw_id_inputs(allow_edit)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if not allow_edit:
                    continue
                if active_input1:
                    if event.key == pygame.K_BACKSPACE:
                        input_text1 = input_text1[:-1]
                    else:
                        if event.unicode and event.unicode.isprintable():
                            input_text1 += event.unicode
                elif selected_mode == "PVP" and active_input2:
                    if event.key == pygame.K_BACKSPACE:
                        input_text2 = input_text2[:-1]
                    else:
                        if event.unicode and event.unicode.isprintable():
                            input_text2 += event.unicode

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                # Agent selector handling (PVE mode, in ID input area)
                if selected_mode == "PVE" and agent_selector_rect and allow_edit:
                    if agent_selector_rect.collidepoint(pos):
                        agent_dropdown_open = not agent_dropdown_open
                        continue
                    if agent_dropdown_open:
                        clicked_item = False
                        for agent_name, item_rect in agent_dropdown_items:
                            if item_rect.collidepoint(pos):
                                selected_agent_type = agent_name
                                ai_label = selected_agent_type
                                agent_dropdown_open = False
                                clicked_item = True
                                moves_started = False
                                ids_confirmed = False
                                # Reset game with new agent
                                game, agent, state, game_over, winner_text, game_data, block_data, block_start_time, state_present_time, trial = reset_game(selected_mode, agent_type=selected_agent_type)
                                break
                        if clicked_item:
                            continue
                        # Close dropdown if clicking elsewhere
                        agent_dropdown_open = False

                # Input focus handling
                if allow_edit and input1_rect.collidepoint(pos):
                    active_input1, active_input2 = True, False
                    continue
                if allow_edit and selected_mode == "PVP" and input2_rect and input2_rect.collidepoint(pos):
                    active_input1, active_input2 = False, True
                    continue

                # Confirm button: set IDs based on current inputs and mode
                if allow_edit and confirm_rect.collidepoint(pos):
                    if selected_mode == "PVP":
                        player1_id = input_text1.strip() or player1_id
                        player2_id = input_text2.strip() or player2_id
                    else:  # PVE keeps AI label fixed
                        player1_id = input_text1.strip() or player1_id
                        player2_id = ai_label
                    block_data['player1']['player_id'] = player1_id
                    block_data['player2']['player_id'] = player2_id
                    ids_confirmed = True
                    active_input1 = active_input2 = False
                    continue

                if pvp_btn.collidepoint(pos):
                    selected_mode = "PVP"
                    ids_confirmed = False
                    moves_started = False
                    input_text1, input_text2 = "", ""
                    game, agent, state, game_over, winner_text, game_data, block_data, block_start_time, state_present_time, trial = reset_game(selected_mode, agent_type="random")
                    continue

                if pve_btn.collidepoint(pos):
                    selected_mode = "PVE"
                    ai_label = selected_agent_type
                    ids_confirmed = False
                    moves_started = False
                    input_text1, input_text2 = "", ""
                    game, agent, state, game_over, winner_text, game_data, block_data, block_start_time, state_present_time, trial = reset_game(selected_mode, agent_type=selected_agent_type)
                    continue

                if restart_btn.collidepoint(pos):
                    # Restart with appropriate agent for current mode
                    agent_type = selected_agent_type if selected_mode == "PVE" else "random"
                    ai_label = selected_agent_type if selected_mode == "PVE" else ai_label
                    ids_confirmed = False
                    moves_started = False
                    input_text1, input_text2 = "", ""
                    game, agent, state, game_over, winner_text, game_data, block_data, block_start_time, state_present_time, trial = reset_game(selected_mode, agent_type=agent_type)
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

                # if IDs not confirmed, set defaults based on inputs before first move
                if not ids_confirmed:
                    if selected_mode == "PVP":
                        player1_id = input_text1.strip() or player1_id
                        player2_id = input_text2.strip() or player2_id
                    else:
                        player1_id = input_text1.strip() or player1_id
                        player2_id = ai_label
                    block_data['player1']['player_id'] = player1_id
                    block_data['player2']['player_id'] = player2_id
                    ids_confirmed = True

                board_before = game.board.copy()
                player_to_move = game.curr_player
                player = "player1" if player_to_move == 0 else "player2"

                log_move(player, [row, col])
                moves_started = True
                active_input1 = active_input2 = False
                response_time = time.perf_counter()
                state, reward, done, info = game.step((row, col))

                # collect data
                winner = 'None' if len(info) == 0 else info.get('winner', 'None')
                game_data['board'].append(board2layout(board_before))
                game_data['play_to_move'].append(player_to_move)
                game_data['action'].append(str((row, col)))
                game_data['done'].append(done)
                game_data['winner'].append(winner)
                game_data['trial'].append(trial)
                elapsed_seconds = response_time - block_start_time
                game_data['time_elapsed'].append(elapsed_seconds)
                rt = response_time - state_present_time
                game_data['rt'].append(rt)
                game_data['player1_id'].append(player1_id)
                game_data['player2_id'].append(player2_id if selected_mode == "PVP" else ai_label)

                board_layout_str = ''.join(board2layout(board_before))
                actor_key = 'player1' if player == 'player1' else 'player2'
                block_data[actor_key]['trial'].append(trial)
                block_data[actor_key]['board'].append(board_layout_str)
                block_data[actor_key]['board_idx'].append(four_in_a_row.board2idx(board_before))
                block_data[actor_key]['action'].append(str((row, col)))
                block_data[actor_key]['action_idx'].append(four_in_a_row.action2idx((row, col)))
                block_data[actor_key]['time_elapsed'].append(elapsed_seconds)
                block_data[actor_key]['rt'].append(rt)
                block_data[actor_key]['env_player_id'].append(player_to_move)
                state_present_time = response_time
                trial += 1

                if done:
                    game_over = True
                    winner = info.get("winner", "draw").lower()
                    winner_text = "Player 1 wins!" if winner == "black" else "Player 2 wins!" if winner == "white" else "It's a draw!"
                    mode_lower = selected_mode.lower()
                    mode_dir = os.path.join("data", mode_lower)
                    os.makedirs(mode_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = os.path.join(mode_dir, f"{mode_lower}-{timestamp}.csv")
                    pd.DataFrame(game_data).to_csv(fname, index=False)
                    block_fname = os.path.join(mode_dir, f"{mode_lower}-blocks-{timestamp}.json")
                    with open(block_fname, "w") as f:
                        json.dump({"mode": selected_mode, "player1": block_data['player1'], "player2": block_data['player2']}, f, indent=2)
                    block_ids['PVP'] += 1


if __name__ == "__main__":
    main()
