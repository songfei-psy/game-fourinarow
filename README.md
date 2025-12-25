# Four in a Row (4x9)

Four in a Row (4x9) with Pygame UI and several agents (player vs player, player vs AI). Inspired by
Van Opheusden et al. (2023) "Expertise increases planning depth in human gameplay".

## Project Overview

This project consists of two main components:

1. **Game & Behavioral Data Collection**: A complete Four-in-a-Row game implementation supporting multiple modes including Player vs AI (PVE) and Player vs Player (PVP). The game collects real-time behavioral data, including the position of each move and the time taken to make it.

2. **Intelligent Algorithm & Behavioral Hypothesis Model**: The game's AI algorithm (Agent) is implemented in `env/agent.py`. Crucially, this Agent also serves as a **hypothesis model for human behavior**.

   In practical applications, we only have human player behavioral data and wish to fit a hypothesis model to it. To evaluate the quality of parameter fitting algorithms, we employ a clever validation approach:
   - **Define a "ground truth" hypothesis model** (our Agent implementation) and use it to generate synthetic data.
   - **Apply parameter fitting algorithms to recover the original model parameters** from the synthetic data.
   - **Compare recovered parameters with true parameters** to assess the accuracy and reliability of the fitting algorithm.

## Features
- 4×9 board, connect four to win (horizontally, vertically, or diagonally).
- Game modes: PVP (two players), PVE (player vs AI).
- AI options: `RandomAgent` (random moves), `HeuristicAgent` (heuristic-based), `BFSAgent` (with forward search and pruning).
- Interface: Pygame-based board rendering with buttons for mode switching, restart, and exit.
- Data logging: Real-time recording of moves and timestamps to `data/<mode>/` in both CSV and JSON formats.

## Requirements
- Python 3.8+ (tested with Anaconda environment).
- Game dependencies: `pygame`, `numpy`, `matplotlib` (matplotlib only for `render()` debugging).
- Parameter fitting dependencies: `pandas`, `scipy`, `pybads` (for parameter optimization in `fitting.ipynb`).
  - IBS (Inverse Binomial Sampling): https://github.com/acerbilab/ibs
  - PyBADS: https://github.com/acerbilab/pybads

Quick installation (recommended in a virtual environment):
```bash
# Basic dependencies for game
pip install pygame numpy matplotlib

# Additional dependencies for parameter fitting
pip install pandas scipy pybads
```

## Quick Start
```bash
python play_game.py
```
- Click top buttons to toggle between PVP and PVE modes.
- Bottom buttons: Restart (resets current game), Exit (closes application).
- In PVE mode: player moves first (black pieces, player1), AI moves second (white pieces, player2).
- Click on board cells to place a piece; invalid moves (out of bounds or occupied cells) are rejected.
- When the game ends, the result is displayed in the center, and data is automatically saved:
  - **CSV file** `data/<mode>/<mode>_<timestamp>.csv`: Contains complete game_data with columns including board, play_to_move, action, done, winner, trial, time_elapsed, rt, etc.
  - **JSON file** `data/<mode>/<mode>-blocks-<timestamp>.json`: Block-format data grouped by player, used for parameter fitting.

## Code Structure
- `play_game.py`: Pygame UI main loop handling mode switching, mouse events, and data logging.
  - `reset_game(mode, agent_type)`: Initializes the environment, selects an agent (random / heuristic / bfs), and resets timing and logging.
  - `main()`: Renders UI, handles user input, drives AI moves in PVE mode, and saves CSV and block-JSON data when game ends.
- `env/chess.py`: Game environment and rules.
  - Board: 4×9, empty cells = 0.75, player1 (black) = 0, player2 (white) = 1.
  - Rules: legal action enumeration via `get_valid_actions`, win/draw checking via `check_win`/`check_draw`, single-step transition via `transit`, high-level interface `step`, visualization via `render`.
- `env/agent.py`: Agent implementations and behavioral hypothesis models.
  - `RandomAgent`: Selects random legal moves.
  - `HeuristicAgent`: Computes 5 features (center, connected-2, unconnected-2, connected-3, connected-4) and uses `minmax` for greedy action selection at the root; supports feature weights and own-vs-opponent coefficient `C`. Also serves as the base hypothesis model for human behavior.
  - `BFSAgent`: Extends heuristic scoring with forward search and pruning.
    - `plan()`: Can make random errors (controlled by `lmbda`), iteratively selects/expands/backtracks, terminates based on `gamma` or action stability.
    - `drop_feature(delta)`: Randomly drops features; `theta` controls pruning width; `C` controls own-vs-opponent weight balance.
    - Parameters can be fitted to model human player decision-making.
- `fitting.ipynb`: Interactive parameter fitting notebook implementing the complete fitting pipeline.
  - Cell 1: Data loading and preprocessing from block-json files in `data/<mode>/`.
  - Cell 2: Parameter optimization using BADS and IBS to minimize NLL and recover Agent parameters.
  - Cell 3: Results analysis and visualization, comparing fitted parameters with ground truth.
- `data/`: Game log directory, auto-generated when games are played.

## AI Parameters (default values in `default_params`)
- `lmbda`: Error probability; `gamma`: Search termination probability; `theta`: Pruning threshold; `delta`: Feature dropout rate; `C`: Feature weight ratio for own vs. opponent.
- Weights: `w_ce` (center), `w_c2` (connected-2), `w_u2` (unconnected-2), `w_c3` (connected-3), `w_c4` (connected-4).

### Parameter Fitting (Implemented in `fitting.ipynb`)
- **Goal**: Recover Agent parameters $\Theta = [\lambda, \gamma, \theta, \delta, C, w_{ce}, w_{c2}, w_{u2}, w_{c3}, w_{c4}]$ from game data by minimizing Negative Log-Likelihood (NLL), and validate fitting algorithm effectiveness.

- **Data Preparation**:
  1) Select a block-json file from `data/<mode>/` containing the move sequence and timing information.
  2) Extract observed data (design): Combination of board state `board_idx` and player identifier `player_id`.
  3) Extract response data (response): Corresponding move actions `action_idx`.

- **Fitting Process**:
  1) **Objective Function**: Use IBS (Inverse Binomial Sampling) to construct likelihood; compute likelihood of Agent's `response_generator` producing observed actions given parameters.
  2) **Optimizer**: Employ BADS (Bayesian Adaptive Direct Search) for efficient Bayesian optimization to minimize NLL.
  3) **Constraints**: Parameters bounded by hard bounds `p_bnds` and plausible bounds `p_pbnds` to ensure reasonable fitting results.

- **Evaluation**:
  1) Compare fitted parameters against ground truth (`default_params`); compute error $\Delta = \hat{\Theta} - \Theta_{\text{true}}$ for each parameter.
  2) Track NLL evolution and parameter trajectories during optimization to assess convergence and stability.
  3) Run optimization from multiple initializations to evaluate robustness of parameter recovery.

## Debugging & Extensions
- UI debugging: To speed up AI responses, reduce the value of `pygame.time.wait(300)`.
- Switching AI: The `agent_type` parameter in `reset_game` accepts `random`, `heuristic`, or `bfs`.
- Data inspection:
  - View `data/<mode>/*.csv` for detailed step-by-step behavioral data (board, action, time_elapsed, rt, etc.).
  - View `data/<mode>/*-blocks-*.json` for player-grouped compact data used in `fitting.ipynb` parameter fitting.

