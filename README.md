# Four in a Row (4x9)

Four in a Row (4x9) with Pygame UI and several agents (player vs player, player vs AI). 本项目受
Van Opheusden et al. (2023) "Expertise increases planning depth in human gameplay" 启发。

## 项目核心

本项目包含两个主要部分：

1. **四子棋游戏与行为数据采集**：实现了一个完整的四子棋游戏，支持人机对弈（PVE）、双人对弈（PVP）等多种模式。游戏可以实时采集玩家行为数据，包括每步落子位置与落子时间等信息。

2. **智能算法与假设行为模型**：游戏中的智能算法（AI Agent）实现在 `env/agent.py` 中。关键的一点是，这个 Agent 同时也被用作**人类行为的假设模型**。
   
   具体地说，实际应用中我们只有人类玩家的行为数据，希望用一个假设模型来拟合这些数据。为了评估参数拟合算法的优劣，我们采用了一个巧妙的验证方法：
   - **给定一个"正确的"假设模型**（即我们的 Agent 实现），用其生成合成数据
   - **然后用参数拟合算法去重新拟合这个合成数据**，恢复出原始的模型参数
   - 通过对比恢复的参数与真实参数，评估拟合算法的准确性与可靠性

## 功能概览
- 4x9 棋盘，四子连线胜出（横、竖、两条对角线）。
- 模式：PVP（双人）、PVE（玩家对 AI）。
- AI 选项：随机 `RandomAgent`、启发式 `HeuristicAgent`、带前向搜索的 `BFSAgent`。
- 界面：Pygame 绘制棋盘、按钮，支持重开与退出。
- 数据记录：每步坐标与落子时间实时记录到 `data/<mode>/`，支持 CSV 和 JSON 两种格式。

## 运行要求
- Python 3.8+（本地测试使用 Anaconda 环境）。
- 依赖：`pygame`, `numpy`, `matplotlib`（仅用于 `render()` 调试）。

快速安装（推荐在虚拟环境中）：
```bash
pip install pygame numpy matplotlib
```

## 快速开始
```bash
python play_game.py
```
- 顶部按钮切换 PVP / PVE。
- 底部按钮：Restart 重开；Exit 退出。
- PVE 下，玩家先手（黑子，player1），AI 后手（白子，player2）。
- 每步点击棋盘落子；越界或已占用格子无效。
- 游戏结束后会在中央显示结果，并自动保存数据：
  - **CSV 文件** `data/<mode>/<mode>_<timestamp>.csv`：包含完整的 game_data，列包括 board、play_to_move、action、done、winner、trial、time_elapsed、rt 等。
  - **JSON 文件** `data/<mode>/<mode>-blocks-<timestamp>.json`：按玩家分组的 block 格式数据，用于参数拟合。

## 代码结构
- `play_game.py`：Pygame UI 主循环，模式切换、事件处理（鼠标）、日志保存。
	- `reset_game(mode, agent_type)`: 初始化环境、选择 agent（random / heuristic / bfs），并重置计时与日志。
	- `main()`: 绘制 UI、响应点击、在 PVE 中驱动 AI 落子，游戏结束时保存 CSV 和 block-JSON 数据。
- `env/chess.py`：环境与规则。
	- 棋盘：4x9，空位值 0.75，先手黑 0，后手白 1。
	- 规则：合法动作枚举 `get_valid_actions`，胜负/和判定 `check_win`/`check_draw`，一步转移 `transit`，高层接口 `step`，可视化 `render`。
- `env/agent.py`：智能体与假设行为模型实现。
	- `RandomAgent`: 纯随机合法落子。
	- `HeuristicAgent`: 计算 5 个特征（中心、连2、非连2、连3、连4），使用 `minmax` 在根结点选择贪心动作；支持特征权重与到手方系数 `C`。同时也被用作人类行为的基础假设模型。
	- `BFSAgent`: 继承启发式评分，加入前向搜索与剪枝。
		- `plan()`: 可随机失误 (`lmbda`)，循环选择/扩展/回溯，按 `gamma` 或动作稳定性终止。
		- `drop_feature(delta)`: 随机丢弃特征；`theta` 控制剪枝宽度；`C` 控制到手方/非到手方权重。
		- 参数可用于拟合人类玩家的决策行为。
- `fitting.ipynb`：交互式参数拟合笔记本，实现完整的拟合管道。
	- 第 1 单元：数据加载与预处理，从 `data/<mode>/` 的 block-json 文件中提取对局序列。
	- 第 2 单元：参数优化，使用 BADS 和 IBS 最小化 NLL，恢复 Agent 参数。
	- 第 3 单元：结果分析与可视化，对比拟合参数与真实参数的误差。
- `data/`: 对局日志目录，运行生成。

## AI 参数（默认值见 `default_params`）
- `lmbda`：失误概率；`gamma`：停止搜索概率；`theta`：剪枝阈值；`delta`：特征丢弃率；`C`：到手方与非到手方的特征权重比。
- 权重：`w_ce`（中心）、`w_c2`（连2）、`w_u2`（非连2）、`w_c3`（连3）、`w_c4`（连4）。

### 参数拟合（拟合管道实现在 `fitting.ipynb`）
- **目标**：基于对局数据，通过最小化负对数似然（Negative Log-Likelihood, NLL）来恢复 Agent 的参数 $\Theta = [\lambda, \gamma, \theta, \delta, C, w_{ce}, w_{c2}, w_{u2}, w_{c3}, w_{c4}]$，验证拟合算法的有效性。

- **数据准备**：
	1) 从 `data/<mode>/` 中选择 block-json 文件（含有对局的落子序列与时间信息）。
	2) 提取观测数据（design）：棋盘状态 `board_idx` 和玩家标识 `player_id` 的组合 $(\text{board\_idx}, \text{player\_id})$。
	3) 提取响应数据（response）：对应的落子动作 `action_idx`。

- **拟合过程**：
	1) **目标函数**：使用 IBS（Importance-Weighted Bayesian Sampling）方法构造似然函数，计算 Agent 的 `response_generator` 在给定参数下生成观测动作的似然。
	2) **优化器**：采用 BADS（Bayesian Adaptive Direct Search）进行高效的贝叶斯优化，搜索参数空间中最小化 NLL 的解。
	3) **约束**：参数受到硬边界 `p_bnds` 和偏好边界 `p_pbnds` 的限制，确保拟合结果在合理的搜索范围内。

- **评估**：
	1) 对比拟合得到的参数与真实参数（`default_params`），计算每个参数的误差 $\Delta = \hat{\Theta} - \Theta_{\text{true}}$。
	2) 追踪优化过程中 NLL 的演变与参数轨迹，评估收敛性与稳定性。
	3) 通过多次不同初始点的优化运行，评估参数恢复的鲁棒性。

## 调试与扩展
- 调试 UI：如需更快 AI 响应可降低 `pygame.time.wait(300)`。
- 更换 AI：`reset_game` 的 `agent_type` 允许 `random` / `heuristic` / `bfs`。
- 录制数据：
  - 查看 `data/<mode>/*.csv` 获取详细的逐步行为数据（board、action、time_elapsed、rt 等）。
  - 查看 `data/<mode>/*-blocks-*.json` 获取按玩家分组的精简数据，用于 `fitting.ipynb` 中的参数拟合。

