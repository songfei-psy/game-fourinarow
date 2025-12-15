# Four in a Row (4x9)

Four in a Row (4x9) with Pygame UI and several agents (player vs player, player vs AI). 本项目受
Van Opheusden et al. (2023) “Expertise increases planning depth in human gameplay” 启发，目标是复现并探索其中关于人类规划深度与启发式权重差异的发现，并在拟合人类对局数据（`default_params` 参数族）时对比不同玩家的偏好与深度。

## 功能概览
- 4x9 棋盘，四子连线胜出（横、竖、两条对角线）。
- 模式：PVP（双人）、PVE（玩家对 AI）。
- AI 选项：随机 `RandomAgent`、启发式 `HeuristicAgent`、带前向搜索的 `BFSAgent`。
- 界面：Pygame 绘制棋盘、按钮，支持重开与退出。
- 数据记录：每步坐标与落子时间毫秒级记录到 `data/`，文件名包含模式和时间戳。

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
- 游戏结束后会在中央显示结果，并写入 `data/<MODE>_<timestamp>.json`。

## 代码结构
- `play_game.py`：Pygame UI 主循环，模式切换、事件处理（鼠标）、日志保存。
	- `reset_game(mode, agent_type)`: 初始化环境、选择 agent（random / heuristic / bfs），并重置计时与日志。
	- `main()`: 绘制 UI、响应点击、在 PVE 中驱动 AI 落子并保存 `data/*.json`。
- `env/chess.py`：环境与规则。
	- 棋盘：4x9，空位值 0.75，先手黑 0，后手白 1。
	- 规则：合法动作枚举 `get_valid_actions`，胜负/和判定 `check_win`/`check_draw`，一步转移 `transit`，高层接口 `step`，可视化 `render`。
- `env/agent.py`：智能体实现。
	- `RandomAgent`: 纯随机合法落子。
	- `HeuristicAgent`: 计算 5 个特征（中心、连2、非连2、连3、连4），使用 `minmax` 在根结点选择贪心动作；支持特征权重与到手方系数 `C`。
	- `BFSAgent`: 继承启发式评分，加入前向搜索与剪枝。
		- `plan()`: 可随机失误 (`lmbda`)，循环选择/扩展/回溯，按 `gamma` 或动作稳定性终止。
		- `drop_feature(delta)`: 随机丢弃特征；`theta` 控制剪枝宽度；`C` 控制到手方/非到手方权重。
- `data/`: 对局日志目录，运行生成。
- `env/fitting.py`（计划补全）：拟合人类玩家参数（参见 `default_params`），对比不同玩家在 `lmbda/gamma/theta/delta/C` 与特征权重上的差异。

## AI 参数（默认值见 `default_params`）
- `lmbda`：失误概率；`gamma`：停止搜索概率；`theta`：剪枝阈值；`delta`：特征丢弃率；`C`：到手方与非到手方的特征权重比。
- 权重：`w_ce`（中心）、`w_c2`（连2）、`w_u2`（非连2）、`w_c3`（连3）、`w_c4`（连4）。

### 参数拟合（规划）
- 目标：基于人类对局数据拟合上述参数，观察个体或群体在注意力/谨慎程度（`C`、`theta`）、随机性（`lmbda`）、搜索深度控制（`gamma`）、特征偏好（`w_*`）上的差异。
- 思路（计划在 `env/fitting.py` 实现）：
	1) 数据：使用 `data/*.json` 的落子序列与时间戳作为观测。
	2) 模型：给定参数 $
		 \Theta = [\lambda, \gamma, \theta, \delta, C, w_{ce}, w_{c2}, w_{u2}, w_{c3}, w_{c4}]$
		 ，用 agent 的 `response_generator` 对同一局面生成动作分布（或贪心动作），计算与人类动作的似然/距离。
	3) 目标：最大化似然或最小化负对数似然；可用粒子群/贝叶斯优化或简单网格+贪心微调。
	4) 输出：每名玩家的拟合参数与置信区间/多次启动的稳定性分析。
	5) 评估：重放对局计算命中率，或用拟合参数驱动 agent 与原局面对比动作一致率。

## 调试与扩展
- 调试 UI：如需更快 AI 响应可降低 `pygame.time.wait(300)`。
- 更换 AI：`reset_game` 的 `agent_type` 允许 `random` / `heuristic` / `bfs`。
- 录制数据：查看 `data/*.json`，包含 mode、每步坐标与毫秒时间、胜者。

## 已知设定
- 棋盘尺寸固定为 4 行 9 列，先手黑子（player1），后手白子（player2），空位标记为 `0.75`。
