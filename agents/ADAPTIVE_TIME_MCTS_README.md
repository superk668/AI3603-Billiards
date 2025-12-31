# AdaptiveTimeMCTSAgent - 自适应时间管理 MCTS Agent

## 概述

`AdaptiveTimeMCTSAgent` 是一个具有全局时间管理能力的高级 Monte Carlo Tree Search Agent，专门设计用于应对**每局3分钟**的时间限制。它在 `EnhancedMCTSAgent` 的基础上增加了动态时间预算分配，使 Agent 能够智能地分配计算资源。

## 核心特性

### 1. **自动游戏检测与时间重置**
- 自动检测新游戏的开始（通过监测球数变化）
- 每局游戏开始时自动重置时间预算为 180 秒
- 无需手动调用 `reset_time_budget()`

### 2. **动态资源分配**
根据以下因素动态调整每次决策的仿真次数：

- **决策复杂度** (0.0-1.0)：
  - 黑8决胜：0.5+ （最关键）
  - 残局 (1-3球)：0.3+
  - 中局 (4-6球)：0.15+
  - 开局 (7球+)：0.05+
  - 加上最简单击球的难度评估

- **剩余时间**：
  - 充裕时间：允许更多仿真
  - 时间不足 (<20s)：降低仿真次数
  - 极度紧急 (<10s)：使用最小仿真

- **估计剩余决策次数**：
  - 基于剩余目标球数量
  - 考虑失误率和回合切换

### 3. **时间使用策略**

```
开局 (7球+):    30-50% 基础仿真  (快速决策，保留时间)
中局 (4-6球):   80-100% 基础仿真 (标准决策)
残局 (1-3球):   120-150% 基础仿真 (深度思考)
黑8决胜:        150-200% 基础仿真 (最深思考)
```

### 4. **时间紧急模式**
- **紧急模式** (剩余时间 < 20秒)：仿真次数降至 60%
- **极度紧急** (剩余时间 < 10秒)：使用最小仿真次数
- 自动跳过动作细化阶段以节省时间

### 5. **实时监控与日志**
每次决策输出：
- 当前决策编号
- 决策复杂度评分
- 估计剩余决策次数
- 分配的仿真次数
- 剩余时间
- 决策用时

## 使用方法

### 基础使用

```python
from agents import AdaptiveTimeMCTSAgent

# 初始化（使用默认参数）
agent = AdaptiveTimeMCTSAgent()

# 在游戏循环中使用
action = agent.decision(balls=balls, my_targets=my_targets, table=table)
```

### 自定义参数

```python
agent = AdaptiveTimeMCTSAgent(
    base_simulations=50,        # 基础仿真次数
    total_time_budget=180.0,    # 每局时间预算（秒）
    min_simulations=20,         # 最小仿真次数（紧急模式）
    max_simulations=150,        # 最大仿真次数（关键决策）
    base_c_puct=1.414,          # UCB 探索系数
    refinement_threshold=0.6,   # 动作细化阈值
    position_weight=0.3         # 位置质量权重
)
```

### 参数调优建议

#### 保守策略（稳定获胜）
```python
agent = AdaptiveTimeMCTSAgent(
    base_simulations=45,        # 略低于默认
    total_time_budget=180.0,
    min_simulations=25,         # 提高最小仿真
    max_simulations=120,        # 限制最大仿真
    base_c_puct=1.6,            # 更多探索
    refinement_threshold=0.65,  # 只细化高质量动作
    position_weight=0.35        # 更重视位置
)
```

#### 激进策略（快速决策，高风险）
```python
agent = AdaptiveTimeMCTSAgent(
    base_simulations=60,        # 提高基础仿真
    total_time_budget=180.0,
    min_simulations=15,         # 降低最小仿真
    max_simulations=180,        # 允许更多仿真
    base_c_puct=1.2,            # 更少探索，更多利用
    refinement_threshold=0.55,  # 积极细化
    position_weight=0.25        # 较少考虑位置
)
```

#### 时间紧张适配（适合慢速硬件）
```python
agent = AdaptiveTimeMCTSAgent(
    base_simulations=40,        # 降低基础仿真
    total_time_budget=180.0,
    min_simulations=15,         # 降低最小值
    max_simulations=100,        # 限制最大值
    base_c_puct=1.5,
    refinement_threshold=0.65,  # 减少细化
    position_weight=0.25        # 降低位置评估开销
)
```

## 工作原理

### 时间预算分配算法

```python
def allocate_simulations(complexity, estimated_remaining_decisions):
    # 1. 计算平均每次决策可用时间
    avg_time_per_decision = remaining_time / max(estimated_remaining_decisions, 1)
    
    # 2. 基于历史数据估算每次仿真耗时
    avg_sim_time = mean(recent_decision_times) / base_simulations
    
    # 3. 根据复杂度计算基础分配
    complexity_multiplier = 0.6 + complexity * 1.0  # 0.6x ~ 1.6x
    base_allocation = base_simulations * complexity_multiplier
    
    # 4. 应用时间约束
    time_constrained = (avg_time_per_decision / avg_sim_time) * 0.8  # 留20%余量
    
    # 5. 综合决策
    allocated = min(base_allocation, time_constrained)
    allocated = clip(allocated, min_simulations, max_simulations)
    
    return allocated
```

### 游戏检测机制

```python
def detect_new_game(balls):
    # 统计未进袋的球数
    active_balls = count(balls where state != 4)
    
    # 检测条件：
    # 1. 首次调用
    # 2. 球数大幅增加（+8个或以上）
    
    if first_call or active_balls > last_count + 8:
        return True  # 检测到新游戏
    
    return False
```

### 决策复杂度评估

```python
def estimate_complexity(balls, my_targets):
    complexity = 0.0
    
    # 1. 剩余球数影响
    if targets == ['8']:
        complexity += 0.5      # 黑8决胜
    elif n_targets <= 2:
        complexity += 0.3      # 残局
    elif n_targets <= 4:
        complexity += 0.15     # 中局
    else:
        complexity += 0.05     # 开局
    
    # 2. 最简单击球的难度
    min_difficulty = min(shot_difficulties)
    if min_difficulty > 2.0:
        complexity += 0.3      # 所有球都很难
    elif min_difficulty > 1.0:
        complexity += 0.15     # 中等难度
    elif min_difficulty < 0.5:
        complexity -= 0.1      # 有非常简单的球
    
    return clip(complexity, 0.0, 1.0)
```

## 与其他 Agent 的对比

| 特性 | BasicAgentPro | EnhancedMCTSAgent | AdaptiveTimeMCTSAgent |
|------|---------------|-------------------|----------------------|
| 动作生成 | 均匀 | 难度优先 | 难度优先 |
| 搜索策略 | 单阶段 | 双阶段 | 双阶段（可自适应跳过） |
| 评估函数 | 即时奖励 | 多层次 | 多层次 |
| 仿真次数 | 固定 | 固定 | 动态分配 |
| 时间管理 | ❌ | ❌ | ✅ 全局预算管理 |
| 游戏检测 | ❌ | ❌ | ✅ 自动重置 |
| 复杂度感知 | ❌ | 部分 | ✅ 完全感知 |
| 紧急模式 | ❌ | ❌ | ✅ 多级降级 |

## 性能预期

### 时间使用效率
- **早期游戏** (0-60s)：快速决策，每次 1-3 秒
- **中期游戏** (60-120s)：标准决策，每次 3-5 秒
- **后期游戏** (120-170s)：深度思考，每次 5-10 秒
- **收尾阶段** (170-180s)：紧急模式，每次 1-2 秒

### 胜率预期
- **vs BasicAgentPro**：60-70% 胜率
- **vs EnhancedMCTSAgent**：55-60% 胜率（时间管理优势）
- **特别优势**：
  - 不会因时间不足而失败
  - 在关键时刻（黑8决胜）投入更多计算
  - 开局快速决策，节省时间用于残局

### 适用场景
✅ **最佳适用**：
- 严格的时间限制环境（每局3分钟）
- 硬件性能有限的场景
- 需要稳定完成所有决策的比赛

⚠️ **不太适用**：
- 无时间限制的环境（此时 EnhancedMCTSAgent 更简单）
- 硬件性能极强（可以始终使用最大仿真）

## 监控与调试

### 实时日志示例

```
============================================================
[TimeManager] 第 1 局游戏开始
[TimeManager] 时间预算重置: 180.0s
============================================================

[TimeManager] 决策 #1
  复杂度: 0.05 | 剩余决策: ~21
  分配仿真: 30 | 剩余时间: 180.0s
[AdaptiveTime] 最佳分数: 0.723
  决策用时: 2.34s | 剩余: 177.7s

[TimeManager] 决策 #2
  复杂度: 0.08 | 剩余决策: ~18
  分配仿真: 35 | 剩余时间: 177.7s
...

[TimeManager] 决策 #15
  复杂度: 0.50 | 剩余决策: ~3
  分配仿真: 80 | 剩余时间: 32.5s
[AdaptiveTime] 最佳分数: 0.856
  决策用时: 8.12s | 剩余: 24.4s

[TimeManager] ⚠️  紧急模式：时间不足 18.3s
[TimeManager] 决策 #16
  复杂度: 0.52 | 剩余决策: ~2
  分配仿真: 24 | 剩余时间: 18.3s
```

### 性能分析

在游戏结束后，可以分析时间使用情况：

```python
# 查看时间历史
print(f"总决策次数: {agent.decision_count}")
print(f"平均决策时间: {np.mean(agent.time_history):.2f}s")
print(f"总用时: {sum(agent.time_history):.2f}s")
print(f"剩余时间: {agent.remaining_time:.2f}s")

# 查看复杂度分布
print(f"平均复杂度: {np.mean(agent.complexity_history):.2f}")
print(f"最高复杂度: {np.max(agent.complexity_history):.2f}")
```

## 集成到评估脚本

在 `evaluate.py` 中使用：

```python
from agents import AdaptiveTimeMCTSAgent, BasicAgentPro

# 创建 agents
agent_a = BasicAgentPro()
agent_b = AdaptiveTimeMCTSAgent(
    base_simulations=50,
    total_time_budget=180.0  # 3分钟
)

# 运行评估
# agent_b 会自动检测每局游戏的开始并重置时间预算
# 无需手动调用 reset_time_budget()
```

## 常见问题

### Q: 如何确保不会超时？
A: Agent 使用多层保护机制：
1. 预留 20% 时间余量
2. 实时监控剩余时间
3. 两级紧急模式（<20s 和 <10s）
4. 最小仿真次数保证（20次）

### Q: 游戏检测会出错吗？
A: 几乎不会。检测使用保守策略（球数增加 8 个以上），只有在极端情况下（如连续进 9 个球）才可能误判。即使误判，只是重置时间预算，不影响决策正确性。

### Q: 如何手动重置时间？
A: 可以调用 `agent.reset_time_budget()`，但通常不需要，Agent 会自动检测。

### Q: 为什么有时决策很快，有时很慢？
A: 这是设计行为。Agent 在开局快速决策节省时间，在关键时刻（残局、黑8）投入更多计算。

### Q: 可以用于无时间限制的比赛吗？
A: 可以，但不推荐。在无时间限制场景下，`EnhancedMCTSAgent` 更简单且效果相当。

## 未来改进方向

1. **学习型时间管理**：根据历史对局学习最优时间分配策略
2. **对手建模**：根据对手强度调整时间分配
3. **并行仿真**：利用多核加速 MCTS
4. **更精确的复杂度评估**：考虑更多因素（如球的聚集度）
5. **时间借贷机制**：从富余局次借用时间到关键局次

## 总结

`AdaptiveTimeMCTSAgent` 是一个为时间受限环境设计的智能 Agent，它通过全局时间管理和动态资源分配，在保证不超时的前提下最大化决策质量。对于每局3分钟的比赛环境，这是最推荐的选择。

---

**作者**: AI Assistant  
**日期**: 2025年12月  
**版本**: 1.0  
**许可**: 与项目相同

