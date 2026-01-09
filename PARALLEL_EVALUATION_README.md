# Parallel Agent Evaluation

This document explains how to use the parallel evaluation scripts to test your agents much faster.

## Why Parallel Evaluation?

**Sequential evaluation** (`evaluate.py`):
- Runs one game at a time
- 120 games takes 6-10 hours
- Wastes CPU resources (only uses 1 core)

**Parallel evaluation** (`evaluate_parallel.py` / `evaluate_parallel_advanced.py`):
- Runs multiple games simultaneously
- 120 games takes 30-90 minutes (on 8+ core CPU)
- Fully utilizes your CPU

## Quick Start

### Method 1: Simple Parallel Evaluation

```bash
# Activate environment
conda activate poolenv

# Run 40 games in parallel (auto-detects CPU cores)
python evaluate_parallel.py
```

**Note**: Edit the agent configuration inside `evaluate_parallel.py`:

```python
# Line ~51-58 in evaluate_parallel.py
agent_a = BasicAgentPro()
agent_b = MCTSAgent(
    n_simulations=60,
    n_noise_samples=5,
    risk_aversion=0.5
)
```

### Method 2: Advanced Parallel Evaluation (Recommended)

The advanced version supports command-line configuration:

```bash
# Basic usage (40 games)
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent

# Full evaluation (120 games)
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --games 120

# Custom number of workers
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --workers 16

# With fixed random seed (reproducible)
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --seed 42

# Save results to file
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --games 120 --output results.json
```

### List Available Agents

```bash
python evaluate_parallel_advanced.py --list-agents
```

## Available Agents

- **Base Agents**: `BasicAgent`, `BasicAgentPro`, `NewAgent`
- **Heuristic Agents**: `HeuristicAgent`, `DynamicHeuristicAgent`, `GlobalDynamicAgent`, `GlobalDynamicAgentOptimized`, `ParallelDynamicAgent`, `StrategicParallelAgent`
- **MCTS Agents**: `MCTSAgent`, `EnhancedMCTSAgent`, `ParallelMCTSAgent`
- **VLM Agents**: `VLMAssistedAgent`

## Performance Comparison

### CPU Core Scaling

| CPU Cores | 40 Games | 120 Games | Speedup |
|-----------|----------|-----------|---------|
| 1 (sequential) | ~2 hours | ~6-10 hours | 1x |
| 4 cores | ~30 min | ~90-120 min | ~4x |
| 8 cores | ~15 min | ~45-60 min | ~8x |
| 16 cores | ~8 min | ~25-35 min | ~15x |

**Note**: Actual speedup depends on:
- Agent computation time
- Physics simulation overhead
- System I/O performance

### Recommended Settings

**For Testing (Quick)**:
```bash
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --games 40
```
- Takes ~10-20 minutes
- Good for quick comparison

**For Final Evaluation**:
```bash
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --games 120 --output final_results.json
```
- Takes ~30-90 minutes
- Provides statistically significant results

## Understanding the Output

### Live Progress Display

```
[████████████████████████████████░░░░░░░░] 80.0% | 完成: 96/120 | A: 52 B: 42 平: 2 | 胜率: 54.2% | 剩余: 5m12s
```

- **Progress Bar**: Visual completion indicator
- **完成**: Completed games / Total games
- **A/B/平**: Wins for Agent A / Agent B / Draws
- **胜率**: Agent A win rate
- **剩余**: Estimated time remaining

### Final Results

```
最终结果:
  BasicAgentPro (Agent A):
    胜利: 52 局
    得分: 53.0
    胜率: 43.33%

  MCTSAgent (Agent B):
    胜利: 68 局
    得分: 68.0
    胜率: 56.67%

  平局: 0 局

游戏统计:
  平均击球数: 24.3
  最少击球数: 8
  最多击球数: 60

性能统计:
  平均游戏时长: 45.2s
  最快游戏: 15.3s
  最慢游戏: 180.7s

总用时: 35m 42s
平均每局: 17.9s
并行加速比: ~8.0x
```

## Common Issues

### Issue 1: "No module named 'pooltool'"

**Solution**: Activate the conda environment first:
```bash
conda activate poolenv
python evaluate_parallel_advanced.py ...
```

### Issue 2: Too Many Processes / System Slowdown

**Solution**: Reduce the number of workers:
```bash
# Use only 4 cores instead of auto-detect
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --workers 4
```

### Issue 3: Out of Memory

**Solution**: Reduce workers or close other applications:
```bash
# Use fewer workers
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --workers 4
```

### Issue 4: Process Hangs or Crashes

**Possible causes**:
- Agent has infinite loop or deadlock
- Physics simulation timeout
- Memory leak

**Solution**: 
1. Test agent with sequential evaluation first
2. Check agent logs for errors
3. Reduce complexity of agent (fewer simulations)

## Tips for Faster Evaluation

### 1. Optimize Agent Parameters

For parallel evaluation, consider reducing simulation counts:

```python
# Instead of:
MCTSAgent(n_simulations=100, n_noise_samples=7)

# Use:
MCTSAgent(n_simulations=60, n_noise_samples=5)
```

This reduces per-game time while maintaining good performance.

### 2. Use Appropriate Number of Workers

**Too few workers**: Wastes CPU resources
**Too many workers**: Overhead from context switching

**Optimal**: 80% of CPU cores (default in our script)

```python
# Auto-detect (recommended)
python evaluate_parallel_advanced.py --agent-a A --agent-b B

# Manual override
python evaluate_parallel_advanced.py --agent-a A --agent-b B --workers 8
```

### 3. Batch Multiple Evaluations

Create a bash script to run multiple comparisons:

```bash
#!/bin/bash
# evaluate_all.sh

python evaluate_parallel_advanced.py --agent-a BasicAgent --agent-b MCTSAgent --games 120 --output results_1.json
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --games 120 --output results_2.json
python evaluate_parallel_advanced.py --agent-a BasicAgent --agent-b EnhancedMCTSAgent --games 120 --output results_3.json
```

Run overnight:
```bash
chmod +x evaluate_all.sh
nohup ./evaluate_all.sh > evaluation.log 2>&1 &
```

## Analyzing Results

### Load Results from JSON

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results.json', 'r') as f:
    data = json.load(f)

# Extract data
results = data['results']
stats = data['statistics']

# Calculate win rate
win_rate = results['AGENT_A_WIN'] / data['config']['n_games'] * 100

print(f"Agent A Win Rate: {win_rate:.2f}%")
print(f"Average Game Time: {stats['avg_game_time']:.1f}s")
print(f"Average Hit Count: {stats['avg_hit_count']:.1f}")

# Plot hit count distribution
plt.hist(stats['hit_counts'], bins=20)
plt.xlabel('Hit Count')
plt.ylabel('Frequency')
plt.title('Distribution of Hit Counts')
plt.savefig('hit_count_distribution.png')
```

### Statistical Significance

For reliable results, use at least 40 games (preferably 120):

- **40 games**: ±10% confidence interval
- **120 games**: ±6% confidence interval

## Advanced Usage

### Running on Remote Server

```bash
# SSH to server
ssh user@server

# Start screen session (survives disconnect)
screen -S evaluation

# Activate environment
conda activate poolenv

# Run evaluation
python evaluate_parallel_advanced.py --agent-a BasicAgentPro --agent-b MCTSAgent --games 120 --output results.json

# Detach: Ctrl+A then D
# Reattach: screen -r evaluation
```

### Multiple Evaluations in Parallel

If you have a very powerful server (32+ cores), you can run multiple evaluations:

```bash
# Terminal 1 (cores 0-7)
taskset -c 0-7 python evaluate_parallel_advanced.py --agent-a A --agent-b B --workers 8 --output r1.json

# Terminal 2 (cores 8-15)
taskset -c 8-15 python evaluate_parallel_advanced.py --agent-a C --agent-b D --workers 8 --output r2.json
```

## Troubleshooting

### Enable Detailed Logging

Modify the scripts to add logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Resources

```bash
# Monitor CPU usage
htop

# Monitor memory
free -h

# Check processes
ps aux | grep python
```

## Summary

| Feature | evaluate.py | evaluate_parallel.py | evaluate_parallel_advanced.py |
|---------|------------|---------------------|------------------------------|
| Parallel | ❌ | ✅ | ✅ |
| CLI Args | ❌ | ❌ | ✅ |
| Progress Bar | ❌ | ✅ | ✅ (Enhanced) |
| Save Results | ❌ | ❌ | ✅ |
| Auto Core Detection | N/A | ✅ | ✅ |
| Speed (120 games) | 6-10 hours | 30-90 min | 30-90 min |
| **Recommended For** | **Quick test** | **Batch eval** | **Production eval** ⭐ |

## Further Reading

- `agents/search_agents/MonteCarlo/README.md` - Robust MCTS agent documentation
- `evaluate.py` - Original sequential evaluation script
- Python multiprocessing docs: https://docs.python.org/3/library/multiprocessing.html


