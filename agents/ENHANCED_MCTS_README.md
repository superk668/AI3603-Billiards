# EnhancedMCTSAgent - Advanced Monte Carlo Agent

## Overview

`EnhancedMCTSAgent` is an improved version of `BasicAgentPro` that uses more sophisticated Monte Carlo Tree Search strategies to outperform the baseline agent. It implements several key improvements to prioritize searching promising branches more effectively.

## Key Improvements Over BasicAgentPro

### 1. **Strategic Action Generation with Difficulty-Based Prioritization**
- **Problem**: BasicAgentPro generates actions uniformly for all target balls and pockets
- **Solution**: Calculate difficulty scores for each (ball, pocket) combination based on:
  - Distance from cue ball to target ball
  - Distance from target ball to pocket
  - Required cut angle (more difficult for extreme angles)
- **Impact**: Prioritizes easier shots that have higher success probability

### 2. **Progressive Action Refinement**
- **Problem**: BasicAgentPro searches a fixed action space without refinement
- **Solution**: Two-phase search approach:
  - Phase 1 (60% of simulations): Explore initial coarse action space
  - Phase 2 (40% of simulations): Refine promising actions (score ≥ 0.6) with small variations
- **Impact**: Better exploitation of good actions while maintaining exploration

### 3. **Multi-Level Evaluation System**
- **Problem**: BasicAgentPro only considers immediate rewards (potting balls, fouls)
- **Solution**: Combined evaluation function:
  - Immediate reward (potting, fouls, legal hits)
  - Position quality assessment:
    - Target balls' proximity to pockets
    - Cue ball positioning for next shot
    - Safety considerations (avoiding rails/pockets)
- **Impact**: Makes more strategic decisions considering future playability

### 4. **Adaptive Exploration Parameter**
- **Problem**: Fixed exploration coefficient doesn't adapt to game state
- **Solution**: Dynamically adjust `c_puct` based on remaining target balls:
  ```python
  adaptive_c_puct = base_c_puct * (1.0 + 0.1 * n_remaining)
  ```
- **Impact**: More exploration when many balls remain, more exploitation when closing game

### 5. **Improved Action Variations**
- **Problem**: BasicAgentPro uses fixed angle variations (±0.5°)
- **Solution**: 
  - Smaller initial variations (±0.3°) for better precision
  - Multiple velocity variations
  - Spin variations (a, b parameters) for top shots
- **Impact**: Higher precision while maintaining robustness

## Usage

### Basic Usage

```python
from agents import EnhancedMCTSAgentBase

# Initialize the agent
agent = EnhancedMCTSAgentBase(
    n_simulations=50,           # Number of MCTS simulations
    base_c_puct=1.414,          # Base exploration coefficient
    refinement_threshold=0.6,   # Threshold for action refinement
    position_weight=0.3         # Weight for position quality (0-1)
)

# Make a decision
action = agent.decision(balls=balls, my_targets=my_targets, table=table)
```

### Parameter Tuning

**For Better Performance (More Compute):**
```python
agent = EnhancedMCTSAgentBase(
    n_simulations=100,          # More simulations
    base_c_puct=1.2,            # Slightly less exploration
    refinement_threshold=0.55,  # Refine more actions
    position_weight=0.35        # Stronger position consideration
)
```

**For Faster Decisions:**
```python
agent = EnhancedMCTSAgentBase(
    n_simulations=30,           # Fewer simulations
    base_c_puct=1.6,            # More exploration (compensate for fewer sims)
    refinement_threshold=0.65,  # Refine only best actions
    position_weight=0.25        # Less position overhead
)
```

**For Endgame Optimization:**
```python
agent = EnhancedMCTSAgentBase(
    n_simulations=50,
    base_c_puct=1.0,            # Less exploration in endgame
    refinement_threshold=0.55,
    position_weight=0.4         # Position more important in endgame
)
```

## Testing Against BasicAgentPro

To evaluate the agent's performance:

```bash
# If you have an evaluation script
python evaluate.py --agent1 EnhancedMCTSAgentBase --agent2 BasicAgentPro --games 100
```

## Expected Performance Gains

Based on the improvements:
- **Win Rate**: Expected 55-65% win rate against BasicAgentPro (depending on parameters)
- **Better Endgame**: Particularly strong in endgame scenarios with fewer balls
- **More Consistent**: Position quality evaluation leads to more consistent play
- **Fewer Fouls**: Strategic action generation reduces risky shots

## Architecture Comparison

| Feature | BasicAgentPro | EnhancedMCTSAgent |
|---------|---------------|-------------------|
| Action Generation | Uniform (all balls/pockets) | Difficulty-prioritized |
| Search Strategy | Single-phase exploration | Two-phase (explore + refine) |
| Evaluation | Immediate reward only | Multi-level (reward + position) |
| Exploration | Fixed c_puct | Adaptive c_puct |
| Action Precision | ±0.5° variations | ±0.3° variations + refinement |
| Simulations per Decision | ~50 | ~50 (same, but smarter allocation) |

## Technical Details

### Difficulty Calculation Formula

```python
difficulty = distance_cue_to_obj * 0.3 + distance_obj_to_pocket * 0.5 + abs(cut_angle) * 0.5
```

### Position Quality Scoring

Position quality is evaluated on [-1, 1] scale considering:
1. **Target ball positioning** (30% weight): Average distance to nearest pocket
2. **Cue ball proximity** (20% weight): Optimal distance to nearest target (0.3-1.2m)
3. **Safety margins** (15% weight): Distance from rails and pockets

### Refinement Strategy

Actions are refined if their normalized score ≥ `refinement_threshold`:
- Generate 3-4 variations with small perturbations
- Allocate remaining simulations to refined action space
- Compare best refined vs best original action

## Future Improvements

Potential enhancements for even better performance:
1. **Multi-step lookahead**: Consider 2-3 shot sequences
2. **Opponent modeling**: Adapt strategy based on opponent strength
3. **Learning from games**: Update difficulty/quality functions based on outcomes
4. **Parallel simulation**: Speed up MCTS with multiprocessing
5. **Neural network evaluation**: Replace hand-crafted position quality with learned function

## License

Same as parent project.

## Author

Created as an enhancement to the AI3603-Billiards project.

