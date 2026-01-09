# VLM-Assisted Search Agent

## Overview

The **VLMAssistedAgent** combines the strategic understanding of Vision-Language Models (VLMs) with the computational power of search algorithms. Instead of directly outputting shot parameters, the VLM provides high-level strategic guidance that informs the search process.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Billiards Environment                       │
│               (balls, my_targets, table)                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────▼────────────────────┐
         │    VLM-Assisted Agent              │
         │  (VlmAssistedAgent.py)             │
         └───────────────┬────────────────────┘
                         │
         ┌───────────────▼────────────────────┐
         │  1. VLM Strategic Analysis         │
         │     - Promising targets (3 balls)  │
         │     - Risk assessment (0-1)        │
         │     - Budget allocation (0-1)      │
         └───────────────┬────────────────────┘
                         │
         ┌───────────────▼────────────────────┐
         │  2. Search Agent (base_search)     │
         │     - Candidate generation         │
         │     - Simulation & evaluation      │
         │     - Two-stage selection          │
         └───────────────┬────────────────────┘
                         │
         ┌───────────────▼────────────────────┐
         │  3. Final Shot Parameters          │
         │     (V0, phi, theta, a, b)         │
         └────────────────────────────────────┘
```

## VLM Guidance Parameters

The VLM analyzes the game state image and outputs three strategic parameters:

### 1. Promising Targets (List[str])

**What it is:** 3 ball IDs that are most likely to yield good results

**How it's used:**
- Candidate generation prioritizes these balls (70% of candidates)
- Other balls still considered (30% of candidates)
- Ensures search focuses on high-value targets

**Example:**
```python
promising_targets = ["1", "3", "5"]
# Agent will generate more shot candidates aimed at balls 1, 3, and 5
```

### 2. Risk (float 0.0 - 1.0)

**What it is:** Assessment of how risky the current game situation is

**How it's used:**
- Adjusts `risk_lambda` parameter (risk-aversion in search)
  - Risk 0.0 → lambda 0.1 (aggressive, favor high-reward shots)
  - Risk 1.0 → lambda 0.5 (conservative, favor stable shots)
- Modifies reward shaping:
  - High risk → penalize negative outcomes more heavily
  - High risk → slightly reduce positive outcomes (be cautious)

**Example:**
```python
risk = 0.8  # High risk situation
# Agent becomes more conservative, avoiding risky shots
# Prefers shots with consistent outcomes over high-variance shots
```

### 3. Budget (float 0.0 - 1.0)

**What it is:** How complex the game situation is (affects computational effort)

**How it's used:**
- Adjusts number of simulations:
  - Budget 0.0 → 60 simulations (simple situation)
  - Budget 0.5 → 120 simulations (moderate)
  - Budget 1.0 → 240 simulations (complex)
- More budget = more thorough search = better decisions (but slower)

**Example:**
```python
budget = 0.9  # Very complex situation
# Agent will run 216 simulations to find the best shot
# Takes longer but makes better decisions in complex scenarios
```

## Usage

### Basic Usage

```python
from agents.vlm_agents import VLMAssistedAgent

# Create agent
agent = VLMAssistedAgent(
    provider='qwen',
    model='qwen-vl-max',
    use_vlm=True,
    vlm_frequency='always'
)

# Make decision
action = agent.decision(
    balls=balls,
    my_targets=['1', '2', '3'],
    table=table
)
```

### Configuration Options

```python
agent = VLMAssistedAgent(
    provider='qwen',              # 'qwen', 'openai', 'claude'
    model='qwen-vl-max',          # VLM model name
    api_key=None,                 # Or set OPENAI_API_KEY env var
    base_url=None,                # Optional custom API endpoint
    use_vlm=True,                 # Enable/disable VLM guidance
    vlm_frequency='always'        # 'always', 'first_n', 'adaptive'
)
```

### VLM Frequency Modes

**1. `'always'` (default):**
- Use VLM for every decision
- Most accurate but slowest
- Recommended for important games

**2. `'first_n'`:**
- Use VLM for first N decisions (default: 10)
- Reuse last guidance for subsequent decisions
- Good balance of accuracy and speed

**3. `'adaptive'`:**
- Use VLM on first decision and when ≤3 balls remain
- Automatic mode for critical moments
- Efficient for long games

### Pure Search Mode

```python
# Disable VLM to use as pure search agent
agent = VLMAssistedAgent(use_vlm=False)
# Equivalent to base SearchAgentV1_5
```

## How VLM Guidance Affects Search

### 1. Candidate Generation

**Without VLM:**
```python
# All targets treated equally
candidates = generate_all_candidates(all_targets)
```

**With VLM:**
```python
# Promising targets get priority
promising_candidates = generate_candidates(promising_targets)  # 70%
other_candidates = generate_candidates(other_targets)          # 30%
candidates = promising_candidates + other_candidates
```

### 2. Risk-Adjusted Evaluation

**Without VLM:**
```python
# Fixed risk-aversion
estimate = mean - 0.3 * std
```

**With VLM:**
```python
# Dynamic risk-aversion based on VLM assessment
risk_lambda = 0.1 + 0.4 * risk  # Range: 0.1 to 0.5
estimate = mean - risk_lambda * std

# Also adjust reward shaping
if shaped_reward < 0:
    shaped_reward *= (1.0 + risk * 0.3)  # Penalize failures more when risky
```

### 3. Budget-Based Search Depth

**Without VLM:**
```python
# Fixed simulation count
n_simulations = 180
```

**With VLM:**
```python
# Dynamic based on complexity
n_simulations = 180 * (0.33 + 1.0 * budget)
# Range: 60 (simple) to 240 (complex)
```

## Example VLM Output

Given this game state:
- My balls: 1, 2, 3 (remaining)
- Enemy balls: 9, 10, 11, 12, 13 (remaining)
- Situation: Behind (need to catch up)
- Table: Cluttered with many balls

VLM might output:
```json
{
    "promising_targets": ["1", "2", "3"],
    "risk": 0.75,
    "budget": 0.85,
    "reasoning": "Behind in game, need aggressive play but table is cluttered. Ball 1 has clearest path to corner pocket. Balls 2 and 3 are also accessible. High risk due to deficit. High budget needed due to cluttered table requiring careful shot selection."
}
```

This guidance tells the agent:
1. **Focus on balls 1, 2, 3** (generate more candidates for these)
2. **Be somewhat conservative** (risk=0.75 → lambda=0.4, avoid very risky shots)
3. **Search thoroughly** (budget=0.85 → 204 simulations)

## Performance Characteristics

### Computational Cost

**VLM Call:**
- Time: 3-10 seconds per call
- Cost: Varies by provider (Qwen: ~$0.01-0.05 per call)

**Search:**
- Time: 5-20 seconds (depends on budget)
- Cost: Free (local computation)

**Total per decision:**
- With VLM: 8-30 seconds
- Without VLM (reuse): 5-20 seconds

### Accuracy

**Compared to pure search:**
- Better target selection (VLM identifies best opportunities)
- Better risk management (adapts to game situation)
- Better resource allocation (more search when needed)

**Compared to pure VLM:**
- Much better shot execution (search finds optimal parameters)
- More robust (search handles edge cases)
- Better under noise (search evaluates multiple scenarios)

## Environment Variables

The agent inherits all environment variables from `base_search_agent.py`:

```bash
# Search parameters
export BILLIARDS_MAX_SIMULATIONS=180        # Base simulation count
export BILLIARDS_RISK_LAMBDA=0.3            # Base risk-aversion (overridden by VLM)
export BILLIARDS_STAGE1_N=90                # Stage 1 candidates
export BILLIARDS_STAGE1_R=1                 # Stage 1 rollouts per candidate
export BILLIARDS_STAGE2_K=12                # Stage 2 finalists
export BILLIARDS_STAGE2_M=3                 # Stage 2 additional rollouts

# Candidate generation
export BILLIARDS_MAX_CANDIDATES=500
export BILLIARDS_PREFILTER_KEEP=90

# No-foul library
export BILLIARDS_NOFOUL_COUNT=10
export BILLIARDS_NOFOUL_V0S=2.2,3.0
export BILLIARDS_NOFOUL_ANGLE_OFFS=0,7,-7
```

## Testing

```bash
# Test VLM-Assisted Agent
cd agents/vlm_agents
python VlmAssistedAgent.py

# Test with different VLM frequencies
python -c "
from VlmAssistedAgent import VLMAssistedAgent
import pooltool as pt

agent = VLMAssistedAgent(vlm_frequency='adaptive')
# ... test code ...
"
```

## Integration with Evaluation

```python
from agents.vlm_agents import VLMAssistedAgent
from poolenv import PoolEnv

# Create environment and agent
env = PoolEnv()
agent = VLMAssistedAgent(
    provider='qwen',
    model='qwen-vl-max',
    vlm_frequency='adaptive'  # Efficient for evaluation
)

# Game loop
env.reset()
while not env.get_done():
    balls, my_targets, table = env.get_observation()
    action = agent.decision(balls, my_targets, table)
    result = env.take_shot(action)
```

## Advantages

1. **Strategic Understanding**: VLM provides human-like strategic assessment
2. **Adaptive Behavior**: Adjusts playstyle based on game situation
3. **Efficient Search**: Focuses computational resources where needed
4. **Robust Execution**: Search ensures reliable shot parameters
5. **Explainable**: VLM provides reasoning for decisions

## Limitations

1. **Slower**: VLM calls add 3-10 seconds per decision
2. **API Dependency**: Requires internet and API access
3. **Cost**: VLM calls cost money (though small)
4. **Complexity**: More moving parts than pure search or pure VLM

## Tips for Best Performance

1. **Use `vlm_frequency='adaptive'`** for evaluation (good balance)
2. **Set API key as environment variable** for security
3. **Monitor VLM call count** to control costs
4. **Use `use_vlm=False`** for testing search logic without API calls
5. **Adjust default risk/budget** if VLM unavailable

## Comparison with Other Agents

| Agent Type | Speed | Accuracy | Adaptability | Cost |
|------------|-------|----------|--------------|------|
| LLMAgent | Fast | Low | Low | Low |
| VLMAgent | Medium | Medium | Medium | Medium |
| SearchAgent | Medium | High | Low | Free |
| **VLMAssistedAgent** | **Slow** | **Highest** | **Highest** | **Medium** |

## Future Improvements

1. **Caching**: Cache VLM guidance for similar game states
2. **Confidence Scores**: VLM outputs confidence for each parameter
3. **Multi-shot Planning**: VLM suggests strategy for next N shots
4. **Opponent Modeling**: VLM predicts opponent's likely moves
5. **Fine-tuning**: Fine-tune VLM on billiards-specific data
6. **Hybrid Frequency**: Learn optimal VLM call frequency

---

**Summary:** VLMAssistedAgent combines the best of both worlds—VLM's strategic understanding and search's computational power—to create a highly capable billiards agent that adapts to game situations and makes intelligent decisions.


