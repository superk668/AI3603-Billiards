# VLM/LLM Agents for Billiards

This directory contains Vision-Language Model (VLM) and Large Language Model (LLM) agents for playing billiards.

## Overview

Two types of agents are implemented:

1. **LLM Agent** (`llmAgent.py`) - Uses pure text input
2. **VLM Agent** (`vlmAgent.py`) - Uses visual input (images) + text

Both agents leverage the same chat interface (`chat.py`) and drawer utilities (`drawer.py`).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Billiards Environment                    │
│                  (balls, my_targets, table)                  │
└────────────────────┬────────────────────┬────────────────────┘
                     │                    │
         ┌───────────▼──────────┐   ┌────▼──────────────────┐
         │    LLM Agent         │   │    VLM Agent          │
         │  (llmAgent.py)       │   │  (vlmAgent.py)        │
         └───────────┬──────────┘   └────┬──────────────────┘
                     │                    │
         ┌───────────▼────────────────────▼──────────────────┐
         │              Chat Interface (chat.py)             │
         │  - Text-only mode (Qwen3-8b, GPT-4, etc.)        │
         │  - Vision mode (Qwen3-vl-8b-instruct, GPT-4V)    │
         └───────────────────────┬───────────────────────────┘
                                 │
                     ┌───────────▼──────────┐
                     │  Drawer (drawer.py)  │
                     │  (VLM Agent only)    │
                     └──────────────────────┘
```

## Files

### Core Files

- **`chat.py`** - Unified chat interface for both LLM and VLM
  - Supports text-only mode (for LLM agent)
  - Supports vision mode (for VLM agent)
  - Providers: OpenAI, Claude, Qwen (Alibaba Cloud)
  - Default models:
    - Text: `qwen-plus` (pure text LLM)
    - Vision: `qwen-vl-max` (vision-language model)

- **`llmAgent.py`** - Pure text LLM agent
  - Converts game state to text description
  - Sends text to LLM
  - LLM outputs shot parameters (V0, phi, theta, a, b)
  - Falls back to random if LLM fails

- **`vlmAgent.py`** - Vision-language model agent
  - Draws game state as image using `drawer.py`
  - Sends image + supplementary text to VLM
  - VLM outputs shot parameters
  - Falls back to random if VLM fails

- **`drawer.py`** - Visualization utilities
  - Draws billiards table state as image
  - Color-coded balls (green=my targets, orange=enemy, red=cue ball)
  - Used by VLM agent

### Legacy Files

- **`VlmAssistedAgent.py`** - Advanced VLM-assisted MCTS agent (more complex)

## Usage

### LLM Agent (Text-Only)

```python
from agents.vlm_agents.llmAgent import LLMAgent

# Initialize agent
agent = LLMAgent(
    provider='qwen',           # 'qwen', 'openai', or 'claude'
    model='qwen-plus',         # Text-only model
    api_key=None               # Or set OPENAI_API_KEY env var
)

# Make decision
action = agent.decision(
    balls=balls,               # Dict of ball objects
    my_targets=['1', '2'],     # List of target ball IDs
    table=table                # Table object
)

# action = {'V0': 3.5, 'phi': 45.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0}
```

### VLM Agent (Vision-Based)

```python
from agents.vlm_agents.vlmAgent import VLMAgent

# Initialize agent
agent = VLMAgent(
    provider='qwen',           # 'qwen', 'openai', or 'claude'
    model='qwen-vl-max',       # Vision model
    api_key=None               # Or set OPENAI_API_KEY env var
)

# Make decision (same interface as LLM agent)
action = agent.decision(
    balls=balls,
    my_targets=['1', '2'],
    table=table
)
```

## Shot Parameters

Both agents output the same action format:

```python
{
    'V0': float,      # Initial velocity (0.5 - 8.0 m/s)
    'phi': float,     # Horizontal angle (0 - 360 degrees)
    'theta': float,   # Vertical angle (0 - 90 degrees)
    'a': float,       # Horizontal offset (-0.5 - 0.5, ball radius fraction)
    'b': float        # Vertical offset (-0.5 - 0.5, ball radius fraction)
}
```

### Angle Convention

- **phi** (horizontal angle):
  - 0° = Right (positive X-axis)
  - 90° = Up (positive Y-axis)
  - 180° = Left (negative X-axis)
  - 270° = Down (negative Y-axis)

- **theta** (vertical angle):
  - 0° = Horizontal shot (most common)
  - 90° = Vertical shot (rare)

## API Configuration

### Environment Variables

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Qwen (Alibaba Cloud):
```bash
export OPENAI_API_KEY="your-dashscope-api-key"
```

### Supported Providers

1. **Qwen (Alibaba Cloud DashScope)**
   - Text model: `qwen-plus`, `qwen-turbo`, `qwen-max`
   - Vision model: `qwen-vl-max`, `qwen-vl-plus`, `qwen3-vl-flash`
   - Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`

2. **OpenAI**
   - Text model: `gpt-4`, `gpt-3.5-turbo`
   - Vision model: `gpt-4-vision-preview`, `gpt-4o`

3. **Claude (Anthropic)**
   - Text model: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`
   - Vision model: Same models (Claude 3 supports vision)

## Fallback Behavior

Both agents implement robust fallback:

1. **LLM/VLM call fails** → Random action
2. **Invalid JSON response** → Random action
3. **Missing parameters** → Random action
4. **Out-of-range parameters** → Clipped to valid range

This ensures the agent always returns a valid action.

## Text Description Format (LLM Agent)

The LLM agent converts game state to text like:

```
**Billiards Game State**

**Table:**
- Dimensions: 1.12m (width) × 2.24m (length)
- Coordinate system: X-axis (0 to 1.12m), Y-axis (0 to 2.24m)

**Pockets:**
- Left-Bottom corner: (0.000, 0.000)
- Left-Center side: (0.000, 1.120)
- Left-Top corner: (0.000, 2.240)
- Right-Bottom corner: (1.120, 0.000)
- Right-Center side: (1.120, 1.120)
- Right-Top corner: (1.120, 2.240)

**Cue Ball (white ball):**
- Position: (0.500, 0.500)

**My Target Balls (2 remaining):**
- Ball 1: (1.000, 0.560)
- Ball 2: (1.200, 0.800)

**Other Balls on Table (3):**
- Ball 8: (1.500, 0.560)
- Ball 9: (1.800, 0.700)
- Ball 10: (0.800, 1.200)

**Strategic Context:**
- You must hit one of your target balls: 1, 2
- Goal: Pocket your target balls into any of the 6 pockets
- Consider: ball positions, distances, angles to pockets, obstacles
```

## Visual Format (VLM Agent)

The VLM agent generates annotated images:

- **Green borders** = Your target balls
- **Orange borders** = Opponent's target balls
- **Red border** = Cue ball (white)
- **Purple border** = 8-ball
- **Black circles** = Pockets (6 total)
- **Green background** = Table surface

## Testing

Test the agents:

```bash
# Test LLM agent
cd agents/vlm_agents
python llmAgent.py

# Test VLM agent
python vlmAgent.py

# Test chat interface
python chat.py

# Test drawer
python drawer.py
```

## Performance Considerations

### LLM Agent
- **Faster**: Text-only, smaller models
- **Cheaper**: Lower API costs
- **Less accurate**: No visual understanding

### VLM Agent
- **Slower**: Image encoding + larger models
- **More expensive**: Higher API costs
- **More accurate**: Can see spatial relationships

## Integration with Evaluation

To use these agents in the evaluation framework:

```python
from agents.vlm_agents.llmAgent import LLMAgent
from agents.vlm_agents.vlmAgent import VLMAgent

# In your evaluation script
agent = LLMAgent(provider='qwen', model='qwen-plus')
# or
agent = VLMAgent(provider='qwen', model='qwen-vl-max')

# Use with PoolEnv
env = PoolEnv()
balls, my_targets, table = env.get_observation()
action = agent.decision(balls=balls, my_targets=my_targets, table=table)
env.take_shot(action)
```

## Troubleshooting

### API Key Issues
```
Error: API key not found
Solution: Set OPENAI_API_KEY environment variable
```

### Import Errors
```
Error: No module named 'openai'
Solution: pip install openai
```

### Timeout Issues
```
Error: API call timeout
Solution: VLM calls can be slow (3-10s). This is expected.
```

### Invalid JSON Response
```
Warning: No JSON found in response
Solution: Agent automatically falls back to random action
```

## Future Improvements

1. **Caching**: Cache similar game states to reduce API calls
2. **Fine-tuning**: Fine-tune models on billiards-specific data
3. **Hybrid approach**: Combine VLM guidance with search algorithms
4. **Multi-shot planning**: Plan multiple shots ahead
5. **Opponent modeling**: Predict opponent's strategy

## License

Part of the AI3603-Billiards project.
