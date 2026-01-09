# Quick Start Guide: LLM and VLM Agents

## 🚀 Quick Start (5 minutes)

### Step 1: Set API Key

```bash
# For Qwen (recommended, free tier available)
export OPENAI_API_KEY="your-dashscope-api-key"
```

Get your key at: https://dashscope.aliyun.com/

### Step 2: Install Dependencies

```bash
pip install openai pillow matplotlib numpy pooltool
```

### Step 3: Test the Agents

```bash
cd agents/vlm_agents

# Test LLM agent (text-only, faster)
python llmAgent.py

# Test VLM agent (vision-based, more accurate)
python vlmAgent.py

# Compare both agents
python example_usage.py --mode compare
```

## 📝 Basic Usage

### LLM Agent (Text-Only)

```python
from agents.vlm_agents import LLMAgent

# Create agent
agent = LLMAgent(provider='qwen', model='qwen-plus')

# Make decision
action = agent.decision(balls=balls, my_targets=['1', '2'], table=table)

# action = {'V0': 3.5, 'phi': 45.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0}
```

### VLM Agent (Vision-Based)

```python
from agents.vlm_agents import VLMAgent

# Create agent
agent = VLMAgent(provider='qwen', model='qwen-vl-max')

# Make decision (same interface!)
action = agent.decision(balls=balls, my_targets=['1', '2'], table=table)
```

## 🎮 Integration with Game

```python
from poolenv import PoolEnv
from agents.vlm_agents import LLMAgent

# Setup
env = PoolEnv()
env.reset()
agent = LLMAgent()

# Game loop
while not env.get_done():
    balls, my_targets, table = env.get_observation()
    action = agent.decision(balls, my_targets, table)
    result = env.take_shot(action)
```

## 🔧 Configuration Options

### LLM Agent

```python
agent = LLMAgent(
    provider='qwen',        # 'qwen', 'openai', 'claude'
    model='qwen-plus',      # Text model name
    api_key=None,           # Or set OPENAI_API_KEY env var
    base_url=None           # Optional custom API endpoint
)
```

### VLM Agent

```python
agent = VLMAgent(
    provider='qwen',        # 'qwen', 'openai', 'claude'
    model='qwen-vl-max',    # Vision model name
    api_key=None,           # Or set OPENAI_API_KEY env var
    base_url=None           # Optional custom API endpoint
)
```

## 📊 Model Recommendations

### For Speed (LLM Agent)
- **Fastest**: `qwen-turbo` (~1s response)
- **Balanced**: `qwen-plus` (~2s response)
- **Best**: `qwen-max` (~3s response)

### For Accuracy (VLM Agent)
- **Fastest**: `qwen3-vl-flash` (~3s response)
- **Balanced**: `qwen-vl-plus` (~5s response)
- **Best**: `qwen-vl-max` (~8s response)

## ❓ Troubleshooting

### "API key not found"
```bash
export OPENAI_API_KEY="your-key-here"
```

### "No module named 'openai'"
```bash
pip install openai
```

### "LLM failed to provide valid parameters"
- Agent automatically falls back to random action
- Check your API key and internet connection
- Verify you have API credits remaining

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Examples**: Run `python example_usage.py`

## 🎯 Key Features

✅ **Easy to Use**: Same interface for both agents  
✅ **Robust**: Always returns valid action (fallback to random)  
✅ **Flexible**: Support multiple providers (Qwen, OpenAI, Claude)  
✅ **Well-Tested**: Includes test scripts and examples  
✅ **Documented**: Comprehensive documentation included  

## 💡 Tips

1. **Start with LLM Agent**: Faster and cheaper for testing
2. **Use VLM for Competition**: Better spatial understanding
3. **Set API Key in Environment**: More secure than hardcoding
4. **Monitor API Usage**: Check your API dashboard for costs
5. **Test Locally First**: Use example scripts before full evaluation

## 🔗 Useful Links

- Qwen API: https://dashscope.aliyun.com/
- OpenAI API: https://platform.openai.com/
- Claude API: https://console.anthropic.com/
- Project Repository: (your repo URL)

---

**Need Help?** Check the full README.md or open an issue.


