"""
test_mcts_agent.py - Quick test for MCTSAgent

Simple test to verify the MCTSAgent works correctly.
"""

from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgentPro, MCTSAgent

# Set seed for reproducibility
set_random_seed(enable=True, seed=42)

env = PoolEnv()
agent_a = BasicAgentPro()  # Baseline
agent_b = MCTSAgent(
    n_simulations=30,      # Reduced for faster testing
    n_noise_samples=3,     # Reduced for faster testing
    risk_aversion=0.5
)

# Single game test
print("=" * 60)
print("Testing MCTSAgent vs BasicAgentPro (1 game)")
print("=" * 60)

env.reset(target_ball='solid')

while True:
    player = env.get_curr_player()
    print(f"\n[Shot {env.hit_count}] Current player: {player}")
    
    obs = env.get_observation(player)
    
    if player == 'A':
        action = agent_a.decision(*obs)
    else:
        action = agent_b.decision(*obs)
    
    env.take_shot(action)
    
    done, info = env.get_done()
    if done:
        print("\n" + "=" * 60)
        print(f"Game over! Winner: {info['winner']}")
        print(f"Total shots: {info['hit_count']}")
        print("=" * 60)
        break

print("\n✓ Test completed successfully!")


