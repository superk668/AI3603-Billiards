"""
evaluate_global.py - Evaluation with Global Time Management

This script properly initializes and uses GlobalMCTSAgent with time management.

Usage:
    python evaluate_global.py
"""

from utils import set_random_seed
from poolenv import PoolEnv
from agents import BasicAgentPro, GlobalMCTSAgent

# Configuration
n_games = 40  # Number of games (can be 120 for full evaluation)
time_per_game = 180.0  # 3 minutes per game
use_random_seed = False

# Set random seed
set_random_seed(enable=use_random_seed, seed=42)

# Initialize environment
env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}

#  ============ IMPORTANT: Initialize Time Manager ============
# This MUST be done before creating agent instances
GlobalMCTSAgent.initialize_time_manager(
    n_games=n_games,
    time_per_game=time_per_game
)
# =============================================================

# Create agents
agent_a = BasicAgentPro()
agent_b = GlobalMCTSAgent()  # Will use time manager

players = [agent_a, agent_b]
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']

print("\n" + "=" * 80)
print("Global Time-Managed MCTS Evaluation")
print("=" * 80)
print(f"Agent A: {agent_a.__class__.__name__}")
print(f"Agent B: {agent_b.__class__.__name__}")
print(f"Games: {n_games}")
print(f"Time per game: {time_per_game}s")
print(f"Total time budget: {n_games * time_per_game}s ({n_games * time_per_game / 60:.1f} minutes)")
print("=" * 80)
print()

for i in range(n_games):
    print()
    print(f"{'='*80}")
    print(f"Game {i+1}/{n_games} - Starting")
    print(f"{'='*80}")
    
    # Notify time manager of game start
    GlobalMCTSAgent.start_game()
    
    env.reset(target_ball=target_ball_choice[i % 4])
    player_class = players[i % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    print(f"Player A (goes first): {player_class}")
    print(f"Target ball type: {ball_type}")
    print()
    
    shot_count = 0
    while True:
        player = env.get_curr_player()
        shot_count += 1
        print(f"\n[Shot {shot_count}] Player: {player}")
        
        obs = env.get_observation(player)
        
        # Assign agent based on player and game number
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done:
            if step_info.get('ENEMY_INTO_POCKET'):
                print(f"  → Opponent balls pocketed: {step_info['ENEMY_INTO_POCKET']}")
        
        if done:
            # Record results
            if info['winner'] == 'SAME':
                results['SAME'] += 1
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            
            print(f"\n{'='*80}")
            print(f"Game {i+1} Complete - Winner: {info['winner']}")
            print(f"Total shots: {shot_count}")
            print(f"Current score: A={results['AGENT_A_WIN']}, B={results['AGENT_B_WIN']}, Draw={results['SAME']}")
            print(f"{'='*80}")
            break
    
    # Notify time manager of game end
    GlobalMCTSAgent.end_game()

# Calculate final scores
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

# Print final results
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"\nAgent A ({agent_a.__class__.__name__}):")
print(f"  Wins: {results['AGENT_A_WIN']}")
print(f"  Score: {results['AGENT_A_SCORE']}")
print(f"  Win Rate: {results['AGENT_A_WIN']/n_games*100:.1f}%")

print(f"\nAgent B ({agent_b.__class__.__name__}):")
print(f"  Wins: {results['AGENT_B_WIN']}")
print(f"  Score: {results['AGENT_B_SCORE']}")
print(f"  Win Rate: {results['AGENT_B_WIN']/n_games*100:.1f}%")

print(f"\nDraws: {results['SAME']}")

# Print time utilization stats
if GlobalMCTSAgent._time_manager:
    stats = GlobalMCTSAgent._time_manager.get_stats()
    print(f"\nTime Management Statistics:")
    print(f"  Total time used: {stats['time_elapsed']:.1f}s ({stats['time_elapsed']/60:.1f} min)")
    print(f"  Total time budget: {n_games * time_per_game:.1f}s ({n_games * time_per_game/60:.1f} min)")
    print(f"  Utilization: {stats['utilization']*100:.1f}%")
    print(f"  Total decisions: {stats['decisions_made']}")
    print(f"  Avg decision time: {stats['avg_decision_time']:.2f}s")
    print(f"  Machine calibrated: {'Yes' if stats['calibrated'] else 'No'}")

print("\n" + "=" * 80)
print(f"Full results: {results}")
print("=" * 80)





