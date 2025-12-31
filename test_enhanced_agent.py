#!/usr/bin/env python3
"""
Quick test script to compare EnhancedMCTSAgent with BasicAgentPro

Usage:
    python test_enhanced_agent.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.basic_agent_pro import BasicAgentPro
from agents.enhanced_mcts_agent import EnhancedMCTSAgent
from poolenv import PoolEnv
import time


def test_single_game(agent1, agent2, agent1_name, agent2_name, verbose=True):
    """
    Run a single game between two agents
    
    Returns:
        winner: 0 (agent1), 1 (agent2), or None (draw/error)
    """
    env = PoolEnv()
    state = env.reset()
    
    agents = [agent1, agent2]
    current_player = 0
    max_steps = 200
    step_count = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game: {agent1_name} vs {agent2_name}")
        print(f"{'='*60}")
    
    while not state['done'] and step_count < max_steps:
        step_count += 1
        current_agent = agents[current_player]
        
        # Get agent decision
        start_time = time.time()
        action = current_agent.decision(
            balls=state['balls'],
            my_targets=state['player_targets'][current_player],
            table=state['table']
        )
        decision_time = time.time() - start_time
        
        if verbose:
            print(f"\nStep {step_count} - Player {current_player} ({[agent1_name, agent2_name][current_player]}):")
            print(f"  Decision time: {decision_time:.2f}s")
            print(f"  Action: V0={action['V0']:.2f}, phi={action['phi']:.1f}°")
        
        # Take action
        state, reward, done, info = env.step(action)
        
        if verbose:
            print(f"  Reward: {reward:.1f}")
            if 'error' in info:
                print(f"  Error: {info['error']}")
        
        # Check if game ended
        if state['done']:
            winner = state.get('winner')
            if verbose:
                if winner is not None:
                    print(f"\n{'='*60}")
                    print(f"Game Over! Winner: Player {winner} ({[agent1_name, agent2_name][winner]})")
                    print(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"Game Over! Draw or error")
                    print(f"{'='*60}")
            return winner
        
        # Switch player if needed
        next_player = state.get('current_player', current_player)
        if next_player != current_player:
            if verbose:
                print(f"  -> Player switch")
        current_player = next_player
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game ended due to max steps ({max_steps})")
        print(f"{'='*60}")
    
    return None


def run_comparison(n_games=5):
    """
    Run a comparison between EnhancedMCTSAgent and BasicAgentPro
    
    Args:
        n_games: Number of games to play
    """
    print("\n" + "="*60)
    print("ENHANCED MCTS AGENT vs BASIC AGENT PRO")
    print("="*60)
    
    # Initialize agents
    enhanced_agent = EnhancedMCTSAgent(
        n_simulations=50,
        base_c_puct=1.414,
        refinement_threshold=0.6,
        position_weight=0.3
    )
    
    basic_agent = BasicAgentPro(
        n_simulations=50,
        c_puct=1.414
    )
    
    results = {
        'enhanced_wins': 0,
        'basic_wins': 0,
        'draws': 0,
        'enhanced_as_p0': 0,
        'enhanced_as_p1': 0,
        'basic_as_p0': 0,
        'basic_as_p1': 0,
    }
    
    for game_num in range(n_games):
        print(f"\n\n{'#'*60}")
        print(f"GAME {game_num + 1}/{n_games}")
        print(f"{'#'*60}")
        
        # Alternate who goes first
        if game_num % 2 == 0:
            # Enhanced agent as player 0
            winner = test_single_game(
                enhanced_agent, basic_agent,
                "EnhancedMCTS", "BasicAgentPro",
                verbose=True
            )
            if winner == 0:
                results['enhanced_wins'] += 1
                results['enhanced_as_p0'] += 1
            elif winner == 1:
                results['basic_wins'] += 1
                results['basic_as_p0'] += 1
            else:
                results['draws'] += 1
        else:
            # Basic agent as player 0
            winner = test_single_game(
                basic_agent, enhanced_agent,
                "BasicAgentPro", "EnhancedMCTS",
                verbose=True
            )
            if winner == 0:
                results['basic_wins'] += 1
                results['basic_as_p1'] += 1
            elif winner == 1:
                results['enhanced_wins'] += 1
                results['enhanced_as_p1'] += 1
            else:
                results['draws'] += 1
    
    # Print final results
    print("\n\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total games: {n_games}")
    print(f"\nEnhanced MCTS Agent: {results['enhanced_wins']} wins ({results['enhanced_wins']/n_games*100:.1f}%)")
    print(f"  - As Player 0: {results['enhanced_as_p0']} wins")
    print(f"  - As Player 1: {results['enhanced_as_p1']} wins")
    print(f"\nBasic Agent Pro: {results['basic_wins']} wins ({results['basic_wins']/n_games*100:.1f}%)")
    print(f"  - As Player 0: {results['basic_as_p0']} wins")
    print(f"  - As Player 1: {results['basic_as_p1']} wins")
    print(f"\nDraws: {results['draws']}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EnhancedMCTSAgent vs BasicAgentPro")
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    args = parser.parse_args()
    
    try:
        results = run_comparison(n_games=args.games)
        
        # Exit with appropriate code
        if results['enhanced_wins'] > results['basic_wins']:
            print("\n✓ EnhancedMCTSAgent outperformed BasicAgentPro!")
            sys.exit(0)
        elif results['enhanced_wins'] < results['basic_wins']:
            print("\n✗ BasicAgentPro performed better")
            sys.exit(1)
        else:
            print("\n= Tie between agents")
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

