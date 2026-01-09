"""
example_usage.py - Example usage of LLM and VLM agents

This script demonstrates how to use both agents in a simple game scenario.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pooltool as pt
from agents.vlm_agents.llmAgent import LLMAgent
from agents.vlm_agents.vlmAgent import VLMAgent


def create_test_scenario():
    """Create a simple test scenario"""
    table = pt.Table.default()
    
    # Create a simple game state
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '2': pt.Ball.create("2", xy=[0.8, 0.8]),
        '8': pt.Ball.create("8", xy=[1.5, 1.2]),
        '9': pt.Ball.create("9", xy=[1.8, 0.7]),
        '10': pt.Ball.create("10", xy=[1.6, 1.5]),
    }
    
    my_targets = ['1', '2']  # Solids
    
    return table, balls, my_targets


def test_llm_agent():
    """Test LLM agent"""
    print("=" * 60)
    print("Testing LLM Agent (Text-Only)")
    print("=" * 60)
    
    # Create scenario
    table, balls, my_targets = create_test_scenario()
    
    # Initialize agent
    print("\n[1] Initializing LLM Agent...")
    agent = LLMAgent(
        provider='qwen',
        model='qwen-plus',  # Text-only model
        api_key=None  # Will use OPENAI_API_KEY from environment
    )
    
    # Make decision
    print("\n[2] Making decision...")
    action = agent.decision(
        balls=balls,
        my_targets=my_targets,
        table=table
    )
    
    # Display result
    print("\n[3] LLM Agent Decision:")
    print(f"    V0 (velocity):     {action['V0']:.2f} m/s")
    print(f"    phi (h-angle):     {action['phi']:.1f}°")
    print(f"    theta (v-angle):   {action['theta']:.1f}°")
    print(f"    a (h-offset):      {action['a']:.3f}")
    print(f"    b (v-offset):      {action['b']:.3f}")
    
    print("\n✓ LLM Agent test completed\n")
    return action


def test_vlm_agent():
    """Test VLM agent"""
    print("=" * 60)
    print("Testing VLM Agent (Vision-Based)")
    print("=" * 60)
    
    # Create scenario
    table, balls, my_targets = create_test_scenario()
    
    # Initialize agent
    print("\n[1] Initializing VLM Agent...")
    agent = VLMAgent(
        provider='qwen',
        model='qwen-vl-max',  # Vision model
        api_key=None  # Will use OPENAI_API_KEY from environment
    )
    
    # Make decision
    print("\n[2] Drawing game state and making decision...")
    action = agent.decision(
        balls=balls,
        my_targets=my_targets,
        table=table
    )
    
    # Display result
    print("\n[3] VLM Agent Decision:")
    print(f"    V0 (velocity):     {action['V0']:.2f} m/s")
    print(f"    phi (h-angle):     {action['phi']:.1f}°")
    print(f"    theta (v-angle):   {action['theta']:.1f}°")
    print(f"    a (h-offset):      {action['a']:.3f}")
    print(f"    b (v-offset):      {action['b']:.3f}")
    
    print("\n✓ VLM Agent test completed\n")
    return action


def compare_agents():
    """Compare both agents on the same scenario"""
    print("=" * 60)
    print("Comparing LLM vs VLM Agents")
    print("=" * 60)
    
    # Create scenario
    table, balls, my_targets = create_test_scenario()
    
    print("\nScenario:")
    print(f"  Cue ball at: (0.50, 0.50)")
    print(f"  Target balls: {my_targets}")
    print(f"  Ball 1 at: (1.00, 0.56)")
    print(f"  Ball 2 at: (0.80, 0.80)")
    
    # Test LLM
    print("\n" + "-" * 60)
    llm_agent = LLMAgent(provider='qwen', model='qwen-plus')
    llm_action = llm_agent.decision(balls=balls, my_targets=my_targets, table=table)
    
    # Test VLM
    print("\n" + "-" * 60)
    vlm_agent = VLMAgent(provider='qwen', model='qwen-vl-max')
    vlm_action = vlm_agent.decision(balls=balls, my_targets=my_targets, table=table)
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)
    print(f"{'Parameter':<15} {'LLM Agent':<20} {'VLM Agent':<20}")
    print("-" * 60)
    print(f"{'V0 (m/s)':<15} {llm_action['V0']:<20.2f} {vlm_action['V0']:<20.2f}")
    print(f"{'phi (degrees)':<15} {llm_action['phi']:<20.1f} {vlm_action['phi']:<20.1f}")
    print(f"{'theta (degrees)':<15} {llm_action['theta']:<20.1f} {vlm_action['theta']:<20.1f}")
    print(f"{'a (offset)':<15} {llm_action['a']:<20.3f} {vlm_action['a']:<20.3f}")
    print(f"{'b (offset)':<15} {llm_action['b']:<20.3f} {vlm_action['b']:<20.3f}")
    print("=" * 60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LLM/VLM agents')
    parser.add_argument('--mode', choices=['llm', 'vlm', 'compare', 'all'], 
                       default='all', help='Which agent to test')
    args = parser.parse_args()
    
    try:
        if args.mode == 'llm':
            test_llm_agent()
        elif args.mode == 'vlm':
            test_vlm_agent()
        elif args.mode == 'compare':
            compare_agents()
        else:  # all
            test_llm_agent()
            print("\n" * 2)
            test_vlm_agent()
            print("\n" * 2)
            compare_agents()
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


