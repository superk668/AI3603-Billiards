"""
llmAgent.py - Pure Text LLM Agent for Billiards

This agent uses a text-only LLM (default: Qwen3-8b) to make decisions.
The environment state is converted to a text description and fed to the LLM.
The LLM outputs shot parameters (V0, phi, theta, a, b).
If the LLM fails to provide valid parameters, falls back to random action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from typing import Dict, List, Optional
import pooltool as pt

from chat import VLMChat


class LLMAgent:
    """Pure text LLM agent for billiards"""
    
    def __init__(self, provider='qwen', model='qwen-plus', api_key=None, base_url=None):
        """
        Initialize LLM Agent
        
        Args:
            provider: 'openai', 'claude', 'qwen'
            model: Model name (default: 'qwen-plus' for text-only)
            api_key: API key (if not provided, read from environment)
            base_url: API base URL
        """
        # Initialize text-only LLM client
        self.llm_chat = VLMChat(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_vision=False  # Text-only mode
        )
        
        print(f"[LLMAgent] Initialized with {provider}/{model} (text-only)")
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        Make a decision based on current game state
        
        Args:
            balls: Dict of ball objects {ball_id: Ball}
            my_targets: List of target ball IDs for this player
            table: Table object
            
        Returns:
            Dict with keys: 'V0', 'phi', 'theta', 'a', 'b'
        """
        if balls is None or my_targets is None or table is None:
            print("[LLMAgent] Warning: Missing game state, using random action")
            return self._random_action()
        
        # Generate text description of game state
        text_description = self._generate_text_description(balls, my_targets, table)
        
        # Get shot parameters from LLM
        shot_params = self.llm_chat.get_shot_parameters(text_description)
        
        # If LLM fails, fall back to random
        if shot_params is None:
            print("[LLMAgent] LLM failed to provide valid parameters, using random action")
            return self._random_action()
        
        # Extract only the shot parameters (remove 'reasoning' if present)
        action = {
            'V0': shot_params['V0'],
            'phi': shot_params['phi'],
            'theta': shot_params['theta'],
            'a': shot_params['a'],
            'b': shot_params['b']
        }
        
        print(f"[LLMAgent] Decision: V0={action['V0']:.2f}, phi={action['phi']:.1f}°, "
              f"theta={action['theta']:.1f}°, a={action['a']:.2f}, b={action['b']:.2f}")
        
        return action
    
    def _generate_text_description(self, balls: Dict, my_targets: List[str], table) -> str:
        """
        Generate a text description of the current game state
        
        Args:
            balls: Dict of ball objects
            my_targets: List of target ball IDs
            table: Table object
            
        Returns:
            String description of game state
        """
        # Get table dimensions
        table_width = getattr(table, 'w', 1.12)  # meters
        table_length = getattr(table, 'l', 2.24)  # meters
        
        # Get cue ball position
        cue_ball = balls.get('cue')
        if cue_ball is None:
            cue_pos = (0, 0)
        else:
            cue_pos = self._get_ball_position(cue_ball)
        
        # Collect active balls (not pocketed)
        active_balls = {}
        for ball_id, ball in balls.items():
            if ball_id == 'cue':
                continue
            # Check if pocketed
            is_pocketed = False
            if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                is_pocketed = (ball.state.s == 4)
            
            if not is_pocketed:
                pos = self._get_ball_position(ball)
                active_balls[ball_id] = pos
        
        # Separate my targets from other balls
        my_active_targets = []
        other_balls = []
        
        for ball_id, pos in active_balls.items():
            if ball_id in my_targets:
                my_active_targets.append((ball_id, pos))
            else:
                other_balls.append((ball_id, pos))
        
        # Get pocket positions
        pocket_info = self._get_pocket_descriptions(table)
        
        # Build description
        description = f"""**Billiards Game State**

**Table:**
- Dimensions: {table_width:.2f}m (width) × {table_length:.2f}m (length)
- Coordinate system: X-axis (0 to {table_width:.2f}m), Y-axis (0 to {table_length:.2f}m)

**Pockets:**
{pocket_info}

**Cue Ball (white ball):**
- Position: ({cue_pos[0]:.3f}, {cue_pos[1]:.3f})

**My Target Balls ({len(my_active_targets)} remaining):**
"""
        
        if my_active_targets:
            for ball_id, pos in my_active_targets:
                description += f"- Ball {ball_id}: ({pos[0]:.3f}, {pos[1]:.3f})\n"
        else:
            description += "- None (all cleared)\n"
        
        description += f"\n**Other Balls on Table ({len(other_balls)}):**\n"
        
        if other_balls:
            for ball_id, pos in other_balls:
                description += f"- Ball {ball_id}: ({pos[0]:.3f}, {pos[1]:.3f})\n"
        else:
            description += "- None\n"
        
        # Add strategic context
        description += f"""
**Strategic Context:**
- You must hit one of your target balls: {', '.join([b[0] for b in my_active_targets]) if my_active_targets else 'Ball 8 (if all cleared)'}
- Goal: Pocket your target balls into any of the 6 pockets
- Consider: ball positions, distances, angles to pockets, and potential obstacles

**Physics Notes:**
- Higher V0 = more power (but less control)
- phi angle: 0° points right (+X), 90° points up (+Y), 180° points left, 270° points down
- theta: usually 0° for standard shots
- a, b: impact offset for spin (usually 0 for simple shots)
"""
        
        return description
    
    def _get_ball_position(self, ball) -> tuple:
        """Get 2D position of a ball"""
        try:
            if hasattr(ball, 'state') and hasattr(ball.state, 'rvw'):
                pos = ball.state.rvw[0][:2]
                return (float(pos[0]), float(pos[1]))
            elif hasattr(ball, 'xyz'):
                return (float(ball.xyz[0]), float(ball.xyz[1]))
            elif hasattr(ball, 'pos'):
                return (float(ball.pos[0]), float(ball.pos[1]))
            else:
                return (0, 0)
        except Exception as e:
            print(f"[LLMAgent] Error getting ball position: {e}")
            return (0, 0)
    
    def _get_pocket_descriptions(self, table) -> str:
        """Get descriptions of pocket positions"""
        if not hasattr(table, 'pockets'):
            return "- 6 standard pockets (corners and middle sides)\n"
        
        pocket_desc = ""
        pocket_names = {
            'lb': 'Left-Bottom corner',
            'lc': 'Left-Center side',
            'lt': 'Left-Top corner',
            'rb': 'Right-Bottom corner',
            'rc': 'Right-Center side',
            'rt': 'Right-Top corner'
        }
        
        for pocket_id, pocket in table.pockets.items():
            name = pocket_names.get(pocket_id, pocket_id)
            pos = pocket.center[:2]
            pocket_desc += f"- {name}: ({pos[0]:.3f}, {pos[1]:.3f})\n"
        
        return pocket_desc
    
    def _random_action(self) -> Dict:
        """Generate random shot action (fallback)"""
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
        return action


def test_llm_agent():
    """Test LLM agent"""
    import pooltool as pt
    
    # Create test scenario
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '8': pt.Ball.create("8", xy=[1.5, 0.56]),
        '9': pt.Ball.create("9", xy=[1.8, 0.7]),
    }
    
    # Create agent
    agent = LLMAgent(provider='qwen', model='qwen-plus')
    
    # Make decision
    action = agent.decision(balls=balls, my_targets=['1'], table=table)
    
    print("\n[Test] LLM Agent Action:")
    print(f"  V0={action['V0']:.2f} m/s")
    print(f"  phi={action['phi']:.1f}°")
    print(f"  theta={action['theta']:.1f}°")
    print(f"  a={action['a']:.3f}, b={action['b']:.3f}")


if __name__ == "__main__":
    test_llm_agent()


