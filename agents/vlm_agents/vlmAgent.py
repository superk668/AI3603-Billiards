"""
vlmAgent.py - Vision-Language Model Agent for Billiards

This agent uses a VLM (default: Qwen3-vl-8b-instruct) to make decisions.
The environment draws a picture of the game state and feeds it to the VLM
along with supplementary information. The VLM outputs shot parameters.
If the VLM fails to provide valid parameters, falls back to random action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from typing import Dict, List, Optional
import pooltool as pt
from PIL import Image

from chat import VLMChat
from drawer import BilliardsDrawer


class VLMAgent:
    """Vision-Language Model agent for billiards"""
    
    def __init__(self, provider='qwen', model='qwen-vl-max', api_key=None, base_url=None):
        """
        Initialize VLM Agent
        
        Args:
            provider: 'openai', 'claude', 'qwen'
            model: Model name (default: 'qwen-vl-max' for vision)
            api_key: API key (if not provided, read from environment)
            base_url: API base URL
        """
        # Initialize VLM client (with vision)
        self.vlm_chat = VLMChat(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_vision=True  # Vision mode
        )
        
        # Initialize drawer
        self.drawer = BilliardsDrawer()
        
        print(f"[VLMAgent] Initialized with {provider}/{model} (vision-enabled)")
    
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
            print("[VLMAgent] Warning: Missing game state, using random action")
            return self._random_action()
        
        # Count remaining balls
        my_remaining = self._count_remaining_balls(balls, my_targets)
        
        # Determine enemy targets (opposite group)
        enemy_targets = self._determine_enemy_targets(balls, my_targets)
        enemy_remaining = self._count_remaining_balls(balls, enemy_targets)
        
        # Draw game state image
        try:
            image = self.drawer.draw_table_state(
                balls=balls,
                my_targets=my_targets,
                enemy_targets=enemy_targets,
                title=f"Game State - My: {my_remaining} vs Enemy: {enemy_remaining}",
                annotate=True,
                table=table
            )
        except Exception as e:
            print(f"[VLMAgent] Error drawing game state: {e}")
            return self._random_action()
        
        # Generate supplementary text information
        supplementary_info = self._generate_supplementary_info(
            balls, my_targets, my_remaining, enemy_remaining, table
        )
        
        # Get shot parameters from VLM
        shot_params = self._get_shot_from_vlm(image, supplementary_info)
        
        # If VLM fails, fall back to random
        if shot_params is None:
            print("[VLMAgent] VLM failed to provide valid parameters, using random action")
            return self._random_action()
        
        # Extract only the shot parameters
        action = {
            'V0': shot_params['V0'],
            'phi': shot_params['phi'],
            'theta': shot_params['theta'],
            'a': shot_params['a'],
            'b': shot_params['b']
        }
        
        print(f"[VLMAgent] Decision: V0={action['V0']:.2f}, phi={action['phi']:.1f}°, "
              f"theta={action['theta']:.1f}°, a={action['a']:.2f}, b={action['b']:.2f}")
        
        return action
    
    def _get_shot_from_vlm(self, image: Image.Image, supplementary_info: str) -> Optional[Dict]:
        """
        Get shot parameters from VLM based on image and text
        
        Args:
            image: PIL Image of game state
            supplementary_info: Text description with additional context
            
        Returns:
            Dict with shot parameters or None if failed
        """
        # Build prompt
        prompt = self._build_vlm_shot_prompt(supplementary_info)
        
        # Call VLM with image
        if self.vlm_chat.provider in ['openai', 'qwen']:
            response = self.vlm_chat._call_openai(image, prompt)
        elif self.vlm_chat.provider == 'claude':
            response = self.vlm_chat._call_claude(image, prompt)
        else:
            return None
        
        # Parse response
        shot_params = self.vlm_chat._parse_shot_response(response)
        
        return shot_params
    
    def _build_vlm_shot_prompt(self, supplementary_info: str) -> str:
        """Build prompt for VLM to generate shot parameters"""
        
        prompt = f"""You are an expert billiards player. Analyze the game state image and the information below to determine the best shot.

{supplementary_info}

**Your Task:**
Based on the image and the information above, choose shot parameters to maximize your chance of pocketing your target balls (marked with GREEN borders in the image).

**Shot Parameters to Output:**
- V0: Initial velocity in m/s (range: 0.5 to 8.0)
- phi: Horizontal angle in degrees (range: 0 to 360)
  * 0° points to the right (positive X-axis)
  * 90° points upward (positive Y-axis)
  * 180° points to the left
  * 270° points downward
- theta: Vertical angle in degrees (range: 0 to 90, usually 0° for standard shots)
- a: Horizontal impact offset as fraction of ball radius (range: -0.5 to 0.5, usually 0)
- b: Vertical impact offset as fraction of ball radius (range: -0.5 to 0.5, usually 0)

**Visual Cues in Image:**
- RED border = Cue ball (white ball you control)
- GREEN borders = Your target balls
- ORANGE borders = Opponent's target balls
- PURPLE border = 8-ball (pocket last, after all your targets)
- Black circles = Pockets (6 total: 4 corners + 2 side pockets)

**Strategy Tips:**
- Aim to hit your target ball toward the nearest pocket
- Consider the angle between cue ball, target ball, and pocket
- Use appropriate power (V0): gentle for close shots, stronger for distant shots
- For simple shots, keep theta=0, a=0, b=0

**Respond in JSON format ONLY (no additional text):**
{{
    "V0": <float>,
    "phi": <float>,
    "theta": <float>,
    "a": <float>,
    "b": <float>,
    "reasoning": "<brief explanation of your shot choice>"
}}

Provide ONLY the JSON response."""
        
        return prompt
    
    def _generate_supplementary_info(self, balls: Dict, my_targets: List[str], 
                                     my_remaining: int, enemy_remaining: int, table) -> str:
        """Generate supplementary text information for VLM"""
        
        # Get table dimensions
        table_width = getattr(table, 'w', 1.12)
        table_length = getattr(table, 'l', 2.24)
        
        # Get cue ball position
        cue_ball = balls.get('cue')
        if cue_ball:
            cue_pos = self._get_ball_position(cue_ball)
            cue_info = f"Cue ball position: ({cue_pos[0]:.3f}, {cue_pos[1]:.3f})"
        else:
            cue_info = "Cue ball position: unknown"
        
        # Determine game phase
        if my_remaining <= 2:
            game_phase = "End game"
        elif my_remaining <= 4:
            game_phase = "Mid game"
        else:
            game_phase = "Early game"
        
        # Determine situation
        if my_remaining < enemy_remaining:
            situation = "You are LEADING (fewer balls remaining)"
        elif my_remaining == enemy_remaining:
            situation = "Game is EVEN"
        else:
            situation = "You are BEHIND (more balls remaining)"
        
        # List active target balls
        active_targets = []
        for ball_id in my_targets:
            if ball_id in balls:
                ball = balls[ball_id]
                if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                    if ball.state.s != 4:  # Not pocketed
                        pos = self._get_ball_position(ball)
                        active_targets.append(f"Ball {ball_id} at ({pos[0]:.3f}, {pos[1]:.3f})")
        
        info = f"""**Game Information:**
- Table size: {table_width:.2f}m × {table_length:.2f}m
- {cue_info}
- Game phase: {game_phase}
- Situation: {situation}
- Your remaining balls: {my_remaining}
- Opponent's remaining balls: {enemy_remaining}

**Your Active Target Balls:**
{chr(10).join(['- ' + t for t in active_targets]) if active_targets else '- None (aim for Ball 8)'}

**Objective:**
Pocket one of your target balls (green borders) by hitting it with the cue ball (red border) into any pocket.
"""
        
        return info
    
    def _count_remaining_balls(self, balls: Dict, targets: List[str]) -> int:
        """Count how many target balls are still on the table"""
        count = 0
        for ball_id in targets:
            if ball_id in balls:
                ball = balls[ball_id]
                if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                    if ball.state.s != 4:  # Not pocketed
                        count += 1
                else:
                    count += 1
        return count
    
    def _determine_enemy_targets(self, balls: Dict, my_targets: List[str]) -> List[str]:
        """Determine enemy target balls (opposite group)"""
        # If my targets include '8', enemy has none
        if '8' in my_targets:
            return []
        
        # Determine if I have solids (1-7) or stripes (9-15)
        solids = ['1', '2', '3', '4', '5', '6', '7']
        stripes = ['9', '10', '11', '12', '13', '14', '15']
        
        # Check which group I have
        has_solid = any(t in solids for t in my_targets)
        has_stripe = any(t in stripes for t in my_targets)
        
        if has_solid:
            return stripes
        elif has_stripe:
            return solids
        else:
            # Unknown, return empty
            return []
    
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
            print(f"[VLMAgent] Error getting ball position: {e}")
            return (0, 0)
    
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


def test_vlm_agent():
    """Test VLM agent"""
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
    agent = VLMAgent(provider='qwen', model='qwen-vl-max')
    
    # Make decision
    action = agent.decision(balls=balls, my_targets=['1'], table=table)
    
    print("\n[Test] VLM Agent Action:")
    print(f"  V0={action['V0']:.2f} m/s")
    print(f"  phi={action['phi']:.1f}°")
    print(f"  theta={action['theta']:.1f}°")
    print(f"  a={action['a']:.3f}, b={action['b']:.3f}")


if __name__ == "__main__":
    test_vlm_agent()


