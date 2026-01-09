"""
VlmAssistedAgent.py - VLM-Guided Search Agent for Billiards

This agent combines VLM strategic guidance with integrated search algorithms:
1. VLM analyzes the game state and outputs:
   - promising_targets: 3 balls most likely to yield good results
   - risk: 0-1, how risky the current situation is
   - budget: 0-1, how complex the game is (affects search depth)

2. These parameters guide the integrated search:
   - Promising targets get higher priority in candidate generation
   - Risk adjusts the risk-aversion parameter (lower risk = more conservative)
   - Budget controls search depth (higher budget = more simulations)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
import signal
import random
import math
from typing import Dict, List, Optional, Set, Tuple
import pooltool as pt

from chat import VLMChat
from drawer import BilliardsDrawer


# ============ Timeout Protection ============
class SimulationTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SimulationTimeoutError("Physics simulation timeout")


def simulate_with_timeout(shot, timeout=3):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        signal.alarm(0)
        return False
    except Exception:
        signal.alarm(0)
        return False
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ Reward Analysis ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """Analyze shot result and calculate reward score"""
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True

    score = 0
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500

    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10

    return score


# ============ Geometry Utilities ============
def ball_pos(balls, ball_id: str) -> Optional[np.ndarray]:
    if ball_id not in balls:
        return None
    ball = balls[ball_id]
    if hasattr(ball, 'state') and ball.state.s == 4:
        return None
    return np.array([ball.state.rvw[0][0], ball.state.rvw[0][1]], dtype=float)


def wrap_angle(angle_deg: float) -> float:
    return float(angle_deg % 360)


def segment_distance_point(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def is_line_blocked(a: np.ndarray, b: np.ndarray, balls, ball_radius: float, ignore_ids: Set[str] = frozenset()) -> bool:
    if a is None or b is None:
        return True
    for bid in balls.keys():
        if bid in ignore_ids:
            continue
        pos = ball_pos(balls, bid)
        if pos is None:
            continue
        if segment_distance_point(pos, a, b) < (2.0 * ball_radius * 0.98):
            return True
    return False


def cut_angle_deg(cue_xy: np.ndarray, obj_xy: np.ndarray, pocket_xy: np.ndarray) -> float:
    v1 = obj_xy - cue_xy
    v2 = pocket_xy - obj_xy
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    u1 = v1 / n1
    u2 = v2 / n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def nearest_pocket_distance(pt_xy: np.ndarray, pocket_centers: List[np.ndarray]) -> float:
    if pt_xy is None or not pocket_centers:
        return float('inf')
    return float(min(np.linalg.norm(pt_xy - pc) for pc in pocket_centers))


def ray_to_pocket_risk(cue_xy: np.ndarray, phi_deg: float, pocket_centers: List[np.ndarray]) -> float:
    if cue_xy is None or not pocket_centers:
        return 0.0
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))], dtype=float)
    threshold = 0.10
    risk = 0.0
    for pc in pocket_centers:
        w = pc - cue_xy
        proj = float(np.dot(w, u))
        if proj <= 0:
            continue
        closest = float(np.linalg.norm(w - proj * u))
        if closest < threshold:
            risk = max(risk, (threshold - closest) / max(1e-6, threshold))
    return float(np.clip(risk, 0.0, 1.0))


class VLMAssistedAgent:
    """VLM-Guided Search Agent"""
    
    def __init__(self, provider='qwen', model='qwen-vl-max', api_key=None, 
                 base_url=None, use_vlm=True, vlm_frequency='always'):
        """
        Initialize VLM-Assisted Agent
        
        Args:
            provider: 'openai', 'claude', 'qwen'
            model: VLM model name
            api_key: API key
            base_url: API base URL
            use_vlm: Whether to use VLM (if False, acts as pure search agent)
            vlm_frequency: 'always', 'first_n', or 'adaptive'
        """
        # Search agent parameters
        self.ball_radius = 0.028575
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005,
        }
        
        # Base search parameters (will be adjusted by VLM)
        self.base_n_simulations = int(os.environ.get('BILLIARDS_MAX_SIMULATIONS', '180'))
        self.base_risk_lambda = float(os.environ.get('BILLIARDS_RISK_LAMBDA', '0.3'))
        
        # VLM configuration
        self.use_vlm = use_vlm
        self.vlm_frequency = vlm_frequency
        
        # Initialize VLM components
        if self.use_vlm:
            self.vlm_chat = VLMChat(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                use_vision=True
            )
            self.drawer = BilliardsDrawer()
            print(f"[VLMAssistedAgent] VLM guidance enabled: {provider}/{model}")
        else:
            self.vlm_chat = None
            self.drawer = None
            print("[VLMAssistedAgent] Running in pure search mode (no VLM)")
        
        # VLM guidance tracking
        self.decision_count = 0
        self.last_vlm_guidance = None
        self.vlm_call_count = 0
        self.vlm_first_n_limit = 10
        
        # Default values when VLM not used
        self.default_risk = 0.5
        self.default_budget = 0.5
    
    def _random_action(self):
        """Generate random shot action"""
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
    
    def _calc_angle_degrees(self, v):
        return float(math.degrees(math.atan2(v[1], v[0])) % 360)
    
    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket < 1e-6:
            return 0.0, 0.0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, float(dist_cue_to_ghost)
    
    def _should_call_vlm(self, balls, my_targets) -> bool:
        """Determine if VLM should be called for this decision"""
        if not self.use_vlm or self.vlm_chat is None:
            return False
        
        if self.vlm_frequency == 'always':
            return True
        elif self.vlm_frequency == 'first_n':
            return self.decision_count < self.vlm_first_n_limit
        elif self.vlm_frequency == 'adaptive':
            remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            return self.decision_count == 0 or len(remaining) <= 3
        else:
            return True
    
    def _get_vlm_guidance(self, balls, my_targets, table) -> Dict:
        """Get strategic guidance from VLM"""
        try:
            # Count remaining balls
            my_remaining = sum(1 for bid in my_targets if bid in balls and balls[bid].state.s != 4)
            enemy_targets = self._determine_enemy_targets(balls, my_targets)
            enemy_remaining = sum(1 for bid in enemy_targets if bid in balls and balls[bid].state.s != 4)
            
            # Draw game state
            image = self.drawer.draw_table_state(
                balls=balls,
                my_targets=my_targets,
                enemy_targets=enemy_targets,
                title=f"VLM Strategy Analysis - My: {my_remaining} vs Enemy: {enemy_remaining}",
                annotate=True,
                table=table
            )
            
            # Build VLM prompt
            prompt = self._build_vlm_guidance_prompt(balls, my_targets, my_remaining, enemy_remaining, table)
            
            # Call VLM
            if self.vlm_chat.provider in ['openai', 'qwen']:
                response = self.vlm_chat._call_openai(image, prompt)
            elif self.vlm_chat.provider == 'claude':
                response = self.vlm_chat._call_claude(image, prompt)
            else:
                return self._default_guidance(my_targets, balls)
            
            # Parse response
            guidance = self._parse_vlm_guidance(response, my_targets, balls)
            
            self.vlm_call_count += 1
            print(f"[VLMAssistedAgent] VLM Guidance #{self.vlm_call_count}:")
            print(f"  Promising targets: {guidance['promising_targets']}")
            print(f"  Risk: {guidance['risk']:.2f}")
            print(f"  Budget: {guidance['budget']:.2f}")
            
            return guidance
            
        except Exception as e:
            print(f"[VLMAssistedAgent] VLM guidance failed: {e}")
            return self._default_guidance(my_targets, balls)
    
    def _build_vlm_guidance_prompt(self, balls, my_targets, my_remaining, enemy_remaining, table) -> str:
        """Build prompt for VLM strategic guidance"""
        cue_ball = balls.get('cue')
        if cue_ball:
            cue_pos = ball_pos(balls, 'cue')
            cue_info = f"Cue ball at ({cue_pos[0]:.2f}, {cue_pos[1]:.2f})" if cue_pos is not None else "Cue ball position unknown"
        else:
            cue_info = "Cue ball not found"
        
        if my_remaining <= 2:
            game_phase = "End game"
        elif my_remaining <= 4:
            game_phase = "Mid game"
        else:
            game_phase = "Early game"
        
        if my_remaining < enemy_remaining:
            situation = "LEADING (fewer balls remaining)"
        elif my_remaining == enemy_remaining:
            situation = "EVEN"
        else:
            situation = "BEHIND (more balls remaining)"
        
        active_targets = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        
        prompt = f"""You are an expert billiards strategist. Analyze the game state image and provide strategic guidance.

**Current Game State:**
- Game phase: {game_phase}
- Situation: {situation}
- My remaining balls: {my_remaining} (targets: {', '.join(active_targets) if active_targets else 'None'})
- Opponent's remaining balls: {enemy_remaining}
- {cue_info}

**Your Task:**
Provide strategic guidance to help the search algorithm make better decisions.

**Output THREE key parameters:**

1. **promising_targets** (array of 3 ball IDs):
   - Identify the 3 most promising balls to target
   - Consider: distance from cue ball, clear paths to pockets, positioning for next shot
   - Prioritize balls that are easiest to pocket or offer strategic advantage
   - If fewer than 3 targets available, list what's available
   - Example: ["1", "3", "5"]

2. **risk** (float 0.0 to 1.0):
   - 0.0 = Very safe situation (e.g., leading by many balls, easy shots available)
   - 0.5 = Moderate risk (even game, some difficult shots)
   - 1.0 = High risk (behind in game, difficult table position, need aggressive play)
   - Consider: current score, ball positions, shot difficulty

3. **budget** (float 0.0 to 1.0):
   - How much computational effort should be spent on this decision?
   - 0.0 = Simple situation, quick decision (obvious shot, easy table)
   - 0.5 = Moderate complexity (several good options, need some analysis)
   - 1.0 = Very complex (cluttered table, difficult shots, critical moment)
   - Consider: number of balls on table, clustering, shot difficulty, game importance

**Visual Reference:**
- GREEN borders = My target balls
- ORANGE borders = Opponent's balls
- RED border = Cue ball (white)
- PURPLE border = 8-ball
- Black circles = Pockets

**Respond in JSON format ONLY:**
{{
    "promising_targets": ["<ball_id>", "<ball_id>", "<ball_id>"],
    "risk": <0.0-1.0>,
    "budget": <0.0-1.0>,
    "reasoning": "<brief explanation of your strategic assessment>"
}}

Provide ONLY the JSON response, no additional text."""
        
        return prompt
    
    def _parse_vlm_guidance(self, response: str, my_targets: List[str], balls) -> Dict:
        """Parse VLM response into guidance parameters"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                guidance = json.loads(json_str)
                
                promising = guidance.get('promising_targets', [])
                if not isinstance(promising, list):
                    promising = []
                
                active_targets = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
                promising = [bid for bid in promising if bid in active_targets]
                
                if len(promising) < 3:
                    remaining = [bid for bid in active_targets if bid not in promising]
                    promising.extend(remaining[:3 - len(promising)])
                
                promising = promising[:3]
                
                risk = float(guidance.get('risk', self.default_risk))
                risk = max(0.0, min(1.0, risk))
                
                budget = float(guidance.get('budget', self.default_budget))
                budget = max(0.0, min(1.0, budget))
                
                return {
                    'promising_targets': promising,
                    'risk': risk,
                    'budget': budget,
                    'reasoning': guidance.get('reasoning', 'VLM guidance')
                }
            else:
                print("[VLMAssistedAgent] No JSON found in VLM response")
                return self._default_guidance(my_targets, balls)
                
        except Exception as e:
            print(f"[VLMAssistedAgent] Failed to parse VLM guidance: {e}")
            return self._default_guidance(my_targets, balls)
    
    def _default_guidance(self, my_targets: List[str], balls) -> Dict:
        """Default guidance when VLM fails or is not used"""
        active_targets = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        promising = active_targets[:3] if len(active_targets) >= 3 else active_targets
        
        return {
            'promising_targets': promising,
            'risk': self.default_risk,
            'budget': self.default_budget,
            'reasoning': 'Default guidance (no VLM)'
        }
    
    def _determine_enemy_targets(self, balls, my_targets) -> List[str]:
        """Determine enemy target balls"""
        if '8' in my_targets:
            return []
        
        solids = ['1', '2', '3', '4', '5', '6', '7']
        stripes = ['9', '10', '11', '12', '13', '14', '15']
        
        has_solid = any(t in solids for t in my_targets)
        has_stripe = any(t in stripes for t in my_targets)
        
        if has_solid:
            return stripes
        elif has_stripe:
            return solids
        else:
            return []
    
    def generate_candidates(self, balls, my_targets, table, promising_targets=None):
        """Generate candidate actions with focus on promising targets"""
        actions: List[dict] = []

        cue_xy = ball_pos(balls, 'cue')
        if cue_xy is None:
            return [self._random_action()]

        target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']

        pocket_centers = [np.array([p.center[0], p.center[1]], dtype=float) for p in table.pockets.values()]

        T = 2.5 * self.ball_radius
        thickness_ts = np.array([-T, -0.75*T, -0.5*T, -0.25*T, 0.0, 0.25*T, 0.5*T, 0.75*T, T], dtype=float)

        # Separate targets: promising vs others
        if promising_targets:
            priority_targets = [t for t in target_ids if t in promising_targets]
            other_targets = [t for t in target_ids if t not in promising_targets]
        else:
            priority_targets = target_ids
            other_targets = []

        # Generate candidates for priority targets first
        for tid in priority_targets:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue

            for pocket_xy in pocket_centers:
                phi_ghost, dist_cue_to_ghost = self._get_ghost_ball_target(cue_xy, obj_xy, pocket_xy)
                if dist_cue_to_ghost < 1e-6:
                    continue

                v_base = float(np.clip(1.5 + dist_cue_to_ghost * 1.5, 1.0, 6.0))
                speeds = [
                    float(np.clip(v_base * 0.7, 1.0, 8.0)),
                    float(np.clip(v_base * 0.9, 1.0, 8.0)),
                    float(np.clip(v_base, 1.0, 8.0)),
                    float(np.clip(v_base * 1.2, 1.0, 8.0)),
                ]

                for t in thickness_ts:
                    delta_phi = float(np.degrees(np.arctan2(t, dist_cue_to_ghost)))
                    phi = wrap_angle(phi_ghost + delta_phi)
                    for V0 in speeds:
                        actions.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': 0,
                            'a': 0.0,
                            'b': 0.0,
                            'type': 'direct_pot',
                            'target': tid,
                            'pocket_xy': pocket_xy,
                            'is_priority': True
                        })

        # Generate candidates for other targets (fewer)
        for tid in other_targets:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue

            for pocket_xy in pocket_centers[:3]:  # Only first 3 pockets
                phi_ghost, dist_cue_to_ghost = self._get_ghost_ball_target(cue_xy, obj_xy, pocket_xy)
                if dist_cue_to_ghost < 1e-6:
                    continue

                v_base = float(np.clip(1.5 + dist_cue_to_ghost * 1.5, 1.0, 6.0))
                speeds = [float(np.clip(v_base, 1.0, 8.0))]  # Only base speed

                for t in [-T, 0.0, T]:  # Fewer thickness variations
                    delta_phi = float(np.degrees(np.arctan2(t, dist_cue_to_ghost)))
                    phi = wrap_angle(phi_ghost + delta_phi)
                    for V0 in speeds:
                        actions.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': 0,
                            'a': 0.0,
                            'b': 0.0,
                            'type': 'direct_pot',
                            'target': tid,
                            'pocket_xy': pocket_xy,
                            'is_priority': False
                        })

        # Add safety shots for priority targets
        for tid in priority_targets:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue
            v = obj_xy - cue_xy
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                continue
            phi_direct = self._calc_angle_degrees(v)

            for angle_offset in (-35, -20, 20, 35):
                phi = wrap_angle(phi_direct + angle_offset)
                for V0 in (1.0, 1.4):
                    actions.append({
                        'V0': float(V0),
                        'phi': phi,
                        'theta': 0,
                        'a': 0.0,
                        'b': 0.0,
                        'type': 'safety',
                        'target': tid,
                        'is_priority': True
                    })

        # Add random exploration
        for _ in range(12):
            a = self._random_action()
            a['type'] = 'random'
            a['is_priority'] = False
            actions.append(a)

        random.shuffle(actions)
        max_candidates = 500
        return actions[:max_candidates]
    
    def _prefilter_candidates(self, balls, candidates: List[dict], pocket_centers: List[np.ndarray], 
                              promising_targets=None) -> List[dict]:
        """Prefilter candidates with priority for promising targets"""
        cue_xy = ball_pos(balls, 'cue')
        if cue_xy is None:
            return candidates

        keep_total = 90
        
        # Separate by priority
        priority_candidates = [c for c in candidates if c.get('is_priority', False)]
        other_candidates = [c for c in candidates if not c.get('is_priority', False)]
        
        # Score and sort each group
        def score_candidate(c):
            score = 0.0
            tid = c.get('target', None)
            obj_xy = ball_pos(balls, tid) if tid else None
            
            if obj_xy is not None and c.get('type') == 'direct_pot':
                pocket_xy = c.get('pocket_xy', pocket_centers[0] if pocket_centers else np.array([0, 0]))
                blocked1 = is_line_blocked(cue_xy, obj_xy, balls, self.ball_radius, ignore_ids={'cue', str(tid)})
                blocked2 = is_line_blocked(obj_xy, pocket_xy, balls, self.ball_radius, ignore_ids={str(tid)})
                
                ca = cut_angle_deg(cue_xy, obj_xy, pocket_xy)
                d_obj_pocket = float(np.linalg.norm(pocket_xy - obj_xy))
                
                score += 2.0 if not blocked1 else -2.0
                score += 2.0 if not blocked2 else -2.0
                score -= ca / 90.0
                score -= d_obj_pocket / 2.54
                
            return score
        
        priority_scored = [(score_candidate(c), c) for c in priority_candidates]
        other_scored = [(score_candidate(c), c) for c in other_candidates]
        
        priority_scored.sort(key=lambda x: x[0], reverse=True)
        other_scored.sort(key=lambda x: x[0], reverse=True)
        
        # Keep 70% priority, 30% others
        n_priority = int(keep_total * 0.7)
        n_others = keep_total - n_priority
        
        kept = []
        kept.extend([c for _, c in priority_scored[:n_priority]])
        kept.extend([c for _, c in other_scored[:n_others]])
        
        random.shuffle(kept)
        return kept[:keep_total]
    
    def simulate_action(self, balls, table, action, my_targets, pocket_centers=None, risk_adjustment=0.0):
        """Simulate action with risk-adjusted reward"""
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

            noisy_V0 = float(np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0))
            noisy_phi = wrap_angle(action['phi'] + np.random.normal(0, self.noise_std['phi']))
            noisy_theta = float(np.clip(action.get('theta', 0) + np.random.normal(0, self.noise_std['theta']), 0, 90))
            noisy_a = float(np.clip(action.get('a', 0) + np.random.normal(0, self.noise_std['a']), -0.5, 0.5))
            noisy_b = float(np.clip(action.get('b', 0) + np.random.normal(0, self.noise_std['b']), -0.5, 0.5))

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)

            success = simulate_with_timeout(shot, timeout=3)
            if not success:
                return None, -500.0, -500.0

            raw = float(analyze_shot_for_reward(shot, last_state_snapshot, my_targets))
            shaped = raw

            # Apply risk adjustment
            if risk_adjustment != 0.0:
                if shaped < 0:
                    shaped *= (1.0 + risk_adjustment)
                elif shaped > 0:
                    shaped *= (1.0 - risk_adjustment * 0.3)

            # Scratch-risk shaping
            if pocket_centers:
                try:
                    cue_after = ball_pos(shot.balls, 'cue')
                    if cue_after is not None:
                        dmin = nearest_pocket_distance(cue_after, pocket_centers)
                        if dmin < 2.2 * self.ball_radius:
                            shaped -= 15.0
                except Exception:
                    pass

            return shot, raw, shaped

        except Exception:
            return None, -1000.0, -1000.0
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Make decision with VLM guidance"""
        self.decision_count += 1
        
        if balls is None:
            return self._random_action()
        
        remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        
        # Get VLM guidance
        if self._should_call_vlm(balls, my_targets):
            guidance = self._get_vlm_guidance(balls, my_targets, table)
            self.last_vlm_guidance = guidance
        elif self.last_vlm_guidance is not None:
            guidance = self.last_vlm_guidance
            print(f"[VLMAssistedAgent] Reusing previous VLM guidance")
        else:
            guidance = self._default_guidance(my_targets, balls)
            print(f"[VLMAssistedAgent] Using default guidance")
        
        promising_targets = guidance['promising_targets']
        risk = guidance['risk']
        budget = guidance['budget']
        
        # Adjust search parameters based on budget (60 to 240 sims)
        n_simulations = int(self.base_n_simulations * (0.33 + 1.0 * budget))
        
        # Adjust risk_lambda based on risk (0.1 to 0.5)
        risk_lambda = 0.1 + 0.4 * risk
        
        # Risk adjustment for reward shaping
        risk_adjustment = risk * 0.3
        
        print(f"[VLMAssistedAgent] Decision #{self.decision_count}:")
        print(f"  Simulations: {n_simulations} (budget={budget:.2f})")
        print(f"  Risk lambda: {risk_lambda:.2f} (risk={risk:.2f})")
        
        pocket_centers = [np.array([p.center[0], p.center[1]], dtype=float) for p in table.pockets.values()]
        
        # Generate and prefilter candidates
        candidates = self.generate_candidates(balls, my_targets, table, promising_targets)
        if not candidates:
            return self._random_action()
        
        candidates = self._prefilter_candidates(balls, candidates, pocket_centers, promising_targets)
        if not candidates:
            return self._random_action()
        
        # Two-stage search
        total_budget = int(max(1, n_simulations))
        stage1_n = min(90, len(candidates))
        stage1_r = 1
        stage2_k = 12
        stage2_m = max(0, (total_budget // stage2_k) - 1)
        
        stage1_actions = list(candidates)[:stage1_n]
        
        def norm(v: float) -> float:
            return float(np.clip((v - (-500.0)) / 650.0, 0.0, 1.0))
        
        # Stage 1
        s1_sums = np.zeros(stage1_n, dtype=float)
        s1_sums2 = np.zeros(stage1_n, dtype=float)
        s1_counts = np.zeros(stage1_n, dtype=int)
        
        for _ in range(stage1_r):
            for i, a in enumerate(stage1_actions):
                _, _, shaped = self.simulate_action(balls, table, a, my_targets, 
                                                    pocket_centers=pocket_centers,
                                                    risk_adjustment=risk_adjustment)
                v = norm(shaped)
                s1_sums[i] += v
                s1_sums2[i] += v * v
                s1_counts[i] += 1
        
        s1_means = s1_sums / (s1_counts + 1e-9)
        s1_vars = (s1_sums2 / (s1_counts + 1e-9)) - s1_means * s1_means
        s1_stds = np.sqrt(np.maximum(0.0, s1_vars))
        s1_est = s1_means - float(risk_lambda) * s1_stds
        
        # Stage 2
        stage2_k = min(stage2_k, stage1_n)
        top_idx = np.argsort(s1_est)[-stage2_k:][::-1]
        finalists = [stage1_actions[int(i)] for i in top_idx]
        
        k = len(finalists)
        sums = np.zeros(k, dtype=float)
        sums2 = np.zeros(k, dtype=float)
        counts = np.zeros(k, dtype=int)
        
        for j, idx in enumerate(top_idx):
            idx = int(idx)
            sums[j] = float(s1_sums[idx])
            sums2[j] = float(s1_sums2[idx])
            counts[j] = int(s1_counts[idx])
        
        for _ in range(stage2_m):
            for j, a in enumerate(finalists):
                _, _, shaped = self.simulate_action(balls, table, a, my_targets, 
                                                    pocket_centers=pocket_centers,
                                                    risk_adjustment=risk_adjustment)
                v = norm(shaped)
                sums[j] += v
                sums2[j] += v * v
                counts[j] += 1
        
        means = sums / (counts + 1e-9)
        vars_ = (sums2 / (counts + 1e-9)) - means * means
        stds = np.sqrt(np.maximum(0.0, vars_))
        estimates = means - float(risk_lambda) * stds
        
        best_idx = int(np.argmax(estimates))
        best_action = finalists[best_idx]
        
        print(f"[VLMAssistedAgent] Selected action targeting ball {best_action.get('target', '?')}")
        
        return {
            'V0': float(best_action['V0']),
            'phi': float(best_action['phi']),
            'theta': float(best_action.get('theta', 0)),
            'a': float(best_action.get('a', 0)),
            'b': float(best_action.get('b', 0)),
        }


def test_vlm_assisted_agent():
    """Test VLM-Assisted Agent"""
    import pooltool as pt
    
    print("Testing VLM-Assisted Agent")
    print("=" * 60)
    
    table = pt.Table.default()
    balls = {
        'cue': pt.Ball.create("cue", xy=[0.5, 0.5]),
        '1': pt.Ball.create("1", xy=[1.0, 0.56]),
        '2': pt.Ball.create("2", xy=[0.8, 0.8]),
        '3': pt.Ball.create("3", xy=[1.2, 0.9]),
        '8': pt.Ball.create("8", xy=[1.5, 1.2]),
        '9': pt.Ball.create("9", xy=[1.8, 0.7]),
    }
    
    my_targets = ['1', '2', '3']
    
    agent = VLMAssistedAgent(
        provider='qwen',
        model='qwen-vl-max',
        use_vlm=True,
        vlm_frequency='always'
    )
    
    action = agent.decision(balls=balls, my_targets=my_targets, table=table)
    
    print("\n" + "=" * 60)
    print("VLM-Assisted Agent Decision:")
    print(f"  V0 (velocity):     {action['V0']:.2f} m/s")
    print(f"  phi (h-angle):     {action['phi']:.1f}°")
    print(f"  theta (v-angle):   {action['theta']:.1f}°")
    print(f"  a (h-offset):      {action['a']:.3f}")
    print(f"  b (v-offset):      {action['b']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    test_vlm_assisted_agent()
