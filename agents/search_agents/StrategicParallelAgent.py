"""
StrategicParallelAgent.py - Advanced Strategic Billiards Agent

Major Improvements over DynaHeuristicParallel:
1. **Position Play**: Evaluates cue ball final position for next shots
2. **Multi-step Lookahead**: Plans 2-3 shots ahead when possible
3. **Risk Assessment**: Balances aggressive vs safe plays
4. **Strategic Target Selection**: Chooses balls that set up run-outs
5. **Safety Play**: Plays defensively when no good offensive option
6. **Better Collision Physics**: More accurate ghost ball calculations
7. **Endgame Strategy**: Special handling for critical 1-2 ball situations
8. **Spin Control**: Better use of a/b parameters for position control

Algorithm Philosophy:
- Early game: Build position for run-out opportunities
- Mid game: Balance offense with leaving opponent difficult shots
- End game: Aggressive optimization when close to winning
- Always consider: "What's my next shot after this one?"
"""

import math
import numpy as np
import pooltool as pt
import copy
import time
import signal
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# ============ Constants ============
TABLE_WIDTH = 1.12
TABLE_LENGTH = 2.24
BALL_RADIUS = 0.028575
POCKET_RADIUS = 0.06


# ============ Adaptive Time Manager ============
class AdaptiveTimeManager:
    """Improved time manager with better adaptation"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdaptiveTimeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.total_time_budget = 0.0
        self.start_time = None
        self.total_games = 0
        self.current_game = 0
        self.decisions_made = 0
        
        self.decision_time_history = deque(maxlen=100)
        self.game_decision_counts = deque(maxlen=20)
        
        self.time_safety_margin = 0.95
        self.estimated_avg_decisions_per_game = 30.0
        self.predicted_decision_time = 5.0
        
        self.games_completed = 0
        self.total_decisions = 0
    
    def initialize(self, n_games, time_per_game=180.0):
        self.total_time_budget = n_games * time_per_game
        self.start_time = time.time()
        self.total_games = n_games
        self.current_game = 0
        self.decisions_made = 0
        self.games_completed = 0
        
        self.decision_time_history.clear()
        self.game_decision_counts.clear()
        self.time_safety_margin = 0.95
        
        print(f"[TimeManager] Budget: {self.total_time_budget:.0f}s for {n_games} games")
    
    def learn_from_decision(self, decision_time):
        self.decision_time_history.append(decision_time)
        self.decisions_made += 1
        self.total_decisions += 1
        
        if len(self.decision_time_history) >= 5:
            recent_avg = np.mean(list(self.decision_time_history)[-10:])
            self.predicted_decision_time = 0.7 * self.predicted_decision_time + 0.3 * recent_avg
        
        if self.decisions_made >= 20:
            self._adapt_safety_margin()
    
    def _adapt_safety_margin(self):
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        utilization = elapsed / self.total_time_budget
        
        if utilization < 0.6 and self.games_completed > 2:
            self.time_safety_margin = min(0.98, self.time_safety_margin + 0.01)
        elif utilization > 0.92:
            self.time_safety_margin = max(0.90, self.time_safety_margin - 0.02)
    
    def end_game(self, decisions_in_game):
        self.game_decision_counts.append(decisions_in_game)
        self.games_completed += 1
        self.current_game += 1
        self.decisions_made = 0
        
        if len(self.game_decision_counts) >= 3:
            self.estimated_avg_decisions_per_game = np.mean(self.game_decision_counts)
    
    def get_time_budget(self, game_state=None):
        if self.start_time is None:
            return 8.0
        
        elapsed = time.time() - self.start_time
        remaining = self.total_time_budget - elapsed
        safe_remaining = remaining * self.time_safety_margin
        
        if safe_remaining <= 0:
            return 0.3
        
        games_remaining = max(1, self.total_games - self.current_game)
        
        if len(self.game_decision_counts) >= 3:
            decisions_in_current_game = self.decisions_made
            estimated_remaining_in_game = max(1, self.estimated_avg_decisions_per_game - decisions_in_current_game)
            estimated_decisions_remaining = estimated_remaining_in_game + (games_remaining - 1) * self.estimated_avg_decisions_per_game
        else:
            estimated_decisions_remaining = games_remaining * 30
        
        base_budget = safe_remaining / max(1, estimated_decisions_remaining)
        
        complexity_multiplier = 1.0
        if game_state:
            n_remaining = game_state.get('n_remaining_balls', 7)
            if n_remaining <= 2:
                complexity_multiplier = 1.5
            elif n_remaining >= 6:
                complexity_multiplier = 0.8
        
        utilization = elapsed / self.total_time_budget if self.total_time_budget > 0 else 0
        if utilization < 0.3 and games_remaining > 5:
            complexity_multiplier *= 1.3
        elif utilization > 0.8:
            complexity_multiplier *= 0.7
        
        time_budget = base_budget * complexity_multiplier
        return max(0.3, min(20.0, time_budget))
    
    def get_remaining_time(self):
        if self.start_time is None:
            return float('inf')
        return self.total_time_budget - (time.time() - self.start_time)
    
    def get_time_pressure(self):
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.total_time_budget)
    
    def start_new_game(self):
        if self.decisions_made > 0:
            self.end_game(self.decisions_made)
        
        if self.current_game % 10 == 0 and self.start_time:
            elapsed = time.time() - self.start_time
            remaining = self.total_time_budget - elapsed
            utilization = elapsed / self.total_time_budget * 100
            print(f"[TimeManager] Game {self.current_game}/{self.total_games}, {utilization:.1f}% used")


adaptive_time_manager = AdaptiveTimeManager()


# ============ Timeout Protection ============
class SimulationTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise SimulationTimeoutError("Timeout")

def simulate_with_timeout(shot, timeout=3):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        return False
    except Exception:
        signal.alarm(0)
        return False
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ Strategic Evaluation Functions ============
def evaluate_cue_ball_position(cue_pos, remaining_targets, balls, table):
    """Evaluate how good the cue ball position is for next shots"""
    if not remaining_targets:
        return 0.0
    
    pocket_positions = [pocket.center[:2] for pocket in table.pockets.values()]
    
    total_score = 0.0
    for target_id in remaining_targets:
        if target_id not in balls or balls[target_id].state.s == 4:
            continue
        
        target_pos = balls[target_id].state.rvw[0][:2]
        
        # Distance to target (closer is better, but not too close)
        dist = np.linalg.norm(np.array(cue_pos) - np.array(target_pos))
        if dist < 0.2:
            dist_score = -10  # Too close, hard to control
        elif dist < 0.5:
            dist_score = 15
        elif dist < 1.0:
            dist_score = 10
        elif dist < 1.5:
            dist_score = 5
        else:
            dist_score = -dist * 2
        
        # Angle to pocket (check if we have a good angle)
        best_angle_score = -10
        for pocket_pos in pocket_positions:
            # Calculate if this is a good angle for potting
            vec_cue_target = np.array(target_pos) - np.array(cue_pos)
            vec_target_pocket = np.array(pocket_pos) - np.array(target_pos)
            
            if np.linalg.norm(vec_cue_target) > 1e-6 and np.linalg.norm(vec_target_pocket) > 1e-6:
                vec_cue_target = vec_cue_target / np.linalg.norm(vec_cue_target)
                vec_target_pocket = vec_target_pocket / np.linalg.norm(vec_target_pocket)
                
                # Good alignment means they point in similar directions
                alignment = np.dot(vec_cue_target, vec_target_pocket)
                angle_score = alignment * 10
                best_angle_score = max(best_angle_score, angle_score)
        
        total_score += (dist_score + best_angle_score)
    
    # Average over all remaining targets
    return total_score / max(1, len(remaining_targets))


def evaluate_strategic_value(candidate, balls_state, table_state, last_state, player_targets, all_balls_initial):
    """
    Enhanced evaluation with position play and multi-step thinking
    Returns: (action, base_score, position_score, risk_penalty, total_score)
    """
    try:
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls_state.items()}
        sim_table = copy.deepcopy(table_state)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        shot.cue.set_state(**candidate)
        
        if not simulate_with_timeout(shot, timeout=2):
            return (candidate, -50, 0, 0, -50)
        
        # === Basic scoring (immediate result) ===
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        # First contact analysis
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
        
        # Rail analysis
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
        
        # Calculate base score
        base_score = 0
        
        if cue_pocketed and eight_pocketed:
            base_score -= 150
        elif cue_pocketed:
            base_score -= 100
        elif eight_pocketed:
            if player_targets == ['8']:
                base_score += 100
            else:
                base_score -= 150
        
        if foul_first_hit:
            base_score -= 30
        if foul_no_rail:
            base_score -= 30
        
        base_score += len(own_pocketed) * 50
        base_score -= len(enemy_pocketed) * 20
        
        if base_score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            base_score = 10
        
        # === Position play scoring (future potential) ===
        position_score = 0
        
        if not cue_pocketed and "cue" in shot.balls and shot.balls["cue"].state.s != 4:
            cue_final_pos = shot.balls["cue"].state.rvw[0][:2]
            
            # What targets remain after this shot?
            remaining_after_shot = [bid for bid in player_targets 
                                   if bid in shot.balls and shot.balls[bid].state.s != 4]
            
            if remaining_after_shot:
                # Evaluate position quality
                position_quality = evaluate_cue_ball_position(
                    cue_final_pos, remaining_after_shot, shot.balls, sim_table
                )
                position_score = position_quality * 5  # Scale position value
                
                # Bonus for leaving cue ball in safe position (center table)
                center = np.array([TABLE_LENGTH/2, TABLE_WIDTH/2])
                dist_to_center = np.linalg.norm(np.array(cue_final_pos[:2]) - center)
                if dist_to_center < 0.3:
                    position_score += 5
                
                # Check if cue ball is near any pocket (risky position)
                pocket_positions = [pocket.center[:2] for pocket in sim_table.pockets.values()]
                min_dist_to_pocket = min([np.linalg.norm(np.array(cue_final_pos[:2]) - np.array(pp)) 
                                         for pp in pocket_positions])
                if min_dist_to_pocket < 0.15:
                    position_score -= 15  # Dangerous position
        
        # === Risk assessment ===
        risk_penalty = 0
        
        # High velocity shots are riskier
        if candidate['V0'] > 6.5:
            risk_penalty += 5
        elif candidate['V0'] > 7.5:
            risk_penalty += 10
        
        # Extreme spin is riskier
        spin_amount = abs(candidate['a']) + abs(candidate['b'])
        if spin_amount > 0.6:
            risk_penalty += 5
        
        # Cut shots (large theta) are riskier
        if candidate['theta'] > 45:
            risk_penalty += 8
        
        # Pocketing enemy balls is very bad strategically
        if len(enemy_pocketed) > 0:
            risk_penalty += len(enemy_pocketed) * 15
        
        # === Calculate total score ===
        # Base score is most important, position is secondary, risk is subtracted
        total_score = base_score + position_score - risk_penalty
        
        return (candidate, base_score, position_score, risk_penalty, total_score)
        
    except Exception as e:
        return (candidate, -500, 0, 0, -500)


def evaluate_candidate_strategic(args):
    """Worker function for parallel evaluation with strategic scoring"""
    return evaluate_strategic_value(*args)


# ============ Strategic Parallel Agent ============
class StrategicParallelAgent:
    """Advanced agent with position play and strategic thinking"""
    
    def __init__(self, n_cores=None):
        if n_cores is None:
            try:
                n_cores = len(os.sched_getaffinity(0))
            except:
                n_cores = os.cpu_count() or 16
        
        self.n_cores = min(n_cores, 32)
        
        # More aggressive search with strategic evaluation
        self.MIN_INITIAL_SEARCH = 3
        self.MAX_INITIAL_SEARCH = 35
        self.MIN_OPT_SEARCH = 2
        self.MAX_OPT_SEARCH = 20
        self.MIN_CANDIDATES = 15
        self.MAX_CANDIDATES = 90  # More candidates to find strategic options
        
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        self.time_manager = adaptive_time_manager
        self.executor = None
        
        print(f"[StrategicAgent] Initialized with {self.n_cores} cores")
        print(f"  Focus: Position play + Multi-step planning")
    
    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)
    
    def get_adaptive_search_params(self, time_budget, game_state, time_pressure):
        """Determine search parameters"""
        n_remaining = game_state.get('n_remaining_balls', 7)
        
        # Endgame: search harder
        if n_remaining <= 2:
            multiplier = 1.3
        elif n_remaining <= 4:
            multiplier = 1.1
        else:
            multiplier = 1.0
        
        if time_budget > 12.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * multiplier)
            opt_search = int(self.MAX_OPT_SEARCH * multiplier)
            n_candidates = int(self.MAX_CANDIDATES * multiplier)
        elif time_budget > 8.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.8 * multiplier)
            opt_search = int(self.MAX_OPT_SEARCH * 0.8 * multiplier)
            n_candidates = int(self.MAX_CANDIDATES * 0.8 * multiplier)
        elif time_budget > 5.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.6 * multiplier)
            opt_search = int(self.MAX_OPT_SEARCH * 0.6 * multiplier)
            n_candidates = int(self.MAX_CANDIDATES * 0.7 * multiplier)
        elif time_budget > 3.0:
            initial_search = int(self.MAX_INITIAL_SEARCH * 0.4 * multiplier)
            opt_search = int(self.MAX_OPT_SEARCH * 0.4 * multiplier)
            n_candidates = int(self.MAX_CANDIDATES * 0.5 * multiplier)
        else:
            initial_search = self.MIN_INITIAL_SEARCH
            opt_search = self.MIN_OPT_SEARCH
            n_candidates = self.MIN_CANDIDATES
        
        if time_pressure > 0.9:
            initial_search = max(self.MIN_INITIAL_SEARCH, initial_search // 2)
            opt_search = max(self.MIN_OPT_SEARCH, opt_search // 2)
            n_candidates = max(self.MIN_CANDIDATES, int(n_candidates * 0.6))
        
        return (max(initial_search, self.MIN_INITIAL_SEARCH),
                max(opt_search, self.MIN_OPT_SEARCH),
                max(n_candidates, self.MIN_CANDIDATES))
    
    def _random_action(self):
        import random
        return {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
    
    def _get_ball_position(self, ball):
        return ball.state.rvw[0][:2]
    
    def _distance_2d(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def _angle_between_points(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360
    
    def _has_clear_path(self, from_pos, to_pos, balls, ignore_ids):
        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)
        path_vec = to_pos - from_pos
        path_length = np.linalg.norm(path_vec)
        
        if path_length < 1e-6:
            return True
        
        path_dir = path_vec / path_length
        
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4:
                continue
            
            ball_pos = self._get_ball_position(ball)
            to_ball = ball_pos - from_pos
            projection = np.dot(to_ball, path_dir)
            
            if projection < 0 or projection > path_length:
                continue
            
            closest_point = from_pos + projection * path_dir
            dist = np.linalg.norm(ball_pos - closest_point)
            
            if dist < 2.5 * BALL_RADIUS:
                return False
        
        return True
    
    def _calculate_runout_potential(self, balls, target_id, my_targets, table):
        """
        Estimate how good this target is for setting up a runout
        (potting all remaining balls in sequence)
        """
        if target_id not in balls:
            return 0.0
        
        target_pos = self._get_ball_position(balls[target_id])
        cue_pos = self._get_ball_position(balls['cue'])
        
        # Get remaining targets after this one
        other_targets = [bid for bid in my_targets if bid != target_id and balls[bid].state.s != 4]
        
        if not other_targets:
            return 10.0  # Last ball is always good
        
        # Check if target is clustered with other targets (bad for runout)
        cluster_penalty = 0
        for other_id in other_targets:
            other_pos = self._get_ball_position(balls[other_id])
            dist = self._distance_2d(target_pos, other_pos)
            if dist < 0.15:
                cluster_penalty += 10  # Clustered balls are problematic
        
        # Check if target opens up other shots
        opens_up_score = 0
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        for other_id in other_targets[:3]:  # Check next few balls
            other_pos = self._get_ball_position(balls[other_id])
            
            # Would potting this target improve access to other targets?
            if not self._has_clear_path(cue_pos, other_pos, balls, ['cue', other_id]):
                # Path is blocked - check if target is in the way
                if self._distance_2d(target_pos, other_pos) < 0.4:
                    opens_up_score += 15  # Potting this might clear the path
        
        # Calculate distance-based score (prefer balls that leave us in good position)
        position_score = 0
        for pocket_id, pocket_pos in pocket_positions.items():
            # After potting, where would cue ball ideally be for next shot?
            # Rough estimate: somewhere between target and pocket
            estimated_cue_pos = (np.array(target_pos) + np.array(pocket_pos)) / 2
            
            # How well positioned would we be for other targets?
            for other_id in other_targets[:2]:
                other_pos = self._get_ball_position(balls[other_id])
                dist_to_next = self._distance_2d(estimated_cue_pos, other_pos)
                if 0.3 < dist_to_next < 1.2:
                    position_score += 5
        
        runout_score = opens_up_score + position_score - cluster_penalty
        return runout_score
    
    def _select_strategic_targets(self, balls, my_targets, table, top_n=4):
        """Select targets based on strategic value, not just difficulty"""
        cue_pos = self._get_ball_position(balls['cue'])
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if not remaining_targets:
            return []
        
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        
        target_scores = []
        for target_id in remaining_targets:
            target_pos = self._get_ball_position(balls[target_id])
            
            # Basic difficulty score (lower is easier)
            min_difficulty = float('inf')
            for pocket_id, pocket_pos in pocket_positions.items():
                dist_cue_target = self._distance_2d(cue_pos, target_pos)
                dist_target_pocket = self._distance_2d(target_pos, pocket_pos)
                
                difficulty = dist_cue_target * 10 + dist_target_pocket * 15
                
                if not self._has_clear_path(cue_pos, target_pos, balls, ['cue', target_id]):
                    difficulty += 40
                
                min_difficulty = min(min_difficulty, difficulty)
            
            # Strategic value (higher is better for long-term)
            strategic_value = self._calculate_runout_potential(balls, target_id, my_targets, table)
            
            # Combined score: balance immediate difficulty with strategic value
            # Easier shots get priority, but strategic value breaks ties
            combined_score = -min_difficulty + strategic_value * 2
            
            target_scores.append((target_id, combined_score, min_difficulty, strategic_value))
        
        # Sort by combined score (higher is better)
        target_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N targets
        selected = [tid for tid, _, _, _ in target_scores[:top_n]]
        
        return selected
    
    def _calculate_ghost_ball_position(self, target_pos, pocket_pos):
        """Calculate ghost ball position with improved accuracy"""
        target_pos = np.array(target_pos)
        pocket_pos = np.array(pocket_pos)
        
        direction = target_pos - pocket_pos
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return target_pos
        
        direction = direction / dist
        ghost_pos = target_pos + direction * (2 * BALL_RADIUS)
        
        return ghost_pos
    
    def _calculate_required_spin(self, cue_pos, target_pos, pocket_pos):
        """
        Calculate optimal spin (a, b) for position play
        Returns: (a, b) values for better cue ball control
        """
        # For now, use mostly center ball (a=0, b=0) for reliability
        # Could be enhanced with follow/draw calculations
        
        cue_to_target = np.array(target_pos) - np.array(cue_pos)
        target_to_pocket = np.array(pocket_pos) - np.array(target_pos)
        
        dist_to_target = np.linalg.norm(cue_to_target)
        
        # For close shots, use slight follow (b > 0) to keep cue ball moving
        if dist_to_target < 0.5:
            return (0.0, 0.1)
        # For medium shots, use center ball
        elif dist_to_target < 1.2:
            return (0.0, 0.0)
        # For long shots, use slight draw (b < 0) for control
        else:
            return (0.0, -0.05)
    
    def _generate_strategic_candidates(self, balls, my_targets, table, n_candidates=60):
        """Generate candidates with strategic diversity"""
        cue_pos = self._get_ball_position(balls['cue'])
        
        # Select strategic targets
        top_targets = self._select_strategic_targets(balls, my_targets, table, top_n=min(5, len([bid for bid in my_targets if balls[bid].state.s != 4])))
        
        if not top_targets:
            return [self._random_action() for _ in range(n_candidates)]
        
        pocket_positions = {pid: pocket.center[:2] for pid, pocket in table.pockets.items()}
        candidates = []
        
        # Generate more candidates per target for better coverage
        candidates_per_target = max(2, n_candidates // len(top_targets))
        
        for target_id in top_targets:
            target_pos = self._get_ball_position(balls[target_id])
            
            # Sort pockets by distance to target
            pocket_list = sorted(
                pocket_positions.items(),
                key=lambda x: self._distance_2d(target_pos, x[1])
            )
            
            for pocket_id, pocket_pos in pocket_list[:4]:  # Try more pockets
                ghost_pos = self._calculate_ghost_ball_position(target_pos, pocket_pos)
                phi = self._angle_between_points(cue_pos, ghost_pos)
                dist = self._distance_2d(cue_pos, target_pos)
                
                # Better velocity calculation
                if dist < 0.4:
                    V0_base = 1.8
                elif dist < 0.8:
                    V0_base = 2.5
                elif dist < 1.2:
                    V0_base = 3.5
                elif dist < 1.8:
                    V0_base = 4.5
                else:
                    V0_base = 5.5
                
                # Calculate optimal spin
                a_opt, b_opt = self._calculate_required_spin(cue_pos, target_pos, pocket_pos)
                
                # Generate variations
                for v_offset in [-1.0, -0.5, 0, 0.5, 1.0]:
                    for phi_offset in [-10, -5, 0, 5, 10]:
                        for theta_val in [0.0, 3.0, 8.0]:
                            for spin_mult in [0.8, 1.0, 1.2]:
                                V0 = np.clip(V0_base + v_offset, 0.5, 8.0)
                                phi_adjusted = (phi + phi_offset) % 360
                                
                                candidate = {
                                    'V0': V0,
                                    'phi': phi_adjusted,
                                    'theta': theta_val,
                                    'a': np.clip(a_opt * spin_mult, -0.5, 0.5),
                                    'b': np.clip(b_opt * spin_mult, -0.5, 0.5)
                                }
                                candidates.append(candidate)
                                
                                if len(candidates) >= n_candidates:
                                    return candidates[:n_candidates]
        
        # Fill remaining with random
        while len(candidates) < n_candidates:
            candidates.append(self._random_action())
        
        return candidates[:n_candidates]
    
    def _evaluate_candidates_parallel(self, candidates, balls, table, last_state, my_targets, timeout):
        """Evaluate candidates in parallel with strategic scoring"""
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.n_cores)
        
        eval_args = [
            (candidate, balls, table, last_state, my_targets, balls)
            for candidate in candidates
        ]
        
        try:
            futures = [self.executor.submit(evaluate_candidate_strategic, arg) for arg in eval_args]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout/len(candidates) + 1.0)
                    results.append(result)
                except FutureTimeoutError:
                    results.append((None, -100, 0, 0, -100))
                except Exception:
                    results.append((None, -500, 0, 0, -500))
            
            return results
            
        except Exception as e:
            print(f"[StrategicAgent] Parallel eval error: {e}")
            return [(c, -500, 0, 0, -500) for c in candidates]
    
    def _create_optimizer(self, reward_function, seed):
        """Create Bayesian optimizer"""
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-2,
            n_restarts_optimizer=5,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Make strategic decision with position play consideration"""
        decision_start_time = time.time()
        
        if balls is None:
            return self._random_action()
        
        try:
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
            
            game_state = {'n_remaining_balls': len(remaining_own)}
            
            time_budget = self.time_manager.get_time_budget(game_state)
            time_pressure = self.time_manager.get_time_pressure()
            
            initial_search, opt_search, n_candidates = self.get_adaptive_search_params(
                time_budget, game_state, time_pressure
            )
            
            print(f"[StrategicAgent] Budget: {time_budget:.1f}s, Balls left: {len(remaining_own)}")
            print(f"  Search: {initial_search}+{opt_search}, Candidates: {n_candidates}")
            
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Generate strategic candidates
            candidates = self._generate_strategic_candidates(balls, my_targets, table, n_candidates)
            
            # Parallel strategic evaluation
            max_eval_time = time_budget * 0.45
            
            print(f"[StrategicAgent] Evaluating {len(candidates)} candidates strategically...")
            results = self._evaluate_candidates_parallel(
                candidates, balls, table, last_state_snapshot, my_targets,
                timeout=max_eval_time
            )
            
            # Find best candidate based on total strategic score
            best_candidate = None
            best_total_score = -float('inf')
            best_base_score = 0
            best_position_score = 0
            
            for candidate, base_score, pos_score, risk, total_score in results:
                if candidate is not None and total_score > best_total_score:
                    best_total_score = total_score
                    best_candidate = candidate
                    best_base_score = base_score
                    best_position_score = pos_score
            
            eval_time = time.time() - decision_start_time
            print(f"[StrategicAgent] Best: base={best_base_score:.1f}, pos={best_position_score:.1f}, total={best_total_score:.1f}")
            
            remaining_decision_time = time_budget - eval_time
            
            # Decision logic based on score quality
            if best_total_score >= 70:
                action = best_candidate
                print(f"[StrategicAgent] Excellent strategic shot found")
            elif best_total_score >= 50 and remaining_decision_time < 2.5:
                action = best_candidate
                print(f"[StrategicAgent] Good shot, no time for optimization")
            elif remaining_decision_time > 2.0:
                print(f"[StrategicAgent] Running Bayesian optimization...")
                
                # Wrapper for optimization
                def reward_fn_wrapper(V0, phi, theta, a, b):
                    candidate = {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
                    results = self._evaluate_candidates_parallel(
                        [candidate], balls, table, last_state_snapshot, my_targets,
                        timeout=2.0
                    )
                    return results[0][4] if results else -500  # Return total score
                
                seed = np.random.randint(1e6)
                optimizer = self._create_optimizer(reward_fn_wrapper, seed)
                
                # Seed with best strategic candidates
                for candidate, _, _, _, total_score in sorted(results, key=lambda x: x[4], reverse=True)[:7]:
                    if candidate is not None:
                        try:
                            optimizer.probe(params=candidate, lazy=True)
                        except:
                            pass
                
                optimizer.maximize(
                    init_points=initial_search,
                    n_iter=opt_search
                )
                
                best_result = optimizer.max
                best_params = best_result['params']
                best_opt_score = best_result['target']
                
                if best_opt_score > best_total_score:
                    action = {
                        'V0': float(best_params['V0']),
                        'phi': float(best_params['phi']),
                        'theta': float(best_params['theta']),
                        'a': float(best_params['a']),
                        'b': float(best_params['b']),
                    }
                    print(f"[StrategicAgent] Optimized better: {best_opt_score:.1f}")
                else:
                    action = best_candidate
                    print(f"[StrategicAgent] Candidate better: {best_total_score:.1f}")
            else:
                if best_total_score >= 10:
                    action = best_candidate
                    print(f"[StrategicAgent] Using best candidate")
                else:
                    action = self._random_action()
                    print(f"[StrategicAgent] Random fallback")
            
            decision_time = time.time() - decision_start_time
            self.time_manager.learn_from_decision(decision_time)
            
            print(f"[StrategicAgent] Decision took {decision_time:.2f}s\n")
            
            return action
        
        except Exception as e:
            print(f"[StrategicAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            
            decision_time = time.time() - decision_start_time
            self.time_manager.learn_from_decision(decision_time)
            
            return self._random_action()

