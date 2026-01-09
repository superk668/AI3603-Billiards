"""
MonteCarloGlobal.py - Global Time-Managed Monte Carlo Agent

Key Innovation: Adaptive Time Management
- Tracks time across ALL games in evaluation (not per-game)
- Dynamically adjusts n_simulations and n_noise_samples
- Calibrates to machine speed automatically
- Achieves 85-95% time utilization
- Adapts to game complexity

This agent is designed to maximize performance within strict time constraints
by intelligently allocating computational budget based on:
1. Remaining time across all games
2. Machine computational power
3. Game state complexity
4. Historical performance
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from .MonteCarloParallel import ParallelMCTSAgent


class GlobalTimeManager:
    """
    Global time management system for multi-game evaluation
    
    Tracks time usage across all games and adaptively allocates
    computational budget to maximize time utilization.
    """
    
    def __init__(self, 
                 n_games: int = 120,
                 time_per_game: float = 180.0,
                 target_utilization: float = 0.95,
                 min_simulations: int = 20,
                 max_simulations: int = 200,
                 min_noise_samples: int = 3,
                 max_noise_samples: int = 12):
        """
        Initialize global time manager
        
        Args:
            n_games: Total number of games
            time_per_game: Time limit per game (seconds)
            target_utilization: Target time usage (0.85-0.95 recommended)
            min/max_simulations: Bounds for n_simulations
            min/max_noise_samples: Bounds for n_noise_samples
        """
        self.n_games = n_games
        self.total_time = n_games * time_per_game
        self.target_utilization = target_utilization
        
        # Parameter bounds
        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.min_noise_samples = min_noise_samples
        self.max_noise_samples = max_noise_samples
        
        # Tracking
        self.start_time = None
        self.time_used = 0.0
        self.games_completed = 0
        self.decisions_made = 0
        self.decision_times = deque(maxlen=50)  # Recent decision times
        
        # Machine calibration
        self.calibrated = False
        self.time_per_simulation = None  # Will be measured
        self.calibration_samples = 0
        
        # Performance tracking
        self.game_start_time = None
        self.decisions_this_game = 0
        self.game_times = []  # Track time per game
        
        # Aggressive scaling
        self.last_game_time = None
        self.target_game_time = time_per_game * 0.95  # Aim for 95% of 3 min per game
        
        print(f"[GlobalTimeManager] Initialized")
        print(f"  Total time: {self.total_time:.0f}s ({n_games} games × {time_per_game:.0f}s)")
        print(f"  Target utilization: {target_utilization*100:.0f}%")
        print(f"  Parameter ranges: sims=[{min_simulations},{max_simulations}], "
              f"noise=[{min_noise_samples},{max_noise_samples}]")
    
    def start_evaluation(self):
        """Called at the beginning of evaluation"""
        self.start_time = time.time()
        self.time_used = 0.0
        self.games_completed = 0
        self.decisions_made = 0
        print(f"\n[GlobalTimeManager] Evaluation started at {time.strftime('%H:%M:%S')}")
    
    def start_game(self):
        """Called at the beginning of each game"""
        self.game_start_time = time.time()
        self.decisions_this_game = 0
    
    def end_game(self):
        """Called at the end of each game"""
        if self.game_start_time:
            game_time = time.time() - self.game_start_time
            self.games_completed += 1
            self.game_times.append(game_time)
            self.last_game_time = game_time
            
            time_elapsed = time.time() - self.start_time
            time_remaining = self.total_time - time_elapsed
            utilization = time_elapsed / self.total_time
            
            avg_game_time = np.mean(self.game_times) if self.game_times else 0
            
            print(f"\n[GlobalTimeManager] Game {self.games_completed}/{self.n_games} complete")
            print(f"  Game time: {game_time:.1f}s (target: {self.target_game_time:.1f}s)")
            print(f"  Avg game time: {avg_game_time:.1f}s")
            print(f"  Total elapsed: {time_elapsed:.1f}s / {self.total_time:.0f}s")
            print(f"  Utilization: {utilization*100:.1f}%")
            print(f"  Time remaining: {time_remaining:.1f}s")
            print(f"  Decisions made: {self.decisions_this_game} this game, "
                  f"{self.decisions_made} total")
    
    def calibrate(self, decision_time: float, n_simulations: int, n_noise_samples: int):
        """
        Calibrate machine speed based on initial decisions
        
        Args:
            decision_time: Time taken for decision
            n_simulations: Number of simulations used
            n_noise_samples: Number of noise samples used
        """
        if not self.calibrated:
            # Estimate time per (simulation × noise_sample)
            total_sims = n_simulations * n_noise_samples
            time_per_unit = decision_time / max(total_sims, 1)
            
            if self.time_per_simulation is None:
                self.time_per_simulation = time_per_unit
            else:
                # Exponential moving average
                alpha = 0.3
                self.time_per_simulation = (
                    alpha * time_per_unit + (1-alpha) * self.time_per_simulation
                )
            
            self.calibration_samples += 1
            
            # Calibrate after 3-5 decisions (faster)
            if self.calibration_samples >= 4:
                self.calibrated = True
                print(f"\n[GlobalTimeManager] ✓ Calibration complete")
                print(f"  Time per (sim×noise): {self.time_per_simulation*1000:.2f}ms")
                print(f"  Machine speed: {1.0/self.time_per_simulation:.0f} sims/sec")
                print(f"  → Starting AGGRESSIVE time utilization mode")
    
    def estimate_decisions_remaining(self, balls_remaining: int) -> int:
        """
        Estimate how many more decisions are needed
        
        Args:
            balls_remaining: Number of balls still on table
        
        Returns:
            Estimated decisions remaining in current game
        """
        # Heuristic: rough estimate based on balls remaining
        # Typical game: 15-30 shots depending on skill
        if balls_remaining > 10:
            return balls_remaining * 2
        elif balls_remaining > 5:
            return balls_remaining * 1.5
        else:
            return balls_remaining * 1.2
    
    def calculate_parameters(self, 
                           balls_remaining: int,
                           time_budget: float) -> Tuple[int, int]:
        """
        Calculate optimal n_simulations and n_noise_samples for time budget
        AGGRESSIVE VERSION: Maximizes parameter usage
        
        Args:
            balls_remaining: Balls left on table (for complexity estimation)
            time_budget: Available time for this decision (seconds)
        
        Returns:
            (n_simulations, n_noise_samples)
        """
        if not self.calibrated or self.time_per_simulation is None:
            # During calibration: use minimal parameters
            return (self.min_simulations, self.min_noise_samples)
        
        # AGGRESSIVE: Use 95% of time budget (was 80%)
        available_units = int(time_budget * 0.95 / self.time_per_simulation)
        
        # Minimum units required
        min_units = self.min_simulations * self.min_noise_samples
        max_units = self.max_simulations * self.max_noise_samples
        
        # Clamp to bounds
        target_units = np.clip(available_units, min_units, max_units)
        
        # AGGRESSIVE: Prioritize using MAX parameters
        # Strategy: Start with high noise for robustness, then maximize simulations
        
        # Game complexity factor
        complexity = min(balls_remaining / 15.0, 1.0)
        
        # Allocate aggressively
        if balls_remaining <= 3:
            # Endgame: maximize both
            noise_priority = 0.65
        elif target_units > max_units * 0.5:
            # Plenty of budget: maximize parameters
            noise_priority = 0.6
        else:
            # Normal: still aggressive
            noise_priority = 0.55
        
        # Calculate actual parameters (aim HIGH)
        n_noise = int(np.sqrt(target_units * noise_priority))
        n_noise = np.clip(n_noise, self.min_noise_samples, self.max_noise_samples)
        
        n_sims = target_units // n_noise
        n_sims = np.clip(n_sims, self.min_simulations, self.max_simulations)
        
        # AGGRESSIVE: Try to use full budget
        estimated_time = n_sims * n_noise * self.time_per_simulation
        if estimated_time < time_budget * 0.8:
            # We have more room, increase parameters
            scale_up = time_budget * 0.9 / estimated_time
            n_sims = min(self.max_simulations, int(n_sims * scale_up))
            n_noise = min(self.max_noise_samples, int(n_noise * scale_up))
        elif estimated_time > time_budget:
            # Scale down only if necessary
            scale = time_budget / estimated_time * 0.95
            n_sims = max(self.min_simulations, int(n_sims * scale))
            n_noise = max(self.min_noise_samples, int(n_noise * scale))
        
        return (n_sims, n_noise)
    
    def get_decision_budget(self, balls_remaining: int) -> Tuple[float, int, int]:
        """
        Get time budget and parameters for next decision
        AGGRESSIVE VERSION: Aims to use ~3 min per game
        
        Args:
            balls_remaining: Number of balls still on table
        
        Returns:
            (time_budget, n_simulations, n_noise_samples)
        """
        if self.start_time is None:
            self.start_evaluation()
        
        # Calculate time metrics
        time_elapsed = time.time() - self.start_time
        time_remaining = self.total_time - time_elapsed
        game_time_so_far = time.time() - self.game_start_time if self.game_start_time else 0
        
        # Estimate decisions remaining
        decisions_this_game_remaining = self.estimate_decisions_remaining(balls_remaining)
        avg_decisions_per_game = max(self.decisions_made / max(self.games_completed, 1), 20)
        games_remaining = self.n_games - self.games_completed
        
        # AGGRESSIVE: Calculate per-game budget
        # Target: Use ~3 minutes per game (target_game_time)
        game_time_target = self.target_game_time
        game_time_remaining = max(game_time_target - game_time_so_far, 10)
        
        # Base budget per decision (for THIS game)
        if decisions_this_game_remaining > 0:
            base_game_budget = game_time_remaining / decisions_this_game_remaining
        else:
            base_game_budget = game_time_remaining * 0.5
        
        # Also consider global time remaining
        total_decisions_remaining = (
            decisions_this_game_remaining + 
            avg_decisions_per_game * max(games_remaining - 1, 0)
        )
        if total_decisions_remaining > 0:
            global_budget = time_remaining / total_decisions_remaining
        else:
            global_budget = time_remaining * 0.1
        
        # AGGRESSIVE STRATEGY:
        # 1. If last game was UNDER target (e.g., 120s < 171s), DOUBLE parameters
        # 2. If last game was OVER target (e.g., 200s > 171s), reduce by 20%
        # 3. Otherwise use available budget aggressively
        
        if self.last_game_time is not None and self.calibrated:
            time_ratio = self.last_game_time / self.target_game_time
            
            if time_ratio < 0.7:
                # WAY under target: DOUBLE budget
                budget_factor = 2.5
                print(f"  [AGGRESSIVE] Last game {self.last_game_time:.1f}s << {self.target_game_time:.1f}s → 2.5x budget")
            elif time_ratio < 0.85:
                # Under target: increase significantly  
                budget_factor = 1.8
                print(f"  [AGGRESSIVE] Last game {self.last_game_time:.1f}s < {self.target_game_time:.1f}s → 1.8x budget")
            elif time_ratio > 1.15:
                # Over target: reduce
                budget_factor = 0.7
                print(f"  [CAUTION] Last game {self.last_game_time:.1f}s > {self.target_game_time:.1f}s → 0.7x budget")
            elif time_ratio > 1.05:
                # Slightly over: minor reduction
                budget_factor = 0.85
            else:
                # Just right: maintain
                budget_factor = 1.0
        else:
            # First game or calibrating: start conservative, then ramp up
            if self.games_completed == 0:
                budget_factor = 1.2  # Start moderate
            else:
                budget_factor = 1.5  # Ramp up after first game
        
        # Use the more generous of game budget vs global budget
        time_budget = max(base_game_budget, global_budget) * budget_factor
        
        # Safety: Don't exceed remaining time for this game
        time_budget = min(time_budget, game_time_remaining * 0.9)
        
        # Safety: Don't exceed global remaining time
        time_budget = min(time_budget, time_remaining / max(total_decisions_remaining, 1) * 2.0)
        
        # Calculate parameters
        n_sims, n_noise = self.calculate_parameters(balls_remaining, time_budget)
        
        return (time_budget, n_sims, n_noise)
    
    def record_decision(self, decision_time: float, 
                       n_simulations: int, n_noise_samples: int):
        """
        Record decision time for tracking and calibration
        
        Args:
            decision_time: Actual time taken
            n_simulations: Simulations used
            n_noise_samples: Noise samples used
        """
        self.decisions_made += 1
        self.decisions_this_game += 1
        self.decision_times.append(decision_time)
        self.time_used += decision_time
        
        # Continue calibration if not yet done
        if not self.calibrated:
            self.calibrate(decision_time, n_simulations, n_noise_samples)
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        time_elapsed = time.time() - self.start_time if self.start_time else 0
        utilization = time_elapsed / self.total_time if self.total_time > 0 else 0
        
        return {
            'time_elapsed': time_elapsed,
            'time_remaining': self.total_time - time_elapsed,
            'utilization': utilization,
            'games_completed': self.games_completed,
            'decisions_made': self.decisions_made,
            'avg_decision_time': np.mean(self.decision_times) if self.decision_times else 0,
            'calibrated': self.calibrated,
        }


class GlobalMCTSAgent(ParallelMCTSAgent):
    """
    Global Time-Managed MCTS Agent
    
    Extends ParallelMCTSAgent with adaptive time management.
    Dynamically adjusts parameters based on time budget.
    """
    
    # Shared time manager across all instances
    _time_manager = None
    
    @classmethod
    def initialize_time_manager(cls, n_games: int = 120, time_per_game: float = 180.0):
        """
        Initialize the global time manager (call before evaluation)
        
        Args:
            n_games: Total number of games in evaluation
            time_per_game: Time limit per game (seconds)
        """
        cls._time_manager = GlobalTimeManager(
            n_games=n_games,
            time_per_game=time_per_game,
            target_utilization=0.95,   # Aggressive: 95% utilization
            min_simulations=20,
            max_simulations=200,       # Double from 120
            min_noise_samples=3,
            max_noise_samples=12       # Increased from 9
        )
        print(f"\n[GlobalMCTSAgent] AGGRESSIVE time manager initialized for {n_games} games")
        print(f"  Target: ~{time_per_game:.0f}s per game ({time_per_game*0.95:.0f}s actual)")
        print(f"  Will DOUBLE parameters if game time << target")
    
    @classmethod
    def start_game(cls):
        """Call at the start of each game"""
        if cls._time_manager:
            cls._time_manager.start_game()
    
    @classmethod
    def end_game(cls):
        """Call at the end of each game"""
        if cls._time_manager:
            cls._time_manager.end_game()
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize Global MCTS Agent
        
        Note: n_simulations and n_noise_samples are determined dynamically
        """
        # Initialize with default parameters
        # These will be overridden by time manager
        super().__init__(
            n_simulations=40,  # Default (will be adjusted)
            n_noise_samples=4,  # Default (will be adjusted)
            c_puct=1.414,
            risk_aversion=0.5,
            n_workers=n_workers
        )
        
        print(f"[GlobalMCTSAgent] Initialized with adaptive time management")
        print(f"  Parameters will be adjusted dynamically based on time budget")
    
    def decision(self, balls: Optional[Dict] = None, 
                my_targets: Optional[List[str]] = None, 
                table = None) -> Dict:
        """
        Make decision with adaptive parameter selection
        
        Args:
            balls: Ball states
            my_targets: Target ball IDs
            table: Table object
        
        Returns:
            Action dictionary
        """
        if balls is None:
            return self._random_action()
        
        # Get time manager
        if self._time_manager is None:
            # Fallback: use default parameters
            print("[GlobalMCTSAgent] Warning: Time manager not initialized, using defaults")
            return super().decision(balls, my_targets, table)
        
        try:
            # Count balls remaining for complexity estimation
            balls_remaining = sum(1 for bid, ball in balls.items() 
                                if ball.state.s != 4 and bid != 'cue')
            
            # Get adaptive parameters
            time_budget, n_sims, n_noise = self._time_manager.get_decision_budget(
                balls_remaining
            )
            
            # Update agent parameters
            self.n_simulations = n_sims
            self.n_noise_samples = n_noise
            
            print(f"\n[GlobalMCTSAgent] Decision #{self._time_manager.decisions_made + 1}")
            print(f"  Balls remaining: {balls_remaining}")
            print(f"  Time budget: {time_budget:.2f}s")
            print(f"  Parameters: sims={n_sims}, noise={n_noise}")
            
            # Make decision and time it
            start_time = time.time()
            action = super().decision(balls, my_targets, table)
            decision_time = time.time() - start_time
            
            # Record for time management
            self._time_manager.record_decision(decision_time, n_sims, n_noise)
            
            # Print stats
            stats = self._time_manager.get_stats()
            print(f"  Actual time: {decision_time:.2f}s")
            print(f"  Utilization: {stats['utilization']*100:.1f}% "
                  f"({stats['time_elapsed']:.0f}s / {self._time_manager.total_time:.0f}s)")
            
            return action
            
        except Exception as e:
            print(f"[GlobalMCTSAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

