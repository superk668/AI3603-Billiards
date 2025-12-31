import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import random
import time
from collections import defaultdict

from .agent import Agent

# ============ Helper Functions ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """分析击球结果并计算奖励分数（完全对齐台球规则）"""
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
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
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
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
        
    # 计算奖励分数
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


def evaluate_position_quality(shot: pt.System, my_targets: list, table):
    """评估击球后的球位质量"""
    score = 0.0
    
    try:
        cue_ball = shot.balls.get('cue')
        if not cue_ball or cue_ball.state.s == 4:
            return -1.0
        
        cue_pos = cue_ball.state.rvw[0]
        remaining_targets = [bid for bid in my_targets if bid in shot.balls and shot.balls[bid].state.s != 4]
        
        if not remaining_targets:
            return 0.5
        
        min_distances = []
        for tid in remaining_targets:
            if tid not in shot.balls:
                continue
            target_pos = shot.balls[tid].state.rvw[0]
            
            min_dist = float('inf')
            for pocket_id, pocket in table.pockets.items():
                dist = np.linalg.norm(np.array(target_pos) - np.array(pocket.center))
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
        
        if min_distances:
            avg_target_to_pocket = np.mean(min_distances)
            score += (3.0 - avg_target_to_pocket) / 3.0 * 0.3
        
        if remaining_targets:
            min_cue_to_target = float('inf')
            for tid in remaining_targets:
                if tid not in shot.balls:
                    continue
                target_pos = shot.balls[tid].state.rvw[0]
                dist = np.linalg.norm(np.array(cue_pos) - np.array(target_pos))
                min_cue_to_target = min(min_cue_to_target, dist)
            
            if 0.3 <= min_cue_to_target <= 1.2:
                score += 0.2
            elif min_cue_to_target < 0.3:
                score += 0.1
            else:
                score -= 0.1
        
        table_length = 2.54
        table_width = 1.27
        x, y = cue_pos[0], cue_pos[1]
        
        min_margin = min(abs(x), abs(table_length - abs(x)), abs(y), abs(table_width - abs(y)))
        if min_margin > 0.3:
            score += 0.15
        elif min_margin < 0.15:
            score -= 0.15
        
        return np.clip(score, -1.0, 1.0)
    
    except Exception:
        return 0.0


class GlobalTimeMCTSAgent(Agent):
    """
    全局时间管理 MCTS Agent
    
    核心特性：
    1. 所有游戏共享一个总时间预算（3分钟 × 游戏局数）
    2. 根据局面重要性和胜负情况动态分配时间
    3. 充分利用所有可用时间
    4. 智能的局间时间分配策略
    
    时间分配策略：
    - 领先时：快速决策，保存时间给关键局
    - 落后时：投入更多时间争取翻盘
    - 平局时：根据剩余局数均衡分配
    - 关键决策（黑8、残局）：额外时间投入
    """
    
    def __init__(self,
                 n_games=40,                # 预期游戏总局数
                 time_per_game=180.0,       # 每局标准时间（3分钟）
                 base_simulations=50,
                 min_simulations=15,
                 max_simulations=200,
                 base_c_puct=1.414,
                 refinement_threshold=0.6,
                 position_weight=0.3):
        super().__init__()
        
        # 时间管理参数
        self.n_games = n_games
        self.total_time_budget = time_per_game * n_games  # 总时间预算
        self.remaining_time = self.total_time_budget
        self.time_per_game = time_per_game
        
        # MCTS 参数
        self.base_simulations = base_simulations
        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.base_c_puct = base_c_puct
        self.refinement_threshold = refinement_threshold
        self.position_weight = position_weight
        self.ball_radius = 0.028575
        
        # 统计信息
        self.games_played = 0
        self.games_won = 0
        self.games_lost = 0
        self.current_game_decisions = 0
        self.decision_count_total = 0
        self.time_history = []
        self.game_time_usage = []
        self.current_game_start_time = None
        self.current_game_time_used = 0.0
        
        # 游戏状态跟踪
        self.last_active_balls_count = 0
        self.current_game_id = 0
        
        # 噪声水平
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        print(f"GlobalTimeMCTSAgent 已初始化")
        print(f"  总时间预算: {self.total_time_budget:.0f}s ({n_games}局 × {time_per_game:.0f}s)")
        print(f"  仿真范围: {min_simulations}-{max_simulations}")
        print(f"  基础仿真: {base_simulations}")

    def detect_new_game(self, balls):
        """检测是否开始了新游戏"""
        if balls is None:
            return False
        
        active_balls = sum(1 for ball in balls.values() if ball.state.s != 4)
        
        is_new_game = False
        if self.last_active_balls_count == 0:
            is_new_game = True
        elif active_balls > self.last_active_balls_count + 8:
            is_new_game = True
        
        self.last_active_balls_count = active_balls
        return is_new_game

    def on_new_game_start(self):
        """新游戏开始时的处理"""
        # 记录上一局的时间使用
        if self.current_game_start_time is not None:
            self.game_time_usage.append(self.current_game_time_used)
        
        self.games_played += 1
        self.current_game_id += 1
        self.current_game_decisions = 0
        self.current_game_start_time = time.time()
        self.current_game_time_used = 0.0
        
        # 计算统计信息
        games_remaining = max(self.n_games - self.games_played + 1, 1)
        avg_time_per_game = self.remaining_time / games_remaining
        win_rate = self.games_won / max(self.games_played - 1, 1) if self.games_played > 1 else 0.5
        
        print(f"\n{'='*70}")
        print(f"[GlobalTime] 第 {self.games_played}/{self.n_games} 局游戏开始")
        print(f"  当前战绩: {self.games_won}胜 {self.games_lost}负 (胜率: {win_rate*100:.1f}%)")
        print(f"  剩余时间: {self.remaining_time:.1f}s / {self.total_time_budget:.0f}s")
        print(f"  平均可用时间/局: {avg_time_per_game:.1f}s")
        print(f"{'='*70}\n")

    def estimate_game_importance(self):
        """
        评估当前局的重要性（0.0-1.0）
        
        考虑因素：
        1. 当前胜负情况（落后时更重要）
        2. 剩余局数（越少越重要）
        3. 胜率情况
        """
        importance = 0.5  # 基础重要性
        
        if self.games_played <= 1:
            return 0.4  # 第一局不太重要，先观察
        
        # 1. 基于胜负情况
        win_rate = self.games_won / (self.games_played - 1)
        if win_rate < 0.4:
            importance += 0.2  # 落后，需要争取
        elif win_rate < 0.45:
            importance += 0.1
        elif win_rate > 0.6:
            importance -= 0.1  # 领先，可以保守
        
        # 2. 基于剩余局数（越接近结束越重要）
        progress = self.games_played / self.n_games
        if progress > 0.8:
            importance += 0.3  # 最后20%的局非常重要
        elif progress > 0.6:
            importance += 0.15  # 后期比较重要
        
        # 3. 基于时间富余程度
        expected_time_used = self.games_played * self.time_per_game
        actual_time_used = self.total_time_budget - self.remaining_time
        
        if actual_time_used < expected_time_used * 0.8:
            # 时间用得少，可以多花时间
            importance += 0.1
        elif actual_time_used > expected_time_used * 1.1:
            # 时间用得多，需要节省
            importance -= 0.15
        
        return np.clip(importance, 0.3, 1.0)

    def allocate_game_time_budget(self):
        """为当前局分配时间预算"""
        games_remaining = max(self.n_games - self.games_played + 1, 1)
        
        # 基础分配：剩余时间平均分配
        base_allocation = self.remaining_time / games_remaining
        
        # 根据局的重要性调整
        importance = self.estimate_game_importance()
        
        # 重要性越高，分配越多时间
        importance_multiplier = 0.7 + importance * 0.8  # 0.7x ~ 1.5x
        allocated_time = base_allocation * importance_multiplier
        
        # 确保不超过剩余时间的50%（为后续局保留）
        max_allocation = self.remaining_time * 0.5
        allocated_time = min(allocated_time, max_allocation)
        
        # 确保至少有最小时间
        min_allocation = 30.0  # 至少30秒
        allocated_time = max(allocated_time, min_allocation)
        
        return allocated_time

    def estimate_decision_complexity(self, balls, my_targets):
        """评估决策复杂度（0.0-1.0）"""
        complexity = 0.0
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return 0.5
        cue_pos = cue_ball.state.rvw[0]
        
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        
        n_targets = len(target_ids)
        
        # 剩余球数影响
        if target_ids == ['8']:
            complexity += 0.5
        elif n_targets <= 2:
            complexity += 0.3
        elif n_targets <= 4:
            complexity += 0.15
        else:
            complexity += 0.05
        
        return np.clip(complexity, 0.0, 1.0)

    def allocate_decision_simulations(self, complexity, game_time_budget, estimated_decisions_remaining):
        """为当前决策分配仿真次数"""
        # 1. 基于游戏时间预算和剩余决策数
        avg_time_per_decision = game_time_budget / max(estimated_decisions_remaining, 1)
        
        # 2. 估算每次仿真的时间
        if len(self.time_history) > 0:
            recent_times = self.time_history[-10:]
            avg_sim_time = np.mean(recent_times) / self.base_simulations
        else:
            avg_sim_time = 0.08
        
        # 3. 基于复杂度调整
        complexity_multiplier = 0.6 + complexity * 1.2  # 0.6x ~ 1.8x
        
        # 4. 计算分配
        time_constrained_sims = int(avg_time_per_decision / avg_sim_time * 0.75)
        complexity_based_sims = int(self.base_simulations * complexity_multiplier)
        
        allocated_sims = min(time_constrained_sims, complexity_based_sims)
        allocated_sims = np.clip(allocated_sims, self.min_simulations, self.max_simulations)
        
        # 5. 紧急情况处理
        if self.remaining_time < 30.0:
            allocated_sims = self.min_simulations
        elif self.remaining_time < 60.0:
            allocated_sims = int(allocated_sims * 0.6)
        
        return int(allocated_sims)

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _random_action(self):
        """生成随机击球动作"""
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
        return action

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        """计算幽灵球位置和击球角度"""
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0:
            return 0, 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost, dist_obj_to_pocket

    def _calculate_shot_difficulty(self, cue_pos, obj_pos, pocket_pos):
        """计算击球难度"""
        phi, dist_cue_to_obj, dist_obj_to_pocket = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
        
        distance_penalty = dist_cue_to_obj * 0.3 + dist_obj_to_pocket * 0.5
        
        vec_cue_to_obj = np.array(obj_pos) - np.array(cue_pos)
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        
        if np.linalg.norm(vec_cue_to_obj) > 0 and np.linalg.norm(vec_obj_to_pocket) > 0:
            cos_angle = np.dot(vec_cue_to_obj, vec_obj_to_pocket) / (
                np.linalg.norm(vec_cue_to_obj) * np.linalg.norm(vec_obj_to_pocket))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angle_penalty = abs(angle) * 0.5
        else:
            angle_penalty = 0
        
        difficulty = distance_penalty + angle_penalty
        return difficulty

    def generate_strategic_actions(self, balls, my_targets, table, n_actions=20):
        """生成战略性候选动作"""
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']

        shot_options = []
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                difficulty = self._calculate_shot_difficulty(cue_pos, obj_pos, pocket_pos)
                phi, dist, _ = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                
                shot_options.append({
                    'target_id': tid,
                    'pocket_id': pocket_id,
                    'difficulty': difficulty,
                    'phi': phi,
                    'distance': dist
                })
        
        shot_options.sort(key=lambda x: x['difficulty'])
        
        for i, shot_opt in enumerate(shot_options[:min(10, len(shot_options))]):
            phi_ideal = shot_opt['phi']
            dist = shot_opt['distance']
            
            v_base = np.clip(1.5 + dist * 1.2, 1.0, 5.5)
            
            variations = [
                {'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0},
                {'V0': v_base + 1.0, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0},
                {'V0': v_base, 'phi': (phi_ideal + 0.3) % 360, 'theta': 0, 'a': 0, 'b': 0},
                {'V0': v_base, 'phi': (phi_ideal - 0.3) % 360, 'theta': 0, 'a': 0, 'b': 0},
            ]
            
            if i < 3:
                variations.extend([
                    {'V0': v_base + 0.5, 'phi': phi_ideal, 'theta': 0, 'a': 0.1, 'b': 0},
                    {'V0': v_base + 0.5, 'phi': phi_ideal, 'theta': 0, 'a': -0.1, 'b': 0},
                ])
            
            actions.extend(variations)
        
        if len(actions) > n_actions:
            actions = actions[:n_actions]
        
        while len(actions) < 10:
            actions.append(self._random_action())
        
        return actions

    def refine_action(self, action, n_refinements=4):
        """围绕有希望的动作生成细化变种"""
        refined_actions = [action]
        
        for _ in range(n_refinements):
            refined = {
                'V0': np.clip(action['V0'] + random.uniform(-0.3, 0.3), 0.5, 8.0),
                'phi': (action['phi'] + random.uniform(-0.8, 0.8)) % 360,
                'theta': np.clip(action['theta'] + random.uniform(-2, 2), 0, 90),
                'a': np.clip(action['a'] + random.uniform(-0.05, 0.05), -0.5, 0.5),
                'b': np.clip(action['b'] + random.uniform(-0.05, 0.05), -0.5, 0.5),
            }
            refined_actions.append(refined)
        
        return refined_actions

    def simulate_action(self, balls, table, action):
        """执行带噪声的物理仿真"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def evaluate_shot(self, shot, last_state, my_targets, table):
        """综合评估击球质量"""
        if shot is None:
            return -500.0
        
        immediate_reward = analyze_shot_for_reward(shot, last_state, my_targets)
        
        if immediate_reward > -100:
            position_score = evaluate_position_quality(shot, my_targets, table)
            position_bonus = position_score * 50 * self.position_weight
        else:
            position_bonus = 0
        
        total_score = immediate_reward + position_bonus
        return total_score

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()
        
        # 检测新游戏
        if self.detect_new_game(balls):
            self.on_new_game_start()
        
        # 开始计时
        decision_start_time = time.time()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 1. 分配本局时间预算（仅在第一次决策时）
        if self.current_game_decisions == 0:
            game_time_budget = self.allocate_game_time_budget()
        else:
            # 后续决策使用剩余时间
            game_time_budget = self.remaining_time
        
        # 2. 评估决策复杂度
        complexity = self.estimate_decision_complexity(balls, my_targets)
        
        # 3. 估计剩余决策次数
        n_targets = len([bid for bid in my_targets if balls[bid].state.s != 4])
        estimated_remaining_decisions = max(int(n_targets * 1.5), 3)
        
        # 4. 分配仿真次数
        n_simulations = self.allocate_decision_simulations(
            complexity, game_time_budget, estimated_remaining_decisions
        )
        
        print(f"[GlobalTime] 决策 #{self.current_game_decisions + 1} (总#{self.decision_count_total + 1})")
        print(f"  复杂度: {complexity:.2f} | 仿真: {n_simulations} | 剩余时间: {self.remaining_time:.1f}s")
        
        # 5. 生成候选动作
        candidate_actions = self.generate_strategic_actions(balls, my_targets, table, n_actions=15)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # 自适应探索系数
        n_remaining = len([bid for bid in my_targets if balls[bid].state.s != 4])
        adaptive_c_puct = self.base_c_puct * (1.0 + 0.1 * n_remaining)
        
        # 6. 第一阶段MCTS
        initial_sims = int(n_simulations * 0.6)
        
        for i in range(initial_sims):
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                ucb_values = (Q / (N + 1e-6)) + adaptive_c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            shot = self.simulate_action(balls, table, candidate_actions[idx])
            raw_reward = self.evaluate_shot(shot, last_state_snapshot, my_targets, table)
            normalized_reward = (raw_reward - (-500)) / 700.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            N[idx] += 1
            Q[idx] += normalized_reward

        # 7. 第二阶段：细化（如果时间允许）
        avg_rewards = Q / (N + 1e-6)
        promising_indices = np.where(avg_rewards >= self.refinement_threshold)[0]
        
        should_refine = (len(promising_indices) > 0) and (self.remaining_time > 30.0)
        
        if should_refine:
            top_k = min(2, len(promising_indices))
            top_indices = promising_indices[np.argsort(-avg_rewards[promising_indices])[:top_k]]
            
            refined_actions = []
            for idx in top_indices:
                refined = self.refine_action(candidate_actions[idx], n_refinements=3)
                refined_actions.extend(refined)
            
            n_refined = len(refined_actions)
            N_refined = np.zeros(n_refined)
            Q_refined = np.zeros(n_refined)
            
            remaining_sims = n_simulations - initial_sims
            
            for i in range(remaining_sims):
                if i < n_refined:
                    idx = i
                else:
                    total_n = np.sum(N_refined)
                    ucb_values = (Q_refined / (N_refined + 1e-6)) + adaptive_c_puct * np.sqrt(np.log(total_n + 1) / (N_refined + 1e-6))
                    idx = np.argmax(ucb_values)
                
                shot = self.simulate_action(balls, table, refined_actions[idx])
                raw_reward = self.evaluate_shot(shot, last_state_snapshot, my_targets, table)
                normalized_reward = (raw_reward - (-500)) / 700.0
                normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
                
                N_refined[idx] += 1
                Q_refined[idx] += normalized_reward
            
            avg_refined = Q_refined / (N_refined + 1e-6)
            best_refined_idx = np.argmax(avg_refined)
            best_refined_score = avg_refined[best_refined_idx]
            
            best_original_idx = np.argmax(avg_rewards)
            best_original_score = avg_rewards[best_original_idx]
            
            if best_refined_score > best_original_score:
                best_action = refined_actions[best_refined_idx]
                best_score = best_refined_score
            else:
                best_action = candidate_actions[best_original_idx]
                best_score = best_original_score
        else:
            best_idx = np.argmax(avg_rewards)
            best_action = candidate_actions[best_idx]
            best_score = avg_rewards[best_idx]
        
        # 8. 更新统计信息
        decision_time = time.time() - decision_start_time
        self.remaining_time -= decision_time
        self.current_game_time_used += decision_time
        self.time_history.append(decision_time)
        self.current_game_decisions += 1
        self.decision_count_total += 1
        
        print(f"  分数: {best_score:.3f} | 用时: {decision_time:.2f}s | 总剩余: {self.remaining_time:.1f}s")
        
        return best_action

    def report_game_result(self, won):
        """报告游戏结果（外部调用）"""
        if won:
            self.games_won += 1
        else:
            self.games_lost += 1
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'games_played': self.games_played,
            'games_won': self.games_won,
            'games_lost': self.games_lost,
            'win_rate': self.games_won / max(self.games_played, 1),
            'total_decisions': self.decision_count_total,
            'remaining_time': self.remaining_time,
            'time_used': self.total_time_budget - self.remaining_time,
            'time_utilization': (self.total_time_budget - self.remaining_time) / self.total_time_budget,
        }

