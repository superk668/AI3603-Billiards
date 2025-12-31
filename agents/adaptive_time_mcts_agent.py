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
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    """
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
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
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
    """
    评估击球后的球位质量（用于战略性决策）
    
    返回值范围：[-1, 1]
    - 正值：有利位置（目标球靠近袋口，白球位置好）
    - 负值：不利位置（目标球远离袋口，白球位置差）
    """
    score = 0.0
    
    try:
        cue_ball = shot.balls.get('cue')
        if not cue_ball or cue_ball.state.s == 4:
            return -1.0  # 白球进袋，最差情况
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 检查目标球是否还在桌上
        remaining_targets = [bid for bid in my_targets if bid in shot.balls and shot.balls[bid].state.s != 4]
        
        if not remaining_targets:
            return 0.5  # 所有目标球已进袋，位置已不重要
        
        # 计算目标球到最近袋口的平均距离（越小越好）
        min_distances = []
        for tid in remaining_targets:
            if tid not in shot.balls:
                continue
            target_pos = shot.balls[tid].state.rvw[0]
            
            # 找到最近的袋口
            min_dist = float('inf')
            for pocket_id, pocket in table.pockets.items():
                dist = np.linalg.norm(np.array(target_pos) - np.array(pocket.center))
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
        
        if min_distances:
            avg_target_to_pocket = np.mean(min_distances)
            # 归一化：假设桌面对角线约为3米，最差情况
            score += (3.0 - avg_target_to_pocket) / 3.0 * 0.3
        
        # 计算白球到最近目标球的距离（适中最好，太远不好打，太近也不好）
        if remaining_targets:
            min_cue_to_target = float('inf')
            for tid in remaining_targets:
                if tid not in shot.balls:
                    continue
                target_pos = shot.balls[tid].state.rvw[0]
                dist = np.linalg.norm(np.array(cue_pos) - np.array(target_pos))
                min_cue_to_target = min(min_cue_to_target, dist)
            
            # 理想距离约0.5-1.0米
            if 0.3 <= min_cue_to_target <= 1.2:
                score += 0.2
            elif min_cue_to_target < 0.3:
                score += 0.1
            else:
                score -= 0.1
        
        # 白球位置安全性：不要太靠近边界和袋口
        table_length = 2.54  # 标准台球桌长度
        table_width = 1.27   # 标准台球桌宽度
        x, y = cue_pos[0], cue_pos[1]
        
        # 距离边界的最小距离
        min_margin = min(abs(x), abs(table_length - abs(x)), abs(y), abs(table_width - abs(y)))
        if min_margin > 0.3:
            score += 0.15
        elif min_margin < 0.15:
            score -= 0.15
        
        return np.clip(score, -1.0, 1.0)
    
    except Exception:
        return 0.0


class AdaptiveTimeMCTSAgent(Agent):
    """
    自适应时间管理MCTS Agent - 动态分配计算资源
    
    核心特性：
    1. 全局时间预算管理：跟踪整局游戏的时间使用
    2. 动态资源分配：根据局面重要性调整仿真次数
    3. 决策复杂度评估：量化当前决策的难度
    4. 自适应策略：早期快速决策，关键时刻深度思考
    5. 时间紧急模式：时间不足时降级到快速决策
    
    时间分配策略：
    - 开局 (7球+): 快速决策 (30-50% 基础仿真)
    - 中局 (4-6球): 标准决策 (80-100% 基础仿真)
    - 残局 (1-3球): 深度思考 (120-150% 基础仿真)
    - 黑8决胜: 最深思考 (150-200% 基础仿真)
    """
    
    def __init__(self,
                 base_simulations=50,
                 total_time_budget=180.0,  # 总时间预算（秒）- 每局3分钟
                 min_simulations=20,       # 最小仿真次数
                 max_simulations=150,      # 最大仿真次数
                 base_c_puct=1.414,
                 refinement_threshold=0.6,
                 position_weight=0.3):
        super().__init__()
        
        # 基础参数
        self.base_simulations = base_simulations
        self.base_c_puct = base_c_puct
        self.refinement_threshold = refinement_threshold
        self.position_weight = position_weight
        self.ball_radius = 0.028575
        
        # 时间管理参数
        self.total_time_budget = total_time_budget
        self.remaining_time = total_time_budget
        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        
        # 统计信息
        self.decision_count = 0
        self.time_history = []
        self.complexity_history = []
        
        # 游戏状态跟踪（用于检测新游戏）
        self.last_active_balls_count = 0
        self.game_number = 0
        
        # 噪声水平
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        print(f"AdaptiveTimeMCTSAgent 已初始化")
        print(f"  每局时间预算: {total_time_budget}s (3分钟)")
        print(f"  仿真范围: {min_simulations}-{max_simulations}")
        print(f"  基础仿真: {base_simulations}")

    def reset_time_budget(self):
        """重置时间预算（新游戏开始时调用）"""
        self.remaining_time = self.total_time_budget
        self.decision_count = 0
        self.time_history = []
        self.complexity_history = []
        self.game_number += 1
        print(f"\n{'='*60}")
        print(f"[TimeManager] 第 {self.game_number} 局游戏开始")
        print(f"[TimeManager] 时间预算重置: {self.total_time_budget}s")
        print(f"{'='*60}\n")

    def detect_new_game(self, balls):
        """
        检测是否开始了新游戏
        
        策略：统计桌面上活跃球的数量
        - 新游戏开始时：通常有15个彩球+白球 = 16个球（状态!=4表示未进袋）
        - 游戏进行中：球数逐渐减少
        - 如果球数突然大幅增加（比如从5个变成15个），说明游戏重置了
        
        返回：
            bool: True 表示检测到新游戏
        """
        if balls is None:
            return False
        
        # 统计未进袋的球数（状态 s != 4）
        active_balls = sum(1 for ball in balls.values() if ball.state.s != 4)
        
        # 检测逻辑：
        # 1. 首次调用（last_active_balls_count == 0）
        # 2. 球数大幅增加（增加 >= 8 个球，说明游戏重置了）
        is_new_game = False
        
        if self.last_active_balls_count == 0:
            # 首次调用，初始化
            is_new_game = True
        elif active_balls > self.last_active_balls_count + 8:
            # 球数大幅增加，检测到游戏重置
            is_new_game = True
        
        # 更新记录
        self.last_active_balls_count = active_balls
        
        return is_new_game

    def estimate_decision_complexity(self, balls, my_targets, table):
        """
        评估当前决策的复杂度
        
        返回值：0.0-1.0，越高越复杂
        
        考虑因素：
        1. 剩余目标球数量（越少越关键）
        2. 球的分布（是否有明显的简单球）
        3. 是否是黑8决胜局
        """
        complexity = 0.0
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return 0.5
        cue_pos = cue_ball.state.rvw[0]
        
        # 获取剩余目标球
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        
        n_targets = len(target_ids)
        
        # 1. 剩余球数影响（越少越重要）
        if target_ids == ['8']:
            complexity += 0.5  # 黑8决胜，非常关键
        elif n_targets <= 2:
            complexity += 0.3  # 残局，很重要
        elif n_targets <= 4:
            complexity += 0.15  # 中局，中等重要
        else:
            complexity += 0.05  # 开局，相对简单
        
        # 2. 最简单击球的难度
        min_difficulty = float('inf')
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                difficulty = self._calculate_shot_difficulty(cue_pos, obj_pos, pocket_pos)
                min_difficulty = min(min_difficulty, difficulty)
        
        # 如果最简单的球也很难，增加复杂度
        if min_difficulty > 2.0:
            complexity += 0.3
        elif min_difficulty > 1.0:
            complexity += 0.15
        elif min_difficulty < 0.5:
            complexity -= 0.1  # 有非常简单的球，降低复杂度
        
        return np.clip(complexity, 0.0, 1.0)

    def allocate_simulations(self, complexity, estimated_decisions_remaining=10):
        """
        根据复杂度和剩余时间动态分配仿真次数
        
        参数：
            complexity: 决策复杂度 (0-1)
            estimated_decisions_remaining: 估计的剩余决策次数
        
        返回：
            分配的仿真次数
        """
        # 1. 计算平均可用时间
        avg_time_per_decision = self.remaining_time / max(estimated_decisions_remaining, 1)
        
        # 2. 估算每次仿真的平均时间（基于历史数据）
        if len(self.time_history) > 0:
            recent_times = self.time_history[-5:]  # 最近5次
            avg_sim_time = np.mean(recent_times) / self.base_simulations
        else:
            avg_sim_time = 0.1  # 初始估计：每次仿真0.1秒
        
        # 3. 基础分配：根据复杂度调整
        complexity_multiplier = 0.6 + complexity * 1.0  # 0.6-1.6x
        base_allocation = int(self.base_simulations * complexity_multiplier)
        
        # 4. 时间约束：确保不超时
        time_constrained_sims = int(avg_time_per_decision / avg_sim_time * 0.8)  # 留20%余量
        
        # 5. 综合决策
        allocated_sims = min(base_allocation, time_constrained_sims)
        allocated_sims = np.clip(allocated_sims, self.min_simulations, self.max_simulations)
        
        # 6. 紧急模式：时间严重不足
        if self.remaining_time < 20.0:
            allocated_sims = max(self.min_simulations, int(allocated_sims * 0.6))
            print(f"[TimeManager] ⚠️  紧急模式：时间不足 {self.remaining_time:.1f}s")
        elif self.remaining_time < 10.0:
            allocated_sims = self.min_simulations
            print(f"[TimeManager] 🚨 极度紧急：仅剩 {self.remaining_time:.1f}s，使用最小仿真")
        
        return int(allocated_sims)

    def estimate_remaining_decisions(self, balls, my_targets):
        """
        估计还需要多少次决策
        
        基于：
        1. 剩余目标球数量
        2. 平均每球需要的击球次数（考虑失误）
        """
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        
        n_targets = len(target_ids)
        
        # 假设平均成功率70%，每个球需要1.4次击球
        # 加上对手的回合
        estimated = int(n_targets * 1.4 * 1.5)  # 1.5倍考虑回合切换
        
        return max(estimated, 3)  # 至少预留3次决策

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
        """计算击球难度分数（越低越容易）"""
        phi, dist_cue_to_obj, dist_obj_to_pocket = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
        
        # 距离惩罚
        distance_penalty = dist_cue_to_obj * 0.3 + dist_obj_to_pocket * 0.5
        
        # 角度因素
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
        """生成战略性候选动作，优先考虑容易进球的方案"""
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']

        # 计算所有可能的 (球, 袋口) 组合的难度
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
        
        # 按难度排序
        shot_options.sort(key=lambda x: x['difficulty'])
        
        # 为最容易的击球生成动作
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
        """围绕一个有希望的动作生成细化变种"""
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
        
        # 检测新游戏并自动重置时间预算
        if self.detect_new_game(balls):
            self.reset_time_budget()
        
        # 开始计时
        decision_start_time = time.time()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 1. 评估决策复杂度
        complexity = self.estimate_decision_complexity(balls, my_targets, table)
        self.complexity_history.append(complexity)
        
        # 2. 估计剩余决策次数
        estimated_remaining = self.estimate_remaining_decisions(balls, my_targets)
        
        # 3. 动态分配仿真次数
        n_simulations = self.allocate_simulations(complexity, estimated_remaining)
        
        print(f"\n[TimeManager] 决策 #{self.decision_count + 1}")
        print(f"  复杂度: {complexity:.2f} | 剩余决策: ~{estimated_remaining}")
        print(f"  分配仿真: {n_simulations} | 剩余时间: {self.remaining_time:.1f}s")
        
        # 4. 生成候选动作
        candidate_actions = self.generate_strategic_actions(balls, my_targets, table, n_actions=15)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # 自适应探索系数
        n_remaining = len([bid for bid in my_targets if balls[bid].state.s != 4])
        adaptive_c_puct = self.base_c_puct * (1.0 + 0.1 * n_remaining)
        
        # 5. 第一阶段MCTS
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

        # 6. 第二阶段：细化（如果时间和质量允许）
        avg_rewards = Q / (N + 1e-6)
        promising_indices = np.where(avg_rewards >= self.refinement_threshold)[0]
        
        # 时间紧急时跳过细化
        should_refine = (len(promising_indices) > 0) and (self.remaining_time > 15.0)
        
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
        
        # 7. 更新时间统计
        decision_time = time.time() - decision_start_time
        self.remaining_time -= decision_time
        self.time_history.append(decision_time)
        self.decision_count += 1
        
        print(f"  最佳分数: {best_score:.3f}")
        print(f"  决策用时: {decision_time:.2f}s | 剩余: {self.remaining_time:.1f}s")
        
        return best_action

