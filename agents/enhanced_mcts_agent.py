import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import random
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


class EnhancedMCTSAgent(Agent):
    """
    增强型MCTS Agent - 使用渐进式动作优化和多层次评估
    
    主要改进：
    1. 渐进式动作生成：从粗粒度搜索开始，逐步细化有希望的动作
    2. 多层次评估：结合即时奖励和位置质量
    3. 自适应探索：根据局面调整探索参数
    4. 优先级启发式：优先搜索成功概率高的球和袋口
    5. 动作聚类：避免重复搜索相似的动作
    """
    
    def __init__(self,
                 n_simulations=50,
                 base_c_puct=1.414,
                 refinement_threshold=0.6,
                 position_weight=0.3):
        super().__init__()
        self.n_simulations = n_simulations
        self.base_c_puct = base_c_puct
        self.refinement_threshold = refinement_threshold  # 触发细化的分数阈值
        self.position_weight = position_weight  # 位置质量权重
        self.ball_radius = 0.028575
        
        # 噪声水平
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        print(f"EnhancedMCTSAgent 已初始化 (sims={n_simulations}, refinement={refinement_threshold})")

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

    def calculate_shot_difficulty(self, cue_pos, obj_pos, pocket_pos):
        """
        计算击球难度分数（越低越容易）
        
        考虑因素：
        1. 白球到目标球的距离
        2. 目标球到袋口的距离
        3. 角度偏差（是否需要大角度切球）
        """
        phi, dist_cue_to_obj, dist_obj_to_pocket = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
        
        # 距离惩罚
        distance_penalty = dist_cue_to_obj * 0.3 + dist_obj_to_pocket * 0.5
        
        # 角度因素：计算是否需要切球
        vec_cue_to_obj = np.array(obj_pos) - np.array(cue_pos)
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        
        if np.linalg.norm(vec_cue_to_obj) > 0 and np.linalg.norm(vec_obj_to_pocket) > 0:
            cos_angle = np.dot(vec_cue_to_obj, vec_obj_to_pocket) / (
                np.linalg.norm(vec_cue_to_obj) * np.linalg.norm(vec_obj_to_pocket))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angle_penalty = abs(angle) * 0.5  # 角度越大越难
        else:
            angle_penalty = 0
        
        difficulty = distance_penalty + angle_penalty
        return difficulty

    def generate_strategic_actions(self, balls, my_targets, table, n_actions=20):
        """
        生成战略性候选动作，优先考虑容易进球的方案
        
        改进点：
        1. 按难度排序，优先生成简单的击球
        2. 为每个目标球只选择最好的1-2个袋口
        3. 生成多种力度和角度变化
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球
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
                difficulty = self.calculate_shot_difficulty(cue_pos, obj_pos, pocket_pos)
                phi, dist, _ = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                
                shot_options.append({
                    'target_id': tid,
                    'pocket_id': pocket_id,
                    'difficulty': difficulty,
                    'phi': phi,
                    'distance': dist
                })
        
        # 按难度排序，优先处理简单的球
        shot_options.sort(key=lambda x: x['difficulty'])
        
        # 为前N个最容易的击球生成动作
        for i, shot_opt in enumerate(shot_options[:min(10, len(shot_options))]):
            phi_ideal = shot_opt['phi']
            dist = shot_opt['distance']
            
            # 基础力度
            v_base = np.clip(1.5 + dist * 1.2, 1.0, 5.5)
            
            # 为每个击球生成多个变种
            variations = [
                {'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0},
                {'V0': v_base + 1.0, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0},
                {'V0': v_base, 'phi': (phi_ideal + 0.3) % 360, 'theta': 0, 'a': 0, 'b': 0},
                {'V0': v_base, 'phi': (phi_ideal - 0.3) % 360, 'theta': 0, 'a': 0, 'b': 0},
            ]
            
            # 只为最简单的几个球生成更多变种
            if i < 3:
                variations.extend([
                    {'V0': v_base + 0.5, 'phi': phi_ideal, 'theta': 0, 'a': 0.1, 'b': 0},
                    {'V0': v_base + 0.5, 'phi': phi_ideal, 'theta': 0, 'a': -0.1, 'b': 0},
                ])
            
            actions.extend(variations)
        
        # 限制动作数量
        if len(actions) > n_actions:
            actions = actions[:n_actions]
        
        # 如果没有生成足够的动作，补充随机动作
        while len(actions) < 10:
            actions.append(self._random_action())
        
        return actions

    def refine_action(self, action, n_refinements=4):
        """
        围绕一个有希望的动作生成细化变种
        
        参数：
            action: 基础动作
            n_refinements: 生成的细化动作数量
        
        返回：
            细化后的动作列表
        """
        refined_actions = [action]  # 包含原始动作
        
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
            # 注入高斯噪声
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
        """
        综合评估击球质量
        
        结合：
        1. 即时奖励（进球、犯规等）
        2. 位置质量（击球后的球位）
        """
        if shot is None:
            return -500.0
        
        # 即时奖励
        immediate_reward = analyze_shot_for_reward(shot, last_state, my_targets)
        
        # 位置质量（只在没有严重犯规时考虑）
        if immediate_reward > -100:
            position_score = evaluate_position_quality(shot, my_targets, table)
            # 位置质量归一化到 [-50, 50] 范围
            position_bonus = position_score * 50 * self.position_weight
        else:
            position_bonus = 0
        
        total_score = immediate_reward + position_bonus
        return total_score

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None:
            return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 第一阶段：生成初始候选动作
        candidate_actions = self.generate_strategic_actions(balls, my_targets, table, n_actions=15)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)  # 访问次数
        Q = np.zeros(n_candidates)  # 累积奖励
        
        # 自适应探索系数：目标球越少，探索越少
        n_remaining = len([bid for bid in my_targets if balls[bid].state.s != 4])
        adaptive_c_puct = self.base_c_puct * (1.0 + 0.1 * n_remaining)
        
        # 第一阶段MCTS：探索初始动作空间
        initial_sims = int(self.n_simulations * 0.6)
        
        for i in range(initial_sims):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                ucb_values = (Q / (N + 1e-6)) + adaptive_c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            raw_reward = self.evaluate_shot(shot, last_state_snapshot, my_targets, table)
            normalized_reward = (raw_reward - (-500)) / 700.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward

        # 第二阶段：识别有希望的动作并细化
        avg_rewards = Q / (N + 1e-6)
        promising_indices = np.where(avg_rewards >= self.refinement_threshold)[0]
        
        # 如果有表现好的动作，进行细化
        if len(promising_indices) > 0:
            # 选择前2个最好的动作进行细化
            top_k = min(2, len(promising_indices))
            top_indices = promising_indices[np.argsort(-avg_rewards[promising_indices])[:top_k]]
            
            refined_actions = []
            for idx in top_indices:
                refined = self.refine_action(candidate_actions[idx], n_refinements=3)
                refined_actions.extend(refined)
            
            # 为细化后的动作分配剩余的模拟次数
            n_refined = len(refined_actions)
            N_refined = np.zeros(n_refined)
            Q_refined = np.zeros(n_refined)
            
            remaining_sims = self.n_simulations - initial_sims
            
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
            
            # 比较细化后的最佳动作与原始最佳动作
            avg_refined = Q_refined / (N_refined + 1e-6)
            best_refined_idx = np.argmax(avg_refined)
            best_refined_score = avg_refined[best_refined_idx]
            
            best_original_idx = np.argmax(avg_rewards)
            best_original_score = avg_rewards[best_original_idx]
            
            if best_refined_score > best_original_score:
                best_action = refined_actions[best_refined_idx]
                best_score = best_refined_score
                print(f"[EnhancedMCTS] Best Score: {best_score:.3f} (Refined, Sims: {self.n_simulations})")
            else:
                best_action = candidate_actions[best_original_idx]
                best_score = best_original_score
                print(f"[EnhancedMCTS] Best Score: {best_score:.3f} (Original, Sims: {self.n_simulations})")
        else:
            # 没有特别好的动作，直接选择最好的原始动作
            best_idx = np.argmax(avg_rewards)
            best_action = candidate_actions[best_idx]
            best_score = avg_rewards[best_idx]
            print(f"[EnhancedMCTS] Best Score: {best_score:.3f} (No Refinement, Sims: {self.n_simulations})")
        
        return best_action

