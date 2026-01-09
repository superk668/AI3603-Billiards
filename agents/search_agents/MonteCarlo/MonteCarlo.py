"""
MonteCarlo.py - Robust Monte Carlo Tree Search Agent

Key Innovation: Robustness-Focused MCTS
- Multiple noise samples per action to estimate performance variance
- Risk-adjusted selection using mean - k*std metric
- Robust child selection considering both mean and consistency
- Adaptive risk aversion based on game state

This agent is designed to outperform BasicAgentPro by being more resilient
to the noise added during evaluation.
"""

import math
import pooltool as pt
import numpy as np
import copy
import random
from typing import Dict, List, Tuple, Optional

from agents.agent import Agent


class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list) -> float:
    """
    分析击球结果并计算奖励分数
    
    参数:
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态
        player_targets: 当前玩家目标球ID列表
    
    返回:
        float: 奖励分数 (范围约 -500 到 +150)
    """
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() 
                    if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed 
                      if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', 
                      '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定
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

    if (len(new_pocketed) == 0 and first_contact_ball_id is not None 
        and (not cue_hit_cushion) and (not target_hit_cushion)):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 500  # 严重犯规
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        if player_targets == ['8']:
            score += 150  # 合法打进黑8
        else:
            score -= 500  # 清台前误打黑8
            
    # 犯规扣分
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if (score == 0 and not cue_pocketed and not eight_pocketed 
        and not foul_first_hit and not foul_no_rail):
        score = 10
        
    return score


class MCTSAgent(Agent):
    """
    鲁棒的蒙特卡洛树搜索 Agent
    
    核心创新：
    1. 多次噪声采样 - 每个动作测试多次以评估稳定性
    2. 风险调整选择 - 使用 mean - k*std 而非仅均值
    3. 鲁棒子节点选择 - 选择既高分又稳定的动作
    4. 自适应风险规避 - 根据游戏状态调整风险参数
    """
    
    def __init__(self, 
                 n_simulations: int = 60,           # MCTS 总模拟次数
                 n_noise_samples: int = 5,          # 每个动作的噪声采样次数
                 c_puct: float = 1.414,             # UCB 探索系数
                 risk_aversion: float = 0.5):       # 风险规避系数 (k in mean-k*std)
        """
        初始化鲁棒 MCTS Agent
        
        参数:
            n_simulations: MCTS 总迭代次数
            n_noise_samples: 每个候选动作的噪声采样数量
            c_puct: UCB 探索系数
            risk_aversion: 风险规避系数 (越大越保守)
        """
        super().__init__()
        self.n_simulations = n_simulations
        self.n_noise_samples = n_noise_samples
        self.c_puct = c_puct
        self.risk_aversion = risk_aversion
        self.ball_radius = 0.028575
        
        # 噪声水平 (与 poolenv 一致)
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        
        print(f"[MCTSAgent] 初始化完成 - 鲁棒性优化版本")
        print(f"  模拟次数: {n_simulations}, 噪声采样: {n_noise_samples}, "
              f"风险规避: {risk_aversion}")

    def _calc_angle_degrees(self, v: np.ndarray) -> float:
        """计算向量角度（度）"""
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos: np.ndarray, 
                               obj_pos: np.ndarray, 
                               pocket_pos: np.ndarray) -> Tuple[float, float]:
        """
        计算瞄准角度（鬼球法）
        
        返回: (phi角度, 距离)
        """
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        
        if dist_obj_to_pocket < 1e-6:
            return 0.0, 0.0
        
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        
        return phi, dist_cue_to_ghost

    def generate_candidate_actions(self, balls: Dict, my_targets: List[str], 
                                   table) -> List[Dict]:
        """
        生成候选动作列表（基于几何启发式）
        
        返回: 候选动作列表
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        
        cue_pos = cue_ball.state.rvw[0]
        
        # 获取目标球
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        
        # 为每个目标球生成候选动作
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]
            
            # 为每个袋口生成动作
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                
                # 基于距离估算力度
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)
                
                # 生成多个变种
                # 1. 标准打法
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                
                # 2. 强力打法
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 
                    'phi': phi_ideal, 
                    'theta': 0, 
                    'a': 0, 
                    'b': 0
                })
                
                # 3-4. 角度微调（应对噪声）
                for angle_offset in [0.5, -0.5]:
                    actions.append({
                        'V0': v_base,
                        'phi': (phi_ideal + angle_offset) % 360,
                        'theta': 0,
                        'a': 0,
                        'b': 0
                    })
                
                # 5-6. 力度微调
                for v_offset in [0.8, -0.8]:
                    actions.append({
                        'V0': np.clip(v_base + v_offset, 0.5, 8.0),
                        'phi': phi_ideal,
                        'theta': 0,
                        'a': 0,
                        'b': 0
                    })
        
        # 如果没有生成任何动作，添加随机动作
        if len(actions) == 0:
            for _ in range(10):
                actions.append(self._random_action())
        
        # 打乱并限制数量
        random.shuffle(actions)
        return actions[:25]  # 限制候选数量以提高效率

    def simulate_action_with_noise(self, balls: Dict, table, 
                                   action: Dict) -> Optional[pt.System]:
        """
        执行带噪声的物理仿真
        
        返回: 仿真后的 System 对象，失败返回 None
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # 注入高斯噪声
            noisy_V0 = np.clip(
                action['V0'] + np.random.normal(0, self.noise_std['V0']), 
                0.5, 8.0
            )
            noisy_phi = (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360
            noisy_theta = np.clip(
                action['theta'] + np.random.normal(0, self.noise_std['theta']), 
                0, 90
            )
            noisy_a = np.clip(
                action['a'] + np.random.normal(0, self.noise_std['a']), 
                -0.5, 0.5
            )
            noisy_b = np.clip(
                action['b'] + np.random.normal(0, self.noise_std['b']), 
                -0.5, 0.5
            )
            
            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, 
                         a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def evaluate_action_robustness(self, balls: Dict, table, action: Dict,
                                   last_state: Dict, my_targets: List[str]) -> Dict:
        """
        评估动作的鲁棒性（核心创新）
        
        多次采样不同噪声，计算：
        - mean_reward: 平均奖励
        - std_reward: 奖励标准差
        - min_reward: 最差情况
        - risk_adjusted_reward: mean - k*std
        
        返回: 包含统计信息的字典
        """
        rewards = []
        
        for _ in range(self.n_noise_samples):
            shot = self.simulate_action_with_noise(balls, table, action)
            
            if shot is None:
                reward = -500.0
            else:
                reward = analyze_shot_for_reward(shot, last_state, my_targets)
            
            rewards.append(reward)
        
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        # 风险调整奖励 = 均值 - k * 标准差
        risk_adjusted_reward = mean_reward - self.risk_aversion * std_reward
        
        return {
            'mean': mean_reward,
            'std': std_reward,
            'min': min_reward,
            'max': max_reward,
            'risk_adjusted': risk_adjusted_reward,
            'samples': len(rewards)
        }

    def decision(self, balls: Optional[Dict] = None, 
                my_targets: Optional[List[str]] = None, 
                table = None) -> Dict:
        """
        使用鲁棒 MCTS 进行决策
        
        参数:
            balls: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回:
            击球动作字典
        """
        if balls is None:
            return self._random_action()
        
        try:
            # 预处理：检查是否需要打黑8
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining) == 0:
                my_targets = ["8"]
                print("[MCTSAgent] 目标球已清空，切换至黑8")
            
            # 保存初始状态
            last_state_snapshot = {bid: copy.deepcopy(ball) 
                                  for bid, ball in balls.items()}
            
            # 生成候选动作
            candidate_actions = self.generate_candidate_actions(balls, my_targets, table)
            n_candidates = len(candidate_actions)
            
            print(f"[MCTSAgent] 生成 {n_candidates} 个候选动作")
            
            # MCTS 统计数组
            N = np.zeros(n_candidates)  # 访问次数
            Q_sum = np.zeros(n_candidates)  # 累计奖励
            Q_sum_sq = np.zeros(n_candidates)  # 累计奖励平方（用于计算方差）
            
            # MCTS 主循环
            for iteration in range(self.n_simulations):
                # Selection: 使用 UCB1
                if iteration < n_candidates:
                    # 初始阶段：每个动作至少尝试一次
                    idx = iteration
                else:
                    # 探索-利用平衡
                    total_n = np.sum(N)
                    mean_q = Q_sum / (N + 1e-9)
                    
                    # 计算 UCB 值
                    exploration_bonus = self.c_puct * np.sqrt(
                        np.log(total_n + 1) / (N + 1e-9)
                    )
                    ucb_values = mean_q + exploration_bonus
                    
                    idx = np.argmax(ucb_values)
                
                # Simulation & Evaluation: 鲁棒性评估
                robustness_stats = self.evaluate_action_robustness(
                    balls, table, candidate_actions[idx], 
                    last_state_snapshot, my_targets
                )
                
                # 使用风险调整奖励进行更新
                reward = robustness_stats['risk_adjusted']
                
                # Backpropagation: 更新统计
                N[idx] += 1
                Q_sum[idx] += reward
                Q_sum_sq[idx] += reward ** 2
            
            # Final Selection: 鲁棒子节点选择
            # 计算每个动作的平均奖励和标准差
            mean_rewards = Q_sum / (N + 1e-9)
            var_rewards = (Q_sum_sq / (N + 1e-9)) - (mean_rewards ** 2)
            std_rewards = np.sqrt(np.maximum(var_rewards, 0))
            
            # 使用风险调整分数选择最终动作
            risk_adjusted_scores = mean_rewards - self.risk_aversion * std_rewards
            
            # 选择风险调整分数最高的动作
            best_idx = np.argmax(risk_adjusted_scores)
            best_action = candidate_actions[best_idx]
            
            # 打印决策信息
            print(f"[MCTSAgent] 决策完成:")
            print(f"  最佳动作 (idx={best_idx}):")
            print(f"    平均奖励: {mean_rewards[best_idx]:.2f}")
            print(f"    标准差: {std_rewards[best_idx]:.2f}")
            print(f"    风险调整分数: {risk_adjusted_scores[best_idx]:.2f}")
            print(f"    访问次数: {int(N[best_idx])}")
            print(f"  动作参数: V0={best_action['V0']:.2f}, "
                  f"phi={best_action['phi']:.2f}, "
                  f"theta={best_action['theta']:.2f}")
            
            return best_action
            
        except Exception as e:
            print(f"[MCTSAgent] 决策出错: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

    def _random_action(self) -> Dict:
        """生成随机击球动作"""
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),
            'phi': round(random.uniform(0, 360), 2),
            'theta': round(random.uniform(0, 90), 2),
            'a': round(random.uniform(-0.5, 0.5), 3),
            'b': round(random.uniform(-0.5, 0.5), 3)
        }
        return action


