#!/usr/bin/env python3
"""
测试 AdaptiveTimeMCTSAgent 的时间管理功能

此脚本演示：
1. 自动游戏检测与时间重置
2. 动态仿真次数分配
3. 时间使用情况统计
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import AdaptiveTimeMCTSAgent, BasicAgentPro
from poolenv import PoolEnv


def test_time_management(n_games=3):
    """
    测试时间管理功能
    
    参数：
        n_games: 测试游戏局数
    """
    print("\n" + "="*80)
    print("AdaptiveTimeMCTSAgent 时间管理测试")
    print("="*80)
    
    # 初始化 agents
    time_agent = AdaptiveTimeMCTSAgent(
        base_simulations=50,
        total_time_budget=180.0,  # 3分钟
        min_simulations=20,
        max_simulations=150,
    )
    
    basic_agent = BasicAgentPro(n_simulations=50)
    
    env = PoolEnv()
    
    game_stats = []
    
    for game_num in range(n_games):
        print(f"\n{'#'*80}")
        print(f"第 {game_num + 1}/{n_games} 局游戏")
        print(f"{'#'*80}\n")
        
        # 重置环境
        env.reset()
        
        # 记录本局统计
        game_stat = {
            'game_num': game_num + 1,
            'decisions': 0,
            'total_time': 0,
            'complexities': [],
            'simulations': [],
            'decision_times': [],
        }
        
        # 游戏循环
        step_count = 0
        max_steps = 200
        
        while step_count < max_steps:
            step_count += 1
            
            # 获取当前玩家
            current_player = env.get_curr_player()
            obs = env.get_observation(current_player)
            balls, my_targets, table = obs
            
            # 选择 agent (假设 AdaptiveTimeMCTSAgent 是玩家 A)
            if current_player == 'A':
                decision_start = time.time()
                action = time_agent.decision(balls, my_targets, table)
                decision_time = time.time() - decision_start
                
                # 记录统计
                game_stat['decisions'] += 1
                game_stat['total_time'] += decision_time
                game_stat['decision_times'].append(decision_time)
                if len(time_agent.complexity_history) > 0:
                    game_stat['complexities'].append(time_agent.complexity_history[-1])
            else:
                action = basic_agent.decision(balls, my_targets, table)
            
            # 执行动作
            env.take_shot(action)
            
            # 检查游戏是否结束
            done, info = env.get_done()
            if done:
                winner = info.get('winner', 'SAME')
                print(f"\n游戏结束！获胜者: {winner}")
                break
        
        # 本局统计
        game_stat['remaining_time'] = time_agent.remaining_time
        game_stat['avg_complexity'] = np.mean(game_stat['complexities']) if game_stat['complexities'] else 0
        game_stat['avg_decision_time'] = np.mean(game_stat['decision_times']) if game_stat['decision_times'] else 0
        
        game_stats.append(game_stat)
        
        # 打印本局统计
        print(f"\n{'='*80}")
        print(f"第 {game_num + 1} 局统计")
        print(f"{'='*80}")
        print(f"决策次数: {game_stat['decisions']}")
        print(f"总用时: {game_stat['total_time']:.2f}s")
        print(f"平均决策时间: {game_stat['avg_decision_time']:.2f}s")
        print(f"平均复杂度: {game_stat['avg_complexity']:.3f}")
        print(f"剩余时间: {game_stat['remaining_time']:.2f}s")
        print(f"时间利用率: {(180 - game_stat['remaining_time']) / 180 * 100:.1f}%")
    
    # 总体统计
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")
    
    total_decisions = sum(s['decisions'] for s in game_stats)
    total_time = sum(s['total_time'] for s in game_stats)
    avg_remaining = np.mean([s['remaining_time'] for s in game_stats])
    
    print(f"测试局数: {n_games}")
    print(f"总决策次数: {total_decisions}")
    print(f"总用时: {total_time:.2f}s")
    print(f"平均每局决策次数: {total_decisions / n_games:.1f}")
    print(f"平均每局用时: {total_time / n_games:.2f}s")
    print(f"平均剩余时间: {avg_remaining:.2f}s")
    print(f"平均时间利用率: {(180 - avg_remaining) / 180 * 100:.1f}%")
    
    # 检查是否有任何局超时
    overtime_games = [s for s in game_stats if s['remaining_time'] < 0]
    if overtime_games:
        print(f"\n⚠️  警告：{len(overtime_games)} 局游戏超时！")
        for s in overtime_games:
            print(f"  第 {s['game_num']} 局超时 {-s['remaining_time']:.2f}s")
    else:
        print(f"\n✓ 所有游戏均未超时")
    
    # 时间分布分析
    print(f"\n{'='*80}")
    print("决策时间分布")
    print(f"{'='*80}")
    
    all_times = []
    for s in game_stats:
        all_times.extend(s['decision_times'])
    
    if all_times:
        print(f"最快决策: {min(all_times):.2f}s")
        print(f"最慢决策: {max(all_times):.2f}s")
        print(f"中位数: {np.median(all_times):.2f}s")
        print(f"第25百分位: {np.percentile(all_times, 25):.2f}s")
        print(f"第75百分位: {np.percentile(all_times, 75):.2f}s")
    
    return game_stats


def compare_with_fixed_simulations(n_games=5):
    """
    对比自适应时间管理 vs 固定仿真次数
    
    测试：在相同时间预算下，哪个策略更好
    """
    print("\n" + "="*80)
    print("自适应时间管理 vs 固定仿真次数对比")
    print("="*80)
    
    agents = {
        'Adaptive': AdaptiveTimeMCTSAgent(
            base_simulations=50,
            total_time_budget=180.0,
            min_simulations=20,
            max_simulations=150,
        ),
        'Fixed-50': BasicAgentPro(n_simulations=50),
    }
    
    env = PoolEnv()
    results = {'Adaptive': 0, 'Fixed-50': 0, 'Draw': 0}
    
    for game_num in range(n_games):
        print(f"\n{'#'*80}")
        print(f"第 {game_num + 1}/{n_games} 局")
        print(f"{'#'*80}")
        
        env.reset()
        
        # Adaptive 先手/后手轮换
        if game_num % 2 == 0:
            players = ['Adaptive', 'Fixed-50']
        else:
            players = ['Fixed-50', 'Adaptive']
        
        print(f"玩家 A: {players[0]} | 玩家 B: {players[1]}")
        
        # 游戏循环
        step_count = 0
        max_steps = 200
        
        while step_count < max_steps:
            step_count += 1
            
            current_player = env.get_curr_player()
            obs = env.get_observation(current_player)
            
            if current_player == 'A':
                agent_name = players[0]
            else:
                agent_name = players[1]
            
            action = agents[agent_name].decision(*obs)
            env.take_shot(action)
            
            done, info = env.get_done()
            if done:
                winner = info.get('winner', 'SAME')
                print(f"\n游戏结束！获胜者: {winner}")
                
                # 记录结果
                if winner == 'SAME':
                    results['Draw'] += 1
                elif winner == 'A':
                    results[players[0]] += 1
                else:
                    results[players[1]] += 1
                break
    
    # 打印结果
    print(f"\n{'='*80}")
    print("对比结果")
    print(f"{'='*80}")
    print(f"总局数: {n_games}")
    print(f"Adaptive 获胜: {results['Adaptive']} ({results['Adaptive']/n_games*100:.1f}%)")
    print(f"Fixed-50 获胜: {results['Fixed-50']} ({results['Fixed-50']/n_games*100:.1f}%)")
    print(f"平局: {results['Draw']} ({results['Draw']/n_games*100:.1f}%)")
    
    if results['Adaptive'] > results['Fixed-50']:
        print(f"\n✓ Adaptive 策略表现更好！")
    elif results['Adaptive'] < results['Fixed-50']:
        print(f"\n✗ Fixed-50 策略表现更好")
    else:
        print(f"\n= 两种策略打平")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 AdaptiveTimeMCTSAgent")
    parser.add_argument('--mode', choices=['time', 'compare', 'both'], 
                        default='time',
                        help='测试模式：time=时间管理测试, compare=对比测试, both=两者都测')
    parser.add_argument('--games', type=int, default=3,
                        help='测试局数')
    args = parser.parse_args()
    
    try:
        if args.mode in ['time', 'both']:
            test_time_management(n_games=args.games)
        
        if args.mode in ['compare', 'both']:
            compare_with_fixed_simulations(n_games=args.games)
        
        print("\n" + "="*80)
        print("测试完成！")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

