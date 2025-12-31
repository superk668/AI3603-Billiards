#!/usr/bin/env python3
"""
测试 GlobalTimeMCTSAgent 的全局时间管理功能

演示：
1. 跨局时间共享与分配
2. 根据胜负情况调整策略
3. 时间利用率统计
4. 与独立时间预算的对比
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import GlobalTimeMCTSAgent, AdaptiveTimeMCTSAgent, BasicAgentPro
from poolenv import PoolEnv


def test_global_time_management(n_games=10):
    """
    测试全局时间管理功能
    
    参数：
        n_games: 测试游戏局数
    """
    print("\n" + "="*80)
    print("GlobalTimeMCTSAgent 全局时间管理测试")
    print("="*80)
    
    # 初始化 agents
    global_agent = GlobalTimeMCTSAgent(
        n_games=n_games,
        time_per_game=180.0,
        base_simulations=50,
        min_simulations=15,
        max_simulations=200
    )
    
    opponent = BasicAgentPro(n_simulations=50)
    
    env = PoolEnv()
    
    game_stats = []
    
    for game_num in range(n_games):
        print(f"\n{'#'*80}")
        print(f"第 {game_num + 1}/{n_games} 局游戏")
        print(f"{'#'*80}\n")
        
        # 重置环境
        env.reset()
        
        # 记录本局开始时间
        game_start_time = time.time()
        game_decisions = 0
        game_time = 0.0
        
        # 游戏循环
        step_count = 0
        max_steps = 200
        winner = None
        
        while step_count < max_steps:
            step_count += 1
            
            # 获取当前玩家
            current_player = env.get_curr_player()
            obs = env.get_observation(current_player)
            balls, my_targets, table = obs
            
            # 选择 agent (假设 GlobalTimeMCTS 是玩家 A)
            if current_player == 'A':
                decision_start = time.time()
                action = global_agent.decision(balls, my_targets, table)
                decision_time = time.time() - decision_start
                
                game_decisions += 1
                game_time += decision_time
            else:
                action = opponent.decision(balls, my_targets, table)
            
            # 执行动作
            env.take_shot(action)
            
            # 检查游戏是否结束
            done, info = env.get_done()
            if done:
                winner = info.get('winner', 'SAME')
                print(f"\n游戏结束！获胜者: {winner}")
                
                # 报告结果
                won = (winner == 'A')
                global_agent.report_game_result(won)
                break
        
        # 记录本局统计
        game_stat = {
            'game_num': game_num + 1,
            'decisions': game_decisions,
            'time_used': game_time,
            'winner': winner,
            'won': (winner == 'A'),
            'remaining_time': global_agent.remaining_time,
        }
        game_stats.append(game_stat)
        
        # 打印本局统计
        print(f"\n{'='*80}")
        print(f"第 {game_num + 1} 局统计")
        print(f"{'='*80}")
        print(f"决策次数: {game_decisions}")
        print(f"本局用时: {game_time:.2f}s")
        print(f"总剩余时间: {global_agent.remaining_time:.2f}s")
        
        # 获取当前统计
        stats = global_agent.get_statistics()
        print(f"当前战绩: {stats['games_won']}胜 {stats['games_lost']}负")
        print(f"胜率: {stats['win_rate']:.1%}")
        print(f"总时间利用率: {stats['time_utilization']:.1%}")
    
    # 总体统计
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")
    
    final_stats = global_agent.get_statistics()
    
    print(f"游戏局数: {final_stats['games_played']}")
    print(f"战绩: {final_stats['games_won']}胜 {final_stats['games_lost']}负")
    print(f"胜率: {final_stats['win_rate']:.1%}")
    print(f"总决策次数: {final_stats['total_decisions']}")
    print(f"总时间预算: {global_agent.total_time_budget:.0f}s")
    print(f"实际使用: {final_stats['time_used']:.2f}s")
    print(f"剩余时间: {final_stats['remaining_time']:.2f}s")
    print(f"时间利用率: {final_stats['time_utilization']:.1%}")
    
    # 时间分布分析
    print(f"\n{'='*80}")
    print("各局时间分布")
    print(f"{'='*80}")
    
    times = [s['time_used'] for s in game_stats]
    print(f"最快一局: {min(times):.2f}s (第{times.index(min(times))+1}局)")
    print(f"最慢一局: {max(times):.2f}s (第{times.index(max(times))+1}局)")
    print(f"平均每局: {np.mean(times):.2f}s")
    print(f"标准差: {np.std(times):.2f}s")
    
    return game_stats, final_stats


def compare_time_strategies(n_games=10):
    """
    对比全局时间管理 vs 独立时间预算
    
    参数：
        n_games: 测试游戏局数
    """
    print("\n" + "="*80)
    print("全局时间管理 vs 独立时间预算 对比测试")
    print("="*80)
    
    agents = {
        'Global': GlobalTimeMCTSAgent(
            n_games=n_games,
            time_per_game=180.0,
            base_simulations=50,
            min_simulations=15,
            max_simulations=200
        ),
        'Adaptive': AdaptiveTimeMCTSAgent(
            base_simulations=50,
            total_time_budget=180.0,  # 每局独立180秒
            min_simulations=20,
            max_simulations=150
        )
    }
    
    opponent = BasicAgentPro(n_simulations=50)
    env = PoolEnv()
    
    results = {
        'Global': {'wins': 0, 'games': [], 'total_time': 0},
        'Adaptive': {'wins': 0, 'games': [], 'total_time': 0}
    }
    
    for agent_name in ['Global', 'Adaptive']:
        agent = agents[agent_name]
        print(f"\n{'#'*80}")
        print(f"测试 {agent_name} Agent")
        print(f"{'#'*80}")
        
        for game_num in range(n_games):
            print(f"\n第 {game_num + 1}/{n_games} 局")
            env.reset()
            
            game_start = time.time()
            step_count = 0
            max_steps = 200
            
            while step_count < max_steps:
                step_count += 1
                
                current_player = env.get_curr_player()
                obs = env.get_observation(current_player)
                
                if current_player == 'A':
                    action = agent.decision(*obs)
                else:
                    action = opponent.decision(*obs)
                
                env.take_shot(action)
                
                done, info = env.get_done()
                if done:
                    winner = info.get('winner', 'SAME')
                    game_time = time.time() - game_start
                    
                    won = (winner == 'A')
                    if won:
                        results[agent_name]['wins'] += 1
                    
                    results[agent_name]['games'].append({
                        'game_num': game_num + 1,
                        'won': won,
                        'time': game_time
                    })
                    results[agent_name]['total_time'] += game_time
                    
                    # 报告结果（仅 Global 需要）
                    if agent_name == 'Global':
                        agent.report_game_result(won)
                    
                    print(f"  结果: {winner}, 用时: {game_time:.2f}s")
                    break
    
    # 打印对比结果
    print(f"\n{'='*80}")
    print("对比结果")
    print(f"{'='*80}")
    
    for agent_name in ['Global', 'Adaptive']:
        print(f"\n{agent_name} Agent:")
        wins = results[agent_name]['wins']
        total_time = results[agent_name]['total_time']
        games = results[agent_name]['games']
        
        print(f"  胜场: {wins}/{n_games} ({wins/n_games*100:.1f}%)")
        print(f"  总用时: {total_time:.2f}s")
        print(f"  平均每局: {total_time/n_games:.2f}s")
        
        times = [g['time'] for g in games]
        print(f"  最快: {min(times):.2f}s")
        print(f"  最慢: {max(times):.2f}s")
        print(f"  标准差: {np.std(times):.2f}s")
    
    # 特定统计
    if hasattr(agents['Global'], 'get_statistics'):
        global_stats = agents['Global'].get_statistics()
        print(f"\nGlobal Agent 时间利用率: {global_stats['time_utilization']:.1%}")
    
    print(f"\nAdaptive Agent 理论最大用时: {n_games * 180:.0f}s")
    print(f"Adaptive Agent 实际总用时: {results['Adaptive']['total_time']:.2f}s")
    print(f"Adaptive Agent 浪费时间: {n_games * 180 - results['Adaptive']['total_time']:.2f}s")
    
    # 比较
    global_time = results['Global']['total_time']
    adaptive_time = results['Adaptive']['total_time']
    global_wins = results['Global']['wins']
    adaptive_wins = results['Adaptive']['wins']
    
    print(f"\n{'='*80}")
    print("综合评价")
    print(f"{'='*80}")
    
    if global_wins > adaptive_wins:
        print(f"✓ Global 胜率更高 ({global_wins} vs {adaptive_wins})")
    elif global_wins < adaptive_wins:
        print(f"✗ Adaptive 胜率更高 ({adaptive_wins} vs {global_wins})")
    else:
        print(f"= 胜率相同 ({global_wins}胜)")
    
    time_saved = (n_games * 180 - adaptive_time) - (n_games * 180 - global_time)
    print(f"\nGlobal 比 Adaptive 多利用时间: {time_saved:.2f}s")
    print(f"时间利用率提升: {time_saved / (n_games * 180) * 100:.1f}%")
    
    return results


def visualize_time_usage(game_stats):
    """
    可视化时间使用情况（需要 matplotlib）
    """
    try:
        games = [s['game_num'] for s in game_stats]
        times = [s['time_used'] for s in game_stats]
        remaining = [s['remaining_time'] for s in game_stats]
        won = [s['won'] for s in game_stats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 图1：每局用时
        colors = ['green' if w else 'red' for w in won]
        ax1.bar(games, times, color=colors, alpha=0.7)
        ax1.axhline(y=np.mean(times), color='blue', linestyle='--', label=f'平均: {np.mean(times):.1f}s')
        ax1.set_xlabel('Game Number')
        ax1.set_ylabel('Time Used (s)')
        ax1.set_title('Time Usage per Game (Green=Won, Red=Lost)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：剩余时间
        ax2.plot(games, remaining, marker='o', linewidth=2, markersize=6)
        ax2.fill_between(games, remaining, alpha=0.3)
        ax2.set_xlabel('Game Number')
        ax2.set_ylabel('Remaining Time (s)')
        ax2.set_title('Remaining Time Budget')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('global_time_usage.png', dpi=150)
        print("\n图表已保存到: global_time_usage.png")
        
    except ImportError:
        print("\n注意: matplotlib 未安装，跳过可视化")
    except Exception as e:
        print(f"\n可视化出错: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 GlobalTimeMCTSAgent")
    parser.add_argument('--mode', choices=['test', 'compare', 'both'],
                        default='test',
                        help='测试模式')
    parser.add_argument('--games', type=int, default=10,
                        help='测试局数')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图表')
    args = parser.parse_args()
    
    try:
        if args.mode in ['test', 'both']:
            game_stats, final_stats = test_global_time_management(n_games=args.games)
            
            if args.visualize:
                visualize_time_usage(game_stats)
        
        if args.mode in ['compare', 'both']:
            compare_time_strategies(n_games=args.games)
        
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

