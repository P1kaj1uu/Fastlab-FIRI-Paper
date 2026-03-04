import os
import pickle
import numpy as np
import time
import json
from datetime import datetime
from performance_evaluator import PerformanceEvaluator
from obstacle_generator import ObstacleSet, place_obstacles, save_obstacles_to_file
from path_planner import generate_initial_waypoints, calculate_path_length
from visualizer import visualize_results, visualize_with_open3d
from utils import analyze_path_smoothness, check_collisions, analyze_path_results
from firi.planning.planner import FIRIPlanner
from firi.utils.obstacle_generator import ObstacleGenerator

def clean_temp_dir():
    """清理临时目录"""
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        # 清理临时文件
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    print("Temp directory cleaned")

def main():
    """
    主函数，执行路径规划并可视化结果
    """
    # 创建性能评估器
    evaluator = PerformanceEvaluator()

    # 清理临时目录
    evaluator.start_timer("clean_temp_dir")
    clean_temp_dir()
    evaluator.stop_timer("clean_temp_dir")

    # 定义起点和终点
    start_point = np.array([1.0, 1.0, 1.0])
    goal_point = np.array([9.0, 9.0, 9.0])

    # 定义空间边界
    space_bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

    # 放置障碍物
    evaluator.start_timer("obstacles_generation")
    obstacles = ObstacleSet()

    # 使用随机生成的障碍物
    obstacle_data = place_obstacles([[0, 10], [0, 10], [0, 10]], n=16)

    for obs in obstacle_data.obstacle_list:
        if obs.shape == 'sphere':
            obstacles.add_obstacle('sphere', obs.center, radius=obs.radius)
        elif obs.shape == 'cylinder':
            obstacles.add_obstacle('cylinder', obs.center, radius=obs.radius, height=obs.height)
        elif obs.shape == 'cuboid':
            obstacles.add_obstacle('cuboid', obs.center, size=obs.size)

    evaluator.record_value("obstacles_count", len(obstacles.obstacle_list))
    evaluator.stop_timer("obstacles_generation")

    # 保存障碍物
    save_obstacles_to_file(obstacles)

    # 创建膨胀障碍物(用于规划和可视化)
    evaluator.start_timer("obstacles_inflation")
    safety_margin = 1.5  # 安全边界系数
    evaluator.stop_timer("obstacles_inflation")

    # 创建FIRI规划器
    evaluator.start_timer("planner_initialization")
    planner = FIRIPlanner(obstacles=obstacles, space_size=(10, 10, 10))
    evaluator.stop_timer("planner_initialization")

    # 生成更多初始路径点，以提高规划成功率
    evaluator.start_timer("initial_waypoints_generation")
    num_waypoints = 15  # 增加到15个点
    initial_waypoints = generate_initial_waypoints(start_point, goal_point, num_waypoints=num_waypoints, jitter=2.0)
    evaluator.record_value("waypoints_count", len(initial_waypoints))
    evaluator.stop_timer("initial_waypoints_generation")

    print("生成带扰动的初始路径点")

    # 使用更安全的参数进行路径规划
    print("规划路径...")
    try:
        # 设置最大重规划次数和安全边界
        evaluator.start_timer("path_planning")
        final_path = planner.plan_path(
            start_point,
            goal_point,
            initial_waypoints=None,
            smoothing=True,
            max_replanning_attempts=7,  # 增加重规划次数
            safety_margin=safety_margin  # 使用相同的安全边界参数
        )
        evaluator.stop_timer("path_planning")

        if final_path is not None:
            evaluator.record_value("path_points_count", len(final_path))
            evaluator.record_value("path_length", calculate_path_length(final_path))
            print(f"Saved path with {len(final_path)} points to file")

            # 应用进一步平滑，使用更强的角度限制和安全性检查
            print("\n应用B样条平滑以降低曲率...")
            evaluator.start_timer("path_smoothing")
            smoothed_path = planner.bspline_smooth(
                final_path,
                smoothing_factor=0.4  # 可以根据实验进行调整，越大越平滑
            )
            evaluator.stop_timer("path_smoothing")

            # 检查平滑后的路径是否安全
            evaluator.start_timer("collision_checking")
            collisions = 0
            for i in range(len(smoothed_path)):
                for obs in obstacles.obstacle_list:
                    obs_center = np.array(obs.center)
                    dist = np.linalg.norm(smoothed_path[i] - obs_center)
                    if dist < obs.radius * 1.05:  # 添加5%的安全余量
                        collisions += 1
            evaluator.stop_timer("collision_checking")

            if collisions > 0:
                print(f"平滑后路径不安全（{collisions}处碰撞），尝试修复...")
                # 尝试修复碰撞点
                evaluator.start_timer("collision_fixing")
                fixed_path = smoothed_path.copy()
                fix_iterations = 0
                for _ in range(5):  # 最多尝试5次修复
                    fix_iterations += 1
                    collision_count = 0
                    for i in range(len(fixed_path)):
                        for obs in obstacles.obstacle_list:
                            obs_center = np.array(obs.center)
                            obs_radius = obs.radius * 1.1  # 10%的安全余量
                            dist = np.linalg.norm(fixed_path[i] - obs_center)
                            if dist < obs_radius:
                                collision_count += 1
                                # 移动点远离障碍物
                                direction = fixed_path[i] - obs_center
                                if np.linalg.norm(direction) > 1e-6:
                                    direction = direction / np.linalg.norm(direction)
                                    fixed_path[i] = obs_center + direction * (obs_radius + 0.3)

                    if collision_count == 0:
                        print(f"碰撞修复成功！")
                        smoothed_path = fixed_path
                        break
                    else:
                        print(f"仍有{collision_count}处碰撞，继续修复...")

                evaluator.record_value("collision_fix_iterations", fix_iterations)
                evaluator.stop_timer("collision_fixing")

            # 保存平滑后的路径
            evaluator.start_timer("saving_results")
            with open('temp/smoothed_path.pkl', 'wb') as f:
                pickle.dump(smoothed_path, f)

            print(f"保存平滑路径，点数: {len(smoothed_path)}")

            # 记录最终路径信息
            evaluator.record_value("smoothed_path_points_count", len(smoothed_path))
            evaluator.record_value("smoothed_path_length", calculate_path_length(smoothed_path))
            evaluator.record_value("final_collisions", collisions)

            # 计算路径平滑度 (平均角度)
            avg_angle = analyze_path_smoothness(smoothed_path)
            evaluator.record_value("path_smoothness", avg_angle)
            evaluator.stop_timer("saving_results")

            # 分析最终路径
            evaluator.start_timer("path_analysis")
            analyze_path_results(final_path, smoothed_path, obstacles)
            evaluator.stop_timer("path_analysis")

            # 使用matplotlib可视化结果
            evaluator.start_timer("matplotlib_visualization")
            visualize_results(smoothed_path, obstacles, space_bounds)
            evaluator.stop_timer("matplotlib_visualization")

            # 使用Open3D进行可视化，包括膨胀障碍物
            print("\n应用B样条平滑以降低曲率...")
            evaluator.start_timer("open3d_visualization")
            visualize_with_open3d(
                smoothed_path,
                obstacles,
                start_point,
                goal_point,
                inflated_obstacles=None,
                safety_margin=safety_margin
            )
            evaluator.stop_timer("open3d_visualization")
            # 保存性能评估结果
            evaluator.save_results()

            return True
        else:
            print("路径规划失败")
            evaluator.record_value("planning_success", False)
            evaluator.save_results()
            return False

    except Exception as e:
        print(f"规划过程出错: {str(e)}")
        evaluator.record_value("error", str(e))
        evaluator.save_results()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n规划结果: {'成功' if success else '失败'}")
