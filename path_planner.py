import numpy as np
import pickle

def generate_initial_waypoints(start, goal, num_waypoints=5, jitter=2.0):
    """生成初始路径点，添加扰动以绕过障碍物"""
    waypoints = [start]

    # 计算起点到终点的向量
    direction = goal - start

    # 生成中间点
    for i in range(1, num_waypoints - 1):
        t = i / (num_waypoints - 1)

        # 基础位置（在起点和终点连线上）
        base_point = start + t * direction

        # 添加垂直于主方向的随机扰动
        # 创建垂直于主方向的扰动（通过创建随机向量然后正交化）
        random_vec = np.random.randn(3)
        random_vec = random_vec - np.dot(random_vec, direction) * direction / np.dot(direction, direction)
        random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-10)  # 归一化

        # 计算扰动大小（在路径中间点处最大）
        disturbance_factor = jitter * np.sin(np.pi * t)

        # 应用扰动
        waypoint = base_point + disturbance_factor * random_vec
        waypoints.append(waypoint)

    waypoints.append(goal)
    return np.array(waypoints)

def calculate_path_length(path):
    """计算路径总长度"""
    if len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        total_length += np.linalg.norm(p2 - p1)

    return total_length
