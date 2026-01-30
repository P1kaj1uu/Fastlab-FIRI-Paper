import numpy as np

def analyze_path_smoothness(path):
    """分析路径平滑度，计算路径拐角的平均角度"""
    if len(path) < 3:
        return 0.0

    angles = []
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i]) - np.array(path[i - 1])
        v2 = np.array(path[i + 1]) - np.array(path[i])

        # 计算向量的模长
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        # 避免除以零
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            continue

        # 计算夹角余弦值
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差

        # 转换为角度
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)

    # 返回平均角度
    return np.mean(angles) if angles else 0.0

def check_collisions(path, obstacles):
    """检查路径是否与障碍物碰撞

    参数:
        path: 路径点列表
        obstacles: 障碍物集合

    返回:
        collision_points: 碰撞点列表
    """
    collision_points = []

    if obstacles is None:
        return collision_points

    # 检查每个路径点
    for i, point in enumerate(path):
        point = np.array(point)
        for obs in obstacles.obstacle_list:
            # 如果半径为None，给它一个默认值（比如1.0）
            obs_radius = obs.radius if obs.radius is not None else 1.0  # 默认半径为1.0

            # 对于球体，检查点到障碍物中心的距离
            if obs.shape == 'sphere':
                dist = np.linalg.norm(point - np.array(obs.center))
                if dist <= obs_radius:
                    collision_points.append(point)
                    break

            # 对于圆柱体，检查点到圆柱体的轴线的距离，并检查是否在圆柱体的高度范围内
            elif obs.shape == 'cylinder':
                # 计算点到圆柱体的轴线的距离
                axis_vector = np.array([0, 0, 1])  # 假设圆柱体的轴线沿着z轴
                cylinder_base_center = np.array(obs.center)
                direction_to_point = point - cylinder_base_center
                horizontal_distance = np.linalg.norm(direction_to_point[:2])  # 只考虑xy平面
                if horizontal_distance <= obs_radius and (cylinder_base_center[2] - obs.height / 2 <= point[2] <= cylinder_base_center[2] + obs.height / 2):
                    collision_points.append(point)
                    break

            # 对于长方体，检查点是否在长方体的范围内
            elif obs.shape == 'cuboid':
                x_min, y_min, z_min = obs.center - obs.size / 2
                x_max, y_max, z_max = obs.center + obs.size / 2
                if (x_min <= point[0] <= x_max and
                    y_min <= point[1] <= y_max and
                    z_min <= point[2] <= z_max):
                    collision_points.append(point)
                    break

    return collision_points

def analyze_path_results(original_path, smoothed_path, obstacles):
    """分析路径规划结果"""
    print("\n=== 规划结果分析 ===")

    # 计算路径长度
    def compute_path_length(path):
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i - 1])
        return length

    # 计算平均角度
    def compute_avg_angle(path):
        """
        计算路径点之间的平均角度
        """
        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i - 1]
            v2 = path[i + 1] - path[i]

            # 计算向量的范数
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            # 检查范数是否为零
            if norm_v1 == 0 or norm_v2 == 0:
                # 如果范数为零，跳过当前点
                continue

            # 计算夹角的余弦值
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保值在 [-1, 1] 范围内
            angle = np.arccos(cos_angle) * 180 / np.pi

            angles.append(angle)

        # 计算平均角度
        return np.mean(angles) if angles else 0

    # 检查碰撞
    def count_collisions(path, obstacles, safety_margin=1.0):
        collisions = 0
        collision_points = []
        for i, point in enumerate(path):
            for obs in obstacles.obstacle_list:
                obs_center = np.array(obs.center)
                dist = np.linalg.norm(point - obs_center)
                if dist < obs.radius * safety_margin:
                    collisions += 1
                    collision_points.append(i)
                    break
        return collisions, collision_points

    # 分析原始路径
    orig_length = compute_path_length(original_path)
    orig_smoothness = compute_avg_angle(original_path)
    orig_collisions, _ = count_collisions(original_path, obstacles)

    # 分析平滑后的路径
    smooth_length = compute_path_length(smoothed_path)
    smooth_smoothness = compute_avg_angle(smoothed_path)
    smooth_collisions, collision_indices = count_collisions(smoothed_path, obstacles)

    # 打印结果
    print(f"路径长度: {smooth_length:.2f}")
    print(f"路径平滑度: {smooth_smoothness:.2f}")

    if smooth_collisions > 0:
        print(f"规划失败: 检测到 {smooth_collisions} 个碰撞点")
    else:
        print(f"规划成功: 无碰撞")

    # 保存分析结果
    with open('temp/path_safety.txt', 'w') as f:
        f.write(f"Length: {smooth_length:.2f}\n")
        f.write(f"Smoothness: {smooth_smoothness:.2f}\n")
        f.write(f"Collisions: {smooth_collisions}\n")
        if collision_indices:
            f.write(f"Collision points: {collision_indices}\n")

    print("\nVisualizing path planning results (path & obstacles only)...")

    if smooth_collisions > 0:
        print("\n 规划结果: 失败")
    else:
        print("\n 规划结果: 成功")
