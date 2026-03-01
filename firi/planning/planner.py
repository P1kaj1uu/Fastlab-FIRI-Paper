import numpy as np
import time
import os
import pickle
from scipy.spatial import KDTree

from .firi import FIRI
from .config import FIRIConfig
from ..geometry import Ellipsoid

class FIRIPlanner:
    """FIRI路径规划器，根据障碍物环境计算安全路径"""
    
    def __init__(self, obstacles, space_size=(10, 10, 10)):
        """
        初始化规划器
        
        参数:
            obstacles: 障碍物列表
            space_size: 空间尺寸
        """
        self.obstacles = obstacles
        self.space_size = space_size
        self.dimension = len(space_size)
        
        # 创建配置
        self.config = FIRIConfig(space_size)
        
        # 调整参数
        self.config.update_adaptive_params(obstacle_count=len(obstacles))
        
        # 创建FIRI实例
        self.firi = FIRI(obstacles, self.dimension)
        
        # 构建障碍物KD树用于快速碰撞检测
        self._build_obstacle_kdtree()
        
        # 保存路径状态
        self.safe_regions = []
        self.path_points = []
        self.path_collisions = []
        
    def _build_obstacle_kdtree(self):
        """构建障碍物KD树，用于快速距离查询"""
        vertices = []
        self.obstacle_radii = []
        
        # 收集所有障碍物顶点
        for obs in self.obstacles:
            try:
                # 尝试不同的障碍物表示
                if hasattr(obs, 'vertices') and obs.vertices is not None:
                    # 如果有顶点表示，使用顶点
                    obs_vertices = np.asarray(obs.vertices)
                    vertices.extend(obs_vertices)
                    # 使用顶点到中心的最大距离作为半径
                    center = np.mean(obs_vertices, axis=0)
                    radius = np.max(np.linalg.norm(obs_vertices - center, axis=1))
                    self.obstacle_radii.append((center, radius))
                elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
                    # 基于中心和半径的表示
                    center = np.array(obs.center)
                    radius = obs.radius
                    # 简单地在球面上采样一些点
                    num_samples = 20
                    # 使用黄金螺旋采样
                    indices = np.arange(0, num_samples, dtype=float) + 0.5
                    phi = np.arccos(1 - 2*indices/num_samples)
                    theta = np.pi * (1 + 5**0.5) * indices
                    x = radius * np.cos(theta) * np.sin(phi)
                    y = radius * np.sin(theta) * np.sin(phi)
                    z = radius * np.cos(phi)
                    
                    # 如果是2D，只使用x和y
                    if self.dimension == 2:
                        points = np.vstack([x, y]).T
                    else:
                        points = np.vstack([x, y, z]).T
                        
                    # 添加中心偏移
                    points = points + center
                    vertices.extend(points)
                    self.obstacle_radii.append((center, radius))
                elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                    # 字典表示
                    center = np.array(obs['center'])
                    radius = obs['radius']
                    # 同样采样表面点
                    num_samples = 20
                    indices = np.arange(0, num_samples, dtype=float) + 0.5
                    phi = np.arccos(1 - 2*indices/num_samples)
                    theta = np.pi * (1 + 5**0.5) * indices
                    x = radius * np.cos(theta) * np.sin(phi)
                    y = radius * np.sin(theta) * np.sin(phi)
                    z = radius * np.cos(phi)
                    
                    # 如果是2D，只使用x和y
                    if self.dimension == 2:
                        points = np.vstack([x, y]).T
                    else:
                        points = np.vstack([x, y, z]).T
                        
                    # 添加中心偏移
                    points = points + center
                    vertices.extend(points)
                    self.obstacle_radii.append((center, radius))
            except Exception as e:
                print(f"构建KD树时处理障碍物出错: {e}")
                continue
        
        # 创建KD树
        if vertices:
            self.obstacle_tree = KDTree(vertices)
            print(f"已构建KD-Tree: {len(vertices)}个顶点")
        else:
            self.obstacle_tree = None
            print("警告: 无法构建障碍物KD树")
    
    def generate_safe_regions(self, start, goal, num_waypoints=4):
        """
        生成从起点到终点的安全区域
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            num_waypoints: 路径段数量
            
        返回:
            安全区域列表
        """
        # 清空之前的安全区域
        self.safe_regions = []
        
        # 创建临时目录
        os.makedirs('temp', exist_ok=True)
        
        # 生成直线路径的中间点
        t_values = np.linspace(0, 1, num_waypoints+1)
        path_points = np.array([start * (1-t) + goal * t for t in t_values])
        
        # 保存调整后的路径点
        pickle.dump(path_points, open('temp/adjusted_path.pkl', 'wb'))
        
        # 按照论文中的方法生成种子点
        # 为每段路径生成安全区域
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            # 计算路径段中点
            mid_point = (p1 + p2) / 2
            
            # 计算路径方向
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                # 如果方向为零，随机选择
                direction = np.random.randn(self.dimension)
                direction = direction / np.linalg.norm(direction)
            
            # 计算与路径方向垂直的两个方向（在3D中）
            if self.dimension == 3:
                # 找到两个与direction垂直的单位向量
                # 首先找到一个与direction不平行的向量
                if abs(direction[0]) < abs(direction[1]):
                    normal1 = np.array([1, 0, 0])
                else:
                    normal1 = np.array([0, 1, 0])
                    
                # 计算叉积获得第一个法向量
                normal1 = np.cross(direction, normal1)
                normal1 = normal1 / np.linalg.norm(normal1)
                
                # 计算第二个法向量
                normal2 = np.cross(direction, normal1)
                normal2 = normal2 / np.linalg.norm(normal2)
            else:
                # 在2D中，只有一个垂直方向
                normal1 = np.array([-direction[1], direction[0]])
                normal2 = -normal1
            
            # 生成种子点: 路径段两端点、中点和侧向点
            seed_points = [p1, p2, mid_point]
            
            # 添加侧向点（在方向向量的法平面内）
            if self.dimension == 3:
                side_dist = direction_norm * 0.2  # 侧向距离
                seed_points.append(mid_point + normal1 * side_dist)
                seed_points.append(mid_point + normal2 * side_dist)
            else:
                side_dist = direction_norm * 0.2
                seed_points.append(mid_point + normal1 * side_dist)
            
            # 计算特定路径段的安全区域
            print(f"为路径段 {i} 计算安全区域 (包含 {len(seed_points)} 个种子点)...")
            
            # 测量计算时间
            start_time = time.time()
            
            # 获取FIRI参数
            iterations = self.config.safety_iterations
            threshold = self.config.volume_threshold
            
            try:
                # 使用FIRI算法计算安全区域
                polytope, ellipsoid = self.firi.compute_safe_region(
                    seed_points, 
                    max_iterations=iterations, 
                    volume_threshold=threshold
                )
                
                # 添加到安全区域列表
                self.safe_regions.append((polytope, ellipsoid))
                
                # 保存安全区域到文件
                region_data = {
                    'polytope_halfspaces': polytope.halfspaces,
                    'ellipsoid_center': ellipsoid.center,
                    'ellipsoid_Q': ellipsoid.Q,
                    'seed_points': seed_points
                }
                pickle.dump(region_data, open(f'temp/safe_region_{i}.pkl', 'wb'))
                
                # 记录计算时间
                end_time = time.time()
                self.config.record_timing('safe_region', (end_time - start_time) * 1000)
                
                # 输出椭球体信息
                print(f"  安全区域 {i} 椭球体体积: {ellipsoid.volume():.6f}")
                
            except Exception as e:
                print(f"计算安全区域 {i} 时出错: {e}")
                # 创建一个默认的安全区域
                default_center = (p1 + p2) / 2
                default_radius = np.linalg.norm(p2 - p1) / 2
                default_ellipsoid = Ellipsoid(default_center, np.eye(self.dimension) * default_radius**2)
                
                # 使用椭球体生成半空间约束
                halfspaces = []
                # 在单位球上采样点
                num_samples = 20
                if self.dimension == 3:
                    indices = np.arange(0, num_samples, dtype=float) + 0.5
                    phi = np.arccos(1 - 2*indices/num_samples)
                    theta = np.pi * (1 + 5**0.5) * indices
                    
                    x = np.cos(theta) * np.sin(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(phi)
                    
                    directions = np.vstack([x, y, z]).T
                else:
                    theta = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
                    x = np.cos(theta)
                    y = np.sin(theta)
                    directions = np.vstack([x, y]).T
                
                for direction in directions:
                    # 计算半空间
                    normal = direction / np.linalg.norm(direction)
                    offset = -np.dot(normal, default_center) + default_radius
                    hs = np.zeros(self.dimension + 1)
                    hs[:-1] = normal
                    hs[-1] = offset
                    halfspaces.append(hs)
                
                from ..geometry import ConvexPolytope
                default_polytope = ConvexPolytope(halfspaces=np.array(halfspaces))
                
                # 添加到安全区域列表
                self.safe_regions.append((default_polytope, default_ellipsoid))
                
                # 保存安全区域到文件
                region_data = {
                    'polytope_halfspaces': default_polytope.halfspaces,
                    'ellipsoid_center': default_ellipsoid.center,
                    'ellipsoid_Q': default_ellipsoid.Q,
                    'seed_points': seed_points
                }
                pickle.dump(region_data, open(f'temp/safe_region_{i}.pkl', 'wb'))
        
        return self.safe_regions
    
    def plan_path(self, start, goal, initial_waypoints=None, smoothing=True, max_replanning_attempts=3, safety_margin=1.0):
        """
        规划从起点到终点的路径
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            initial_waypoints: 自定义初始路径点，如果为None则使用直线路径
            smoothing: 是否进行路径平滑
            max_replanning_attempts: 最大重规划尝试次数
            safety_margin: 安全距离系数，用于碰撞检测
            
        返回:
            路径点序列
        """
        print("规划路径...")
        
        # 确保已生成安全区域
        if not self.safe_regions:
            self.generate_safe_regions(start, goal)
        
        # 设置安全距离
        self.config.collision_threshold *= safety_margin
        
        # 初始路径设置
        if initial_waypoints is not None:
            # 使用用户提供的初始路径点
            init_path = initial_waypoints
            print("使用提供的初始路径点")
        else:
            # 使用直线路径作为初始猜测
            num_segments = len(self.safe_regions)
            init_path = np.zeros((num_segments + 1, self.dimension))
            init_path[0] = start
            init_path[-1] = goal
            
            # 填充中间点（创建初始猜测）
            for i in range(1, num_segments):
                t = i / num_segments
                init_path[i] = start * (1-t) + goal * t
        
        print("初始路径点:", init_path)
        
        # 检查初始路径是否安全
        collisions = self.check_path_safety(init_path)
        
        # 如果有碰撞，尝试重新规划
        if collisions:
            print("发现碰撞! 尝试重新规划路径...")
            
            for attempt in range(max_replanning_attempts):
                print(f"重新规划尝试 {attempt+1}/{max_replanning_attempts}")
                
                # 尝试找到更好的中间点
                replan_path = init_path.copy()
                
                # 第一种策略：向安全区域椭球体中心移动
                if attempt == 0:
                    for idx in range(1, len(replan_path)-1):  # 不调整起点和终点
                        # 获取相关安全区域的椭球体
                        region_idx = min(idx-1, len(self.safe_regions)-1)
                        if region_idx >= 0:
                            _, ellipsoid = self.safe_regions[region_idx]
                            
                            # 向椭球体中心移动点
                            direction = ellipsoid.center - replan_path[idx]
                            dist = np.linalg.norm(direction)
                            if dist > 1e-10:
                                # 沿椭球体中心方向移动点，调整幅度随尝试次数增加
                                move_dist = min(dist * 0.6, 2.0)
                                replan_path[idx] += direction / dist * move_dist
                
                # 第二种策略：添加随机扰动
                elif attempt == 1:
                    for idx in range(1, len(replan_path)-1):
                        if idx in collisions or idx-1 in collisions:
                            # 计算起点到终点的向量
                            path_vector = goal - start
                            path_length = np.linalg.norm(path_vector)
                            
                            # 创建垂直于路径的随机方向
                            random_vec = np.random.randn(3)
                            random_vec = random_vec - np.dot(random_vec, path_vector) * path_vector / np.dot(path_vector, path_vector)
                            random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-10)
                            
                            # 添加扰动，幅度为路径长度的一定比例
                            displacement = random_vec * path_length * 0.3
                            replan_path[idx] += displacement
                
                # 第三种策略：使用3D提升策略（抛物线形状）
                elif attempt == 2:
                    # 使用直线连接起点和终点，但沿z轴稍微抬高
                    for i in range(1, len(replan_path)-1):
                        t = i / (len(replan_path)-1)
                        # 基础位置（在起点和终点连线上）
                        replan_path[i] = start * (1-t) + goal * t
                        
                        # 如果是3D，尝试向上抬升路径
                        if self.dimension >= 3:
                            # 添加一个抛物线形状的抬升
                            lift = 4 * t * (1-t)  # 最大抬升在中点
                            # 提高抬升幅度
                            replan_path[i, 2] += lift * np.linalg.norm(goal - start) * 0.3
                
                # 最后的策略：使用更激进的Z字形路径
                else:
                    midpoint = (start + goal) / 2
                    offset_amount = np.linalg.norm(goal - start) * 0.4
                    
                    # 获取一个垂直于路径的方向
                    path_dir = goal - start
                    path_dir = path_dir / np.linalg.norm(path_dir)
                    
                    # 找一个垂直向量
                    if np.abs(path_dir[0]) < np.abs(path_dir[1]):
                        perp_dir = np.array([1, 0, 0])
                    else:
                        perp_dir = np.array([0, 1, 0])
                    
                    # 确保垂直
                    perp_dir = perp_dir - np.dot(perp_dir, path_dir) * path_dir
                    perp_dir = perp_dir / np.linalg.norm(perp_dir)
                    
                    # 创建Z字形路径
                    if len(replan_path) >= 3:
                        zigzag_path = np.zeros_like(replan_path)
                        zigzag_path[0] = start
                        zigzag_path[-1] = goal
                        
                        # 中间点向两侧偏移
                        for i in range(1, len(zigzag_path)-1):
                            t = i / (len(zigzag_path)-1)
                            # 基础位置
                            zigzag_path[i] = start * (1-t) + goal * t
                            # 添加偏移
                            if i % 2 == 1:  # 奇数点向一个方向偏移
                                zigzag_path[i] += perp_dir * offset_amount
                            else:  # 偶数点向另一个方向偏移
                                zigzag_path[i] -= perp_dir * offset_amount
                        
                        replan_path = zigzag_path
                
                # 检查新路径是否安全
                new_collisions = self.check_path_safety(replan_path)
                
                if not new_collisions:
                    print("找到安全路径!")
                    init_path = replan_path
                    collisions = []
                    break
                elif len(new_collisions) < len(collisions):
                    print(f"碰撞减少: {len(collisions)} -> {len(new_collisions)}")
                    init_path = replan_path
                    collisions = new_collisions
            
            # 更新路径状态
            self.path_collisions = collisions
            
            if collisions:
                # 如果仍有碰撞，添加警告
                print(f"警告: 路径规划未能完全消除碰撞，仍有 {len(collisions)} 处碰撞")
        
        # 保存最终路径
        final_path = init_path
        
        # 如果需要平滑且路径无碰撞，进行路径平滑
        if smoothing and not collisions:
            try:
                smooth_path = self.smooth_path(
                    final_path, 
                    window_size=self.config.smoothing_window, 
                    iterations=self.config.smoothing_iterations
                )
                
                # 检查平滑后的路径是否安全
                smooth_collisions = self.check_path_safety(smooth_path)
                
                if not smooth_collisions:
                    print("路径平滑成功且安全")
                    final_path = smooth_path
                else:
                    print(f"平滑路径不安全，有 {len(smooth_collisions)} 处碰撞，使用原始路径")
            except Exception as e:
                print(f"路径平滑出错: {e}")
        
        # 保存最终路径
        self.path_points = final_path
        print("最终路径点:", final_path)
        
        # 保存路径相关信息
        try:
            # 保存路径点
            pickle.dump(final_path, open('temp/path_points.pkl', 'wb'))
            pickle.dump(final_path, open('temp/adjusted_path.pkl', 'wb'))
            
            # 保存路径安全性信息
            with open('temp/path_safety.txt', 'w') as f:
                f.write(f"path_points: {len(final_path)}\n")
                f.write(f"collision_segments: {len(collisions)}\n")
                f.write(f"collision_indices: {collisions}\n")
                f.write(f"path_safety: {'Safe' if not collisions else 'Unsafe'}\n")
                
                # 添加平滑度信息
                angles = []
                for i in range(1, len(final_path) - 1):
                    v1 = final_path[i] - final_path[i-1]
                    v2 = final_path[i+1] - final_path[i]
                    
                    # 计算夹角
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在[-1, 1]范围内
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
                
                # 写入角度统计
                if angles:
                    max_angle = np.max(angles) if angles else 0
                    avg_angle = np.mean(angles) if angles else 0
                    count_large = sum(1 for a in angles if a > 90)
                    f.write(f"max_angle: {max_angle:.2f}° avg_angle: {avg_angle:.2f}° angles>90°: {count_large}\n")
                else:
                    f.write(f"max_angle: 0.00° avg_angle: 0.00° angles>90°: 0\n")
                    
        except Exception as e:
            print(f"保存路径信息出错: {e}")
        
        return final_path
    
    def check_path_safety(self, path):
        """
        检查路径是否安全
        
        参数:
            path: 路径点序列
            
        返回:
            碰撞段索引列表
        """
        collisions = []
        
        # 对每个路径段进行检测
        for i in range(len(path) - 1):
            if self.check_segment_collision(path[i], path[i+1]):
                collisions.append(i)
        
        return collisions
    
    def check_segment_collision(self, p1, p2, samples=None):
        """
        检查路径段是否与障碍物碰撞
        
        参数:
            p1, p2: 路径段两端点
            samples: 采样点数量
            
        返回:
            是否碰撞
        """
        if samples is None:
            samples = self.config.path_samples
            
        # 根据路径段长度自适应调整采样点数量
        dist = np.linalg.norm(p2 - p1)
        if dist > 2.0:
            samples = max(samples, int(dist * 5))
        
        # 对路径段进行采样
        t_values = np.linspace(0, 1, samples)
        for t in t_values:
            point = p1 * (1-t) + p2 * t
            
            if self.check_point_collision(point):
                return True
        
        return False
    
    def check_point_collision(self, point, safe_distance=None):
        """
        检查点是否与障碍物碰撞
        
        参数:
            point: 待检测点
            safe_distance: 安全距离阈值
            
        返回:
            是否碰撞
        """
        if safe_distance is None:
            safe_distance = self.config.collision_threshold
            
        # 使用KD树快速找到最近的障碍物
        if self.obstacle_tree is not None:
            # 查询最近距离
            distance, _ = self.obstacle_tree.query(point, k=1)
            
            # 如果距离小于安全距离，认为碰撞
            if distance < safe_distance:
                return True
                
        # 检查与障碍物的精确距离
        for center, radius in self.obstacle_radii:
            dist = np.linalg.norm(point - center) - radius
            if dist < safe_distance:
                return True
        
        return False
    
    def smooth_path(self, path, window_size=3, iterations=50, max_angle=60.0, angle_weight=0.7, safety_margin=1.2):
        """
        使用移动平均法平滑路径，并添加角度约束
        
        参数:
            path: 路径点列表
            window_size: 平滑窗口大小
            iterations: 平滑迭代次数
            max_angle: 最大允许角度变化（度）
            angle_weight: 角度约束权重
            safety_margin: 碰撞检测安全系数
        
        返回:
            平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        # 计算初始角度
        def compute_angles(pts):
            angles = []
            for i in range(1, len(pts) - 1):
                v1 = pts[i] - pts[i-1]
                v2 = pts[i+1] - pts[i]
                # 用余弦定理计算角度（弧度）
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                # 确保值在-1到1之间以防止数值误差
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                # 将角度转换为0-180度
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
            return np.array(angles)
        
        initial_angles = compute_angles(path)
        # 确定哪些点的角度超过阈值
        large_angle_indices = np.where(initial_angles > max_angle)[0] + 1  # +1因为angles比path少两个元素
        
        smoothed_path = np.copy(path)
        best_path = np.copy(path)
        best_max_angle = 180.0
        
        # 保存起点和终点，我们不希望移动它们
        start_point = path[0].copy()
        end_point = path[-1].copy()
        
        for iteration in range(iterations):
            previous_path = np.copy(smoothed_path)
            
            # 自适应窗口大小，大角度点使用较小窗口
            for i in range(1, len(smoothed_path) - 1):
                current_window = window_size
                if i in large_angle_indices:
                    current_window = max(2, window_size - 1)
                
                # 计算窗口范围
                window_start = max(0, i - current_window // 2)
                window_end = min(len(smoothed_path), i + current_window // 2 + 1)
                
                # 基于窗口内点的平均位置计算新位置
                window_points = previous_path[window_start:window_end]
                avg_position = np.mean(window_points, axis=0)
                
                # 如果是大角度点，对平均位置进行额外处理
                if i in large_angle_indices:
                    # 计算前后段方向
                    v_prev = previous_path[i] - previous_path[i-1]
                    v_next = previous_path[i+1] - previous_path[i]
                    v_prev_norm = v_prev / np.linalg.norm(v_prev)
                    v_next_norm = v_next / np.linalg.norm(v_next)
                    
                    # 计算平均方向
                    avg_direction = (v_prev_norm + v_next_norm) / 2
                    avg_direction = avg_direction / np.linalg.norm(avg_direction)
                    
                    # 基于平均方向修正位置
                    direction_correction = avg_direction * np.linalg.norm(v_prev + v_next) * 0.25
                    avg_position = previous_path[i-1] + direction_correction
                
                # 角度权重自适应调整
                adaptive_weight = angle_weight
                if i in large_angle_indices:
                    adaptive_weight = min(0.9, angle_weight + 0.2)
                
                # 应用权重混合原始位置和平均位置
                smoothed_path[i] = (1-adaptive_weight) * previous_path[i] + adaptive_weight * avg_position
            
            # 固定起点和终点
            smoothed_path[0] = start_point
            smoothed_path[-1] = end_point
            
            # 碰撞检查
            collision_count = 0
            for i in range(len(smoothed_path)):
                point = smoothed_path[i]
                # 检查当前点是否与障碍物碰撞
                for obs in self.obstacles.obstacle_list:
                    obs_center = np.array(obs.center)
                    obs_radius = obs.radius * safety_margin  # 使用安全系数
                    dist = np.linalg.norm(point - obs_center)
                    if dist < obs_radius:
                        collision_count += 1
                        # 移动点远离障碍物
                        direction = point - obs_center
                        if np.linalg.norm(direction) > 1e-6:  # 避免零向量
                            direction = direction / np.linalg.norm(direction)
                            # 移动到障碍物边缘之外
                            smoothed_path[i] = obs_center + direction * (obs_radius + 0.1)
            
            # 检查路径是否安全，如果不安全则停止平滑
            is_safe = True
            if collision_count > 0:
                # 尝试修复碰撞
                for j in range(10):  # 最多尝试10次修复
                    collision_count = 0
                    for i in range(len(smoothed_path)):
                        point = smoothed_path[i]
                        for obs in self.obstacles.obstacle_list:
                            obs_center = np.array(obs.center)
                            obs_radius = obs.radius * safety_margin
                            dist = np.linalg.norm(point - obs_center)
                            if dist < obs_radius:
                                collision_count += 1
                                # 移动点远离障碍物
                                direction = point - obs_center
                                if np.linalg.norm(direction) > 1e-6:
                                    direction = direction / np.linalg.norm(direction)
                                    # 增加移动距离
                                    smoothed_path[i] = obs_center + direction * (obs_radius + 0.2 * (j+1))
                
                    if collision_count == 0:
                        break
                
                # 修复后仍有碰撞则放弃这次平滑结果
                if collision_count > 0:
                    is_safe = False
                    smoothed_path = previous_path
            
            # 计算当前路径的最大角度
            current_angles = compute_angles(smoothed_path)
            current_max_angle = np.max(current_angles) if len(current_angles) > 0 else 0
            
            # 如果当前路径角度更好，且安全，则更新最佳路径
            if is_safe and current_max_angle < best_max_angle:
                best_path = np.copy(smoothed_path)
                best_max_angle = current_max_angle
            
            # 如果角度已经足够小，则提前结束
            if best_max_angle < max_angle * 0.8:
                break
        
        # 添加额外的插值点来降低大角度
        if best_max_angle > max_angle:
            final_path = []
            angles = compute_angles(best_path)
            
            for i in range(len(best_path)):
                final_path.append(best_path[i])
                
                # 在大角度点前后添加插值点
                if 0 < i < len(best_path) - 1 and i-1 < len(angles) and angles[i-1] > max_angle:
                    # 在当前点和前一点之间插入中点
                    mid_point_prev = 0.5 * (best_path[i] + best_path[i-1])
                    final_path.insert(len(final_path)-1, mid_point_prev)
                    
                    # 在当前点和后一点之间插入中点
                    if i < len(best_path) - 1:
                        mid_point_next = 0.5 * (best_path[i] + best_path[i+1])
                        final_path.append(mid_point_next)
            
            # 转换回numpy数组
            return np.array(final_path)
        
        return best_path 