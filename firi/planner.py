import numpy as np
import time
import os
import pickle
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev

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
        self.obstacle_radii = []  # 存储每个障碍物的半径或尺寸
    
        for obs in self.obstacles:
            try:
                if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                   # 处理球体障碍物
                    center = np.array(obs.center)
                    radius = obs.radius if obs.radius is not None else 1.0  # 默认值
                    num_samples = 20
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    z = np.linspace(center[2] - radius, center[2] + radius, num_samples)
                
                    vertices.extend(np.column_stack((x, y, z)))
                    self.obstacle_radii.append((center, radius))
                elif hasattr(obs, 'size'):
                    # 处理长方体障碍物
                    center = np.array(obs.center)
                    size = np.array(obs.size)
                    half_sizes = size / 2
                    x = np.array([-half_sizes[0], half_sizes[0], half_sizes[0], -half_sizes[0], -half_sizes[0], half_sizes[0], half_sizes[0], -half_sizes[0]])
                    y = np.array([-half_sizes[1], -half_sizes[1], half_sizes[1], half_sizes[1], -half_sizes[1], -half_sizes[1], half_sizes[1], half_sizes[1]])
                    z = np.array([-half_sizes[2], -half_sizes[2], -half_sizes[2], -half_sizes[2], half_sizes[2], half_sizes[2], half_sizes[2], half_sizes[2]])
                    vertices.extend(np.column_stack((x, y, z)) + center)
                    self.obstacle_radii.append((center, half_sizes))  # 存储长方体的半尺寸
                elif hasattr(obs, 'height'):
                    # 处理圆柱体障碍物
                    center = np.array(obs.center)
                    radius = obs.radius if obs.radius is not None else 1.0
                    height = obs.height if obs.height is not None else 1.0
                    self.obstacle_radii.append((center, radius, height))  # 存储圆柱体的半径和高度
                    num_samples = 20
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    z = np.linspace(center[2] - height / 2, center[2] + height / 2, num_samples)
                
                    vertices.extend(np.column_stack((x, y, z)))
            except Exception as e:
                print(f"构建KD树时处理障碍物出错: {e}")
                continue

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
                direction = np.random.randn(self.dimension)
                direction = direction / np.linalg.norm(direction)
        
            # 计算与路径方向垂直的两个方向（在3D中）
            if self.dimension == 3:
                if abs(direction[0]) < abs(direction[1]):
                    normal1 = np.array([1, 0, 0])
                else:
                    normal1 = np.array([0, 1, 0])
            
                normal1 = np.cross(direction, normal1)
                normal1 = normal1 / np.linalg.norm(normal1)
            
                normal2 = np.cross(direction, normal1)
                normal2 = normal2 / np.linalg.norm(normal2)
            else:
                normal1 = np.array([-direction[1], direction[0]])
                normal2 = -normal1
        
            # 生成种子点: 路径段两端点、中点和侧向点
            seed_points = [p1, p2, mid_point]
        
            # 添加侧向点（在方向向量的法平面内）
            if self.dimension == 3:
                side_dist = direction_norm * 0.2
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
                num_samples = 20
                if self.dimension == 3:
                    indices = np.arange(0, num_samples, dtype=float) + 0.5
                    phi = np.arccos(1 - 2 * indices / num_samples)
                    theta = np.pi * (1 + 5**0.5) * indices
                
                    x = np.cos(theta) * np.sin(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(phi)
                
                    directions = np.vstack([x, y, z]).T
                else:
                    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
                    x = np.cos(theta)
                    y = np.sin(theta)
                    directions = np.vstack([x, y]).T
            
                for direction in directions:
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

    def bspline_smooth(self, path, smoothing_factor=0.2):
        """
        使用B样条平滑路径，使路径更加符合运动学连续性。
        参数:
            path: 原始路径 (N x 3) ndarray
            smoothing_factor: 控制平滑程度，越大越平滑，0为插值
        返回:
            平滑路径 (M x 3) ndarray
        """
        if len(path) < 4:
            print("路径点数量不足，无法生成B样条曲线。尝试插入额外的点。")
            # 插入额外的点
            extra_points = np.linspace(path[0], path[1], 4 - len(path) + 1)[1:-1]
            path = np.vstack((path[0], extra_points, path[1:]))
        
        path = np.array(path)
        x, y, z = path[:, 0], path[:, 1], path[:, 2]
        
        try:
            tck, u = splprep([x, y, z], s=smoothing_factor, k=3)
            u_new = np.linspace(0, 1, max(100, len(path) * 10))
            x_new, y_new, z_new = splev(u_new, tck)
            smoothed_path = np.vstack((x_new, y_new, z_new)).T
            return smoothed_path
        except Exception as e:
            print(f"B样条平滑失败: {e}")
            return path

    def plan_path(self, start, goal, initial_waypoints=None, smoothing=True, max_replanning_attempts=3, safety_margin=1.2):
        print("规划路径...")
        
        if not self.safe_regions:
            self.generate_safe_regions(start, goal)
        
        self.config.collision_threshold *= safety_margin
        
        if initial_waypoints is not None:
            init_path = initial_waypoints
            print("使用提供的初始路径点")
        else:
            num_segments = len(self.safe_regions)
            init_path = np.zeros((num_segments + 1, self.dimension))
            init_path[0] = start
            init_path[-1] = goal
            for i in range(1, num_segments):
                t = i / num_segments
                init_path[i] = start * (1 - t) + goal * t
            
        print("初始路径点:", init_path)
        collisions = self.check_path_safety(init_path)
        
        used_zigzag = False
        
        if collisions:
            print("发现碰撞! 尝试重新规划路径...")
            for attempt in range(max_replanning_attempts):
                print(f"重新规划尝试 {attempt+1}/{max_replanning_attempts}")
                replan_path = init_path.copy()
                
                if attempt == 0:
                    for idx in range(1, len(replan_path)-1):
                        region_idx = min(idx-1, len(self.safe_regions)-1)
                        if region_idx >= 0:
                            _, ellipsoid = self.safe_regions[region_idx]
                            direction = ellipsoid.center - replan_path[idx]
                            dist = np.linalg.norm(direction)
                            if dist > 1e-10:
                                move_dist = min(dist * 0.6, 2.0)
                                replan_path[idx] += direction / dist * move_dist
                elif attempt == 1:
                    for idx in range(1, len(replan_path)-1):
                        if idx in collisions or idx-1 in collisions:
                            path_vector = goal - start
                            path_length = np.linalg.norm(path_vector)
                            random_vec = np.random.randn(3)
                            random_vec = random_vec - np.dot(random_vec, path_vector) * path_vector / np.dot(path_vector, path_vector)
                            random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-10)
                            displacement = random_vec * path_length * 0.3
                            replan_path[idx] += displacement
                else:
                    if len(replan_path) >= 3:
                        midpoint = (start + goal) / 2
                        offset_amount = np.linalg.norm(goal - start) * 0.4
                        path_dir = goal - start
                        path_dir = path_dir / np.linalg.norm(path_dir)
                        if np.abs(path_dir[0]) < np.abs(path_dir[1]):
                            perp_dir = np.array([1, 0, 0])
                        else:
                            perp_dir = np.array([0, 1, 0])
                        perp_dir = perp_dir - np.dot(perp_dir, path_dir) * path_dir
                        perp_dir = perp_dir / np.linalg.norm(perp_dir)
                        if len(replan_path) >= 3:
                            zigzag_path = np.zeros_like(replan_path)
                            zigzag_path[0] = start
                            zigzag_path[-1] = goal
                            for i in range(1, len(zigzag_path)-1):
                                t = i / (len(zigzag_path)-1)
                                zigzag_path[i] = start * (1-t) + goal * t
                                if i % 2 == 1:
                                    zigzag_path[i] += perp_dir * offset_amount
                                else:
                                    zigzag_path[i] -= perp_dir * offset_amount
                            replan_path = zigzag_path
                            used_zigzag = True
                
                new_collisions = self.check_path_safety(replan_path)
                
                if not new_collisions:
                    print("找到安全路径!")
                    init_path = replan_path

                    # === NEW: 对Z字形路径进行B样条拟合 ===
                    if used_zigzag:
                        print("使用B样条拟合Z字形路径...")
                        bspline_path = self.bspline_smooth(init_path, smoothing_factor=0.2)
                        bspline_collisions = self.check_path_safety(bspline_path)
                        if not bspline_collisions:
                            print("B样条路径安全，采用拟合后的轨迹")
                            init_path = bspline_path
                        else:
                            print(f"B样条轨迹不安全，继续使用Z字形路径（碰撞段: {bspline_collisions}）")
                    break
                elif len(new_collisions) < len(collisions):
                    print(f"碰撞减少: {len(collisions)} -> {len(new_collisions)}")
                    init_path = replan_path
                    collisions = new_collisions

            self.path_collisions = collisions
            if collisions:
                print(f"警告: 路径规划未能完全消除碰撞，仍有 {len(collisions)} 处碰撞")
        
        final_path = init_path
        
        # 如果不是Z字形且需要平滑
        if smoothing and not collisions and not used_zigzag:
            try:
                bspline_path = self.bspline_smooth(final_path, smoothing_factor=0.2)
                bspline_collisions = self.check_path_safety(bspline_path)
                if not bspline_collisions:
                    print("路径平滑成功且安全")
                    final_path = bspline_path
                else:
                    print(f"平滑路径不安全，有 {len(bspline_collisions)} 处碰撞，使用原始路径")
            except Exception as e:
                print(f"B样条平滑出错: {e}")
        
        self.path_points = final_path
        print("最终路径点:", final_path)
        
        try:
            pickle.dump(final_path, open('temp/path_points.pkl', 'wb'))
            pickle.dump(final_path, open('temp/adjusted_path.pkl', 'wb'))
            with open('temp/path_safety.txt', 'w') as f:
                f.write(f"path_points: {len(final_path)}\n")
                f.write(f"collision_segments: {len(collisions)}\n")
                f.write(f"collision_indices: {collisions}\n")
                f.write(f"path_safety: {'Safe' if not collisions else 'Unsafe'}\n")
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
            distance, _ = self.obstacle_tree.query(point, k=1)
            if distance < safe_distance:
                return True
    
        # 对于不同的障碍物类型进行碰撞检测
        for center, *params in self.obstacle_radii:
            if len(params) == 1:  # 球体
                radius = params[0]
                dist = np.linalg.norm(point - center) - radius
            elif len(params) == 2:  # 长方体
                half_sizes = params[0]
                dist = np.abs(point - center) - half_sizes
                dist = np.max(dist)
            elif len(params) == 3:  # 圆柱体
                radius, height = params[0], params[1]
                dist = np.linalg.norm(point - center[:2]) - radius  # 只考虑xy平面
                if dist < 0 and center[2] - height / 2 <= point[2] <= center[2] + height / 2:
                    return True
            if dist < safe_distance:
                return True
        return False

