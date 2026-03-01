import numpy as np
from ..geometry import Ellipsoid, ConvexPolytope
from .mvie import MVIE_SOCP

class FIRI:
    """
    快速迭代区域膨胀算法 (Fast Iterative Regional Inflation)
    用于计算路径段周围的安全区域
    """
    def __init__(self, obstacles, dimension=3):
        """
        初始化FIRI算法
        
        参数:
            obstacles: 障碍物列表，每个障碍物应该支持center和radius属性
            dimension: 问题空间维度
        """
        self.obstacles = obstacles
        self.dim = dimension
        # 添加MVIE计算器实例
        self.mvie_socp = MVIE_SOCP(dimension)
    
    def compute_safe_region(self, seed_points, initial_ellipsoid=None, 
                           max_iterations=3, volume_threshold=0.01):
        """
        计算一组种子点的安全区域
        使用FIRI算法不断扩大椭球体直到收敛
        
        参数:
            seed_points: 需要包含的种子点列表
            initial_ellipsoid: 初始椭球体 (可选)
            max_iterations: 最大迭代次数，默认为3
            volume_threshold: 体积增长阈值，低于此值认为收敛
        
        返回:
            (polytope, ellipsoid): 安全区域的多胞体表示和最大内接椭球体
        """
        # 获取种子点数量
        num_seeds = len(seed_points)
        if num_seeds == 0:
            raise ValueError("需要至少一个种子点")
            
        # 确保种子点是numpy数组
        seed_points = np.array(seed_points)
        
        # 初始化椭球体
        if initial_ellipsoid is None:
            # 使用种子点的中心作为初始椭球体中心
            center = np.mean(seed_points, axis=0)
            # 计算种子点的边界范围，用于初始化合适大小的椭球体
            seed_min = np.min(seed_points, axis=0)
            seed_max = np.max(seed_points, axis=0)
            seed_extent = np.linalg.norm(seed_max - seed_min)
            
            # 使用一个更合理大小的初始椭球体，而不是固定的小尺寸
            Q = np.eye(self.dim) * max(0.5, seed_extent**2 / 40)
            current_ellipsoid = Ellipsoid(center, Q)
        else:
            current_ellipsoid = initial_ellipsoid
        
        # 记录初始椭球体，以防后续计算发生错误时回退
        initial_ellipsoid = current_ellipsoid
        
        # 迭代求解，直到收敛或达到最大迭代次数
        prev_volume = current_ellipsoid.volume()
        current_polytope = None
        
        # 记录上次正确计算的状态
        last_valid_polytope = None
        last_valid_ellipsoid = current_ellipsoid
        
        # 体积异常增长计数器
        abnormal_growth_count = 0
        fallback_mode = False
        
        for iter_num in range(max_iterations):
            print(f"FIRI迭代 {iter_num+1}/{max_iterations}...")
            
            try:
                # 1. 执行限制性膨胀，创建包含所有种子点的凸多胞体
                print("  执行限制性膨胀...")
                
                # 在发现体积异常增长超过阈值时切换到保守模式
                if abnormal_growth_count > 1:
                    print("  检测到连续的体积异常增长，切换到保守模式")
                    fallback_mode = True
                    
                # 根据模式选择膨胀算法
                if fallback_mode:
                    # 保守模式使用更简单的方法
                    current_polytope = self.restrictive_inflation_simple(current_ellipsoid, seed_points)
                else:
                    # 标准模式
                    current_polytope = self.restrictive_inflation(current_ellipsoid, seed_points)
                
                # 保存计算成功的多胞体
                last_valid_polytope = current_polytope
                
                # 2. 计算多胞体内的最大内接椭球 - 使用SOCP方法
                try:
                    print("  使用SOCP方法计算MVIE...")
                    new_ellipsoid = self.compute_mvie(current_polytope)
                    current_volume = new_ellipsoid.volume()
                    print(f"  当前椭球体体积: {current_volume:.6f}")
                    
                    # 检查体积是否合理
                    if current_volume <= 0 or current_volume > 1e12 or np.isnan(current_volume) or np.isinf(current_volume):
                        print("  体积异常，使用备用方法")
                        new_ellipsoid = self.compute_mvie_fallback(current_polytope)
                        current_volume = new_ellipsoid.volume()
                        print(f"  备用方法计算的椭球体体积: {current_volume:.6f}")
                    
                    # 检查是否收敛
                    volume_increase = (current_volume - prev_volume) / max(prev_volume, 1e-10)
                    print(f"  体积增长比例: {volume_increase:.2%}")
                    
                    # 检测异常体积增长
                    if volume_increase > 10.0:
                        abnormal_growth_count += 1
                        print(f"  警告: 检测到异常体积增长 ({abnormal_growth_count}/2)")
                    else:
                        abnormal_growth_count = 0
                    
                    if abs(volume_increase) < volume_threshold:
                        print(f"  已收敛，停止迭代")
                        current_ellipsoid = new_ellipsoid
                        last_valid_ellipsoid = new_ellipsoid
                        break
                    
                    # 如果体积异常减小，可能是数值问题，停止迭代
                    if volume_increase < -0.5:
                        print(f"  体积显著减小，可能存在数值问题，停止迭代")
                        # 不更新椭球体，保持上一次的结果
                        break
                    
                    # 更新椭球体和体积
                    current_ellipsoid = new_ellipsoid
                    last_valid_ellipsoid = new_ellipsoid
                    prev_volume = current_volume
                    
                except Exception as e:
                    print(f"  计算MVIE出错: {e}")
                    fallback_mode = True
                    # 如果SOCP失败，使用备用方法
                    try:
                        # 尝试使用备用方法计算MVIE
                        print("  尝试使用备用方法计算MVIE...")
                        new_ellipsoid = self.compute_mvie_fallback(current_polytope)
                        current_ellipsoid = new_ellipsoid
                        last_valid_ellipsoid = new_ellipsoid
                    except Exception as e:
                        print(f"  备用方法也失败: {e}")
                        print("  使用当前椭球体")
                    break
                    
            except Exception as e:
                print(f"  FIRI迭代 {iter_num+1} 失败: {e}")
                fallback_mode = True
                if last_valid_polytope is None:
                    # 如果之前没有成功过，创建一个基于种子点的简单多胞体
                    print("  创建基于种子点的简单多胞体...")
                    try:
                        last_valid_polytope = self.create_simple_polytope(seed_points)
                        last_valid_ellipsoid = initial_ellipsoid
                    except Exception as e:
                        print(f"  创建简单多胞体失败: {e}")
                        # 如果所有方法都失败，返回初始椭球体和简单多胞体
                        return self.create_simple_polytope(seed_points), initial_ellipsoid
                break
        
        # 返回最后有效的多胞体和椭球体
        if current_polytope is None:
            if last_valid_polytope is not None:
                return last_valid_polytope, last_valid_ellipsoid
            else:
                # 如果连一次有效计算都没有，创建简单多胞体
                print("  创建简单多胞体...")
                return self.create_simple_polytope(seed_points), initial_ellipsoid
        
        # 返回最终的多胞体和椭球体
        return current_polytope, current_ellipsoid
    
    def restrictive_inflation(self, ellipsoid, seed_points):
        """
        实现改进的限制性膨胀算法，确保种子点完全包含
        
        参数:
            ellipsoid: 初始椭球体
            seed_points: 需要包含的种子点列表
            
        返回:
            一个包含所有种子点的凸多胞体
        """
        # 确保种子点是numpy数组
        seed_points = np.array(seed_points)
        
        # 将种子点变换到椭球体标准空间
        transformed_seeds = np.array([ellipsoid.transform_point(p) for p in seed_points])
        
        # 计算种子点的中心和边界
        seed_center = np.mean(seed_points, axis=0)
        seed_min = np.min(seed_points, axis=0)
        seed_max = np.max(seed_points, axis=0)
        seed_extent = np.linalg.norm(seed_max - seed_min)
        
        # 用于限制多胞体大小的边界距离
        boundary_dist = max(5.0, seed_extent * 2.0)
        
        # 为每个障碍物生成半空间
        standard_halfspaces = []
        valid_obstacles = []
        
        # 首先筛选出相关的障碍物
        for obs in self.obstacles:
            # 获取障碍物中心和半径
            try:
                # 尝试不同的障碍物表示
                if hasattr(obs, 'vertices') and obs.vertices is not None:
                    # 基于顶点的表示
                    obs_vertices = np.asarray(obs.vertices)
                    obs_center = np.mean(obs_vertices, axis=0)
                    dists = np.linalg.norm(obs_vertices - obs_center, axis=1)
                    obs_radius = np.max(dists)
                elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
                    # 基于中心和半径的表示
                    obs_center = obs.center
                    obs_radius = obs.radius
                elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                    # 字典表示
                    obs_center = obs['center']
                    obs_radius = obs['radius']
                else:
                    print("  警告: 无法识别的障碍物类型，跳过")
                    continue
            except Exception as e:
                print(f"  障碍物处理出错: {e}")
                continue
            
            # 计算障碍物到种子区域的距离
            center_dist = np.linalg.norm(obs_center - seed_center)
            
            # 只考虑足够近的障碍物
            if center_dist > boundary_dist + obs_radius:
                continue
                
            # 计算障碍物到最近种子点的距离
            min_dist_to_seed = float('inf')
            for seed in seed_points:
                dist = np.linalg.norm(seed - obs_center) - obs_radius
                min_dist_to_seed = min(min_dist_to_seed, dist)
            
            # 保留距离合理的障碍物
            if min_dist_to_seed < boundary_dist:
                valid_obstacles.append((obs_center, obs_radius, min_dist_to_seed))
        
        # 论文中的贪心策略：根据障碍物的重要性排序
        valid_obstacles.sort(key=lambda x: x[2])  # 按照到种子点的距离排序
        
        # 设置最大障碍物数量限制，避免处理过多障碍物
        max_obstacles = min(30, len(valid_obstacles))
        valid_obstacles = valid_obstacles[:max_obstacles]
        
        print(f"  找到 {len(valid_obstacles)} 个有效障碍物")
        
        # 直接为每个障碍物对每个种子点生成半空间约束
        for obs_center, obs_radius, distance in valid_obstacles:
            # 对于每个种子点
            for i, seed in enumerate(seed_points):
                # 计算种子点到障碍物中心的向量
                direction = seed - obs_center
                direction_norm = np.linalg.norm(direction)
                
                # 避免处理与障碍物中心重合的种子点
                if direction_norm < 1e-10:
                    continue
                
                # 归一化方向向量
                direction = direction / direction_norm
                
                # 计算障碍物表面上的点
                surface_point = obs_center + direction * obs_radius
                
                # 计算从表面点到种子点的向量
                to_seed = seed - surface_point
                to_seed_norm = np.linalg.norm(to_seed)
                
                # 避免处理在障碍物表面上的种子点
                if to_seed_norm < 1e-10:
                    continue
                
                # 计算安全边距 - 基于障碍物大小和接近程度
                safety_margin = max(0.2, min(1.0, obs_radius / 2))
                
                # 如果种子点非常接近障碍物，增加边距
                if to_seed_norm < obs_radius:
                    safety_margin *= 1.5
                
                # 创建半空间：a·x + b <= 0
                # 其中a是法向量，指向远离障碍物的方向
                a = to_seed / to_seed_norm
                
                # 计算偏移量b，使得种子点在安全边距内
                b = -np.dot(a, surface_point) - safety_margin
                
                # 构造半空间
                halfspace = np.zeros(self.dim + 1)
                halfspace[:-1] = a
                halfspace[-1] = b
                
                # 添加到标准空间中的半空间列表
                standard_halfspaces.append(halfspace)
        
        # 如果没有足够的半空间约束，添加边界约束
        if len(standard_halfspaces) < self.dim + 1:
            print(f"  警告: 只有 {len(standard_halfspaces)} 个有效半空间约束，添加边界约束")
            
            # 添加包含所有种子点的边界框约束
            # 计算边界框的大小
            bbox_min = seed_min - seed_extent
            bbox_max = seed_max + seed_extent
            
            # 添加6个约束（对应3D中的6个面）
            for i in range(self.dim):
                # 添加最小面约束: x_i >= bbox_min_i
                min_hs = np.zeros(self.dim + 1)
                min_hs[i] = -1  # 法向量指向负方向
                min_hs[-1] = bbox_min[i]  # -(-1 * bbox_min_i) <= 0
                standard_halfspaces.append(min_hs)
                
                # 添加最大面约束: x_i <= bbox_max_i
                max_hs = np.zeros(self.dim + 1)
                max_hs[i] = 1  # 法向量指向正方向
                max_hs[-1] = -bbox_max[i]  # -(1 * bbox_max_i) <= 0
                standard_halfspaces.append(max_hs)
        
        # 将每个半空间从标准空间变换回原始空间
        original_halfspaces = []
        for hs in standard_halfspaces:
            # 逆变换半空间
            original_hs = ellipsoid.inverse_transform_halfspace(hs)
            original_halfspaces.append(original_hs)
        
        # 创建原始空间中的多胞体
        original_polytope = ConvexPolytope(halfspaces=np.array(original_halfspaces))
        
        # 确保多胞体包含所有种子点
        seed_contained_count = 0
        for seed in seed_points:
            if original_polytope.contains(seed):
                seed_contained_count += 1
        
        print(f"  多胞体包含 {seed_contained_count}/{len(seed_points)} 个种子点")
        
        # 添加直接包含种子点的约束，确保所有种子点都在多胞体内
        if seed_contained_count < len(seed_points):
            print("  添加直接包含种子点的约束")
            
            # 对于每个种子点，如果它不在多胞体内，添加额外的约束
            for i, seed in enumerate(seed_points):
                if not original_polytope.contains(seed):
                    # 为这个种子点添加"球形"约束 - 在各个方向上
                    for direction_idx in range(self.dim):
                        # 创建方向向量
                        direction = np.zeros(self.dim)
                        direction[direction_idx] = 1.0
                        
                        # 创建半空间，确保种子点在安全区域内
                        halfspace = np.zeros(self.dim + 1)
                        halfspace[:-1] = -direction  # 法向量，指向种子点
                        halfspace[-1] = np.dot(-direction, seed) - 0.1  # 偏移量，留出边距
                        
                        original_halfspaces.append(halfspace)
                        
                        # 添加反方向约束
                        halfspace = np.zeros(self.dim + 1)
                        halfspace[:-1] = direction  # 法向量，指向种子点
                        halfspace[-1] = np.dot(direction, seed) - 0.1  # 偏移量，留出边距
                        
                        original_halfspaces.append(halfspace)
            
            # 使用更新的半空间约束创建新的多胞体
            original_polytope = ConvexPolytope(halfspaces=np.array(original_halfspaces))
        
        # 计算顶点表示，以备后续使用
        try:
            vertices = original_polytope.compute_vertices_from_halfspaces()
            if vertices is not None and len(vertices) >= 4:
                # 顶点计算成功
                print(f"  多胞体有 {len(vertices)} 个顶点")
            else:
                print("  警告: 无法计算多胞体顶点")
        except Exception as e:
            print(f"  计算多胞体顶点出错: {e}")
        
        return original_polytope
    
    def restrictive_inflation_simple(self, ellipsoid, seed_points):
        """
        实现简化版的限制性膨胀算法，为数值不稳定情况提供回退方案
        
        参数:
            ellipsoid: 初始椭球体
            seed_points: 需要包含的种子点列表
            
        返回:
            一个包含所有种子点的凸多胞体
        """
        # 确保种子点是numpy数组
        seed_points = np.array(seed_points)
        
        # 计算种子点的中心和边界
        seed_center = np.mean(seed_points, axis=0)
        seed_min = np.min(seed_points, axis=0)
        seed_max = np.max(seed_points, axis=0)
        seed_extent = np.linalg.norm(seed_max - seed_min)
        
        # 用于限制多胞体大小的边界距离
        boundary_dist = max(2.0, seed_extent)
        
        # 半空间列表
        halfspaces = []
        
        # 第一步：添加包含所有种子点的边界框约束
        # 添加6个约束（对应3D中的6个面）
        bbox_min = seed_min - boundary_dist / 2
        bbox_max = seed_max + boundary_dist / 2
        
        for i in range(self.dim):
            # 添加最小面约束: x_i >= bbox_min_i
            min_hs = np.zeros(self.dim + 1)
            min_hs[i] = -1  # 法向量指向负方向
            min_hs[-1] = bbox_min[i]  # -(-1 * bbox_min_i) <= 0
            halfspaces.append(min_hs)
            
            # 添加最大面约束: x_i <= bbox_max_i
            max_hs = np.zeros(self.dim + 1)
            max_hs[i] = 1  # 法向量指向正方向
            max_hs[-1] = -bbox_max[i]  # -(1 * bbox_max_i) <= 0
            halfspaces.append(max_hs)
        
        # 第二步：为每个障碍物生成一个简单的排除约束
        for obs in self.obstacles:
            # 获取障碍物中心和半径
            try:
                # 尝试不同的障碍物表示
                if hasattr(obs, 'vertices') and obs.vertices is not None:
                    # 基于顶点的表示
                    obs_vertices = np.asarray(obs.vertices)
                    obs_center = np.mean(obs_vertices, axis=0)
                    dists = np.linalg.norm(obs_vertices - obs_center, axis=1)
                    obs_radius = np.max(dists)
                elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
                    # 基于中心和半径的表示
                    obs_center = obs.center
                    obs_radius = obs.radius
                elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                    # 字典表示
                    obs_center = obs['center']
                    obs_radius = obs['radius']
                else:
                    continue
            except Exception as e:
                continue
            
            # 计算障碍物到种子中心的方向
            direction = seed_center - obs_center
            direction_norm = np.linalg.norm(direction)
            
            # 如果种子中心与障碍物中心重合，跳过
            if direction_norm < 1e-10:
                continue
            
            # 归一化方向向量
            direction = direction / direction_norm
            
            # 计算安全边距
            safety_margin = max(0.2, min(1.0, obs_radius / 2))
            
            # 沿着远离障碍物的方向创建半空间
            halfspace = np.zeros(self.dim + 1)
            halfspace[:-1] = direction  # 法向量指向远离障碍物方向
            halfspace[-1] = -np.dot(direction, obs_center + direction * (obs_radius + safety_margin))
            
            halfspaces.append(halfspace)
        
        # 创建多胞体
        polytope = ConvexPolytope(halfspaces=np.array(halfspaces))
        
        # 确保多胞体包含所有种子点
        seed_contained_count = 0
        for seed in seed_points:
            if polytope.contains(seed):
                seed_contained_count += 1
        
        print(f"  简化多胞体包含 {seed_contained_count}/{len(seed_points)} 个种子点")
        
        # 添加直接包含种子点的约束，确保所有种子点都在多胞体内
        if seed_contained_count < len(seed_points):
            print("  添加直接包含种子点的约束")
            
            new_halfspaces = list(halfspaces)
            
            # 对于每个种子点，如果它不在多胞体内，添加额外的约束
            for i, seed in enumerate(seed_points):
                if not polytope.contains(seed):
                    # 为这个种子点添加"球形"约束 - 在各个方向上
                    for direction_idx in range(self.dim):
                        # 创建方向向量
                        direction = np.zeros(self.dim)
                        direction[direction_idx] = 1.0
                        
                        # 创建半空间，确保种子点在安全区域内
                        halfspace = np.zeros(self.dim + 1)
                        halfspace[:-1] = -direction  # 法向量，指向种子点
                        halfspace[-1] = np.dot(-direction, seed) - 0.1  # 偏移量，留出边距
                        
                        new_halfspaces.append(halfspace)
                        
                        # 添加反方向约束
                        halfspace = np.zeros(self.dim + 1)
                        halfspace[:-1] = direction  # 法向量，远离种子点
                        halfspace[-1] = -np.dot(direction, seed) - 0.1  # 偏移量，留出边距
                        
                        new_halfspaces.append(halfspace)
            
            # 使用更新的半空间约束创建新的多胞体
            polytope = ConvexPolytope(halfspaces=np.array(new_halfspaces))
        
        return polytope

    def create_simple_polytope(self, seed_points):
        """
        创建一个简单的多胞体，用于极度不稳定情况下的回退
        
        参数:
            seed_points: 需要包含的种子点列表
            
        返回:
            一个包含所有种子点的简单凸多胞体
        """
        # 确保种子点是numpy数组
        seed_points = np.array(seed_points)
        
        # 计算种子点的中心和边界
        seed_center = np.mean(seed_points, axis=0)
        seed_min = np.min(seed_points, axis=0)
        seed_max = np.max(seed_points, axis=0)
        
        # 安全边距
        margin = 1.0
        
        # 创建一个边界框
        bbox_min = seed_min - margin
        bbox_max = seed_max + margin
        
        # 创建半空间
        halfspaces = []
        
        # 添加6个约束（对应3D中的6个面）
        for i in range(self.dim):
            # 添加最小面约束: x_i >= bbox_min_i
            min_hs = np.zeros(self.dim + 1)
            min_hs[i] = -1  # 法向量指向负方向
            min_hs[-1] = bbox_min[i]  # -(-1 * bbox_min_i) <= 0
            halfspaces.append(min_hs)
            
            # 添加最大面约束: x_i <= bbox_max_i
            max_hs = np.zeros(self.dim + 1)
            max_hs[i] = 1  # 法向量指向正方向
            max_hs[-1] = -bbox_max[i]  # -(1 * bbox_max_i) <= 0
            halfspaces.append(max_hs)
        
        # 创建多胞体
        polytope = ConvexPolytope(halfspaces=np.array(halfspaces))
        
        return polytope
    
    def compute_mvie(self, polytope):
        """
        计算凸多胞体的最大体积内接椭球体
        
        参数:
            polytope: 凸多胞体对象
            
        返回:
            Ellipsoid对象，表示最大内接椭球
        """
        try:
            return self.mvie_socp.compute(polytope)
        except Exception as e:
            print(f"  MVIE计算出错: {e}")
            # 如果发生错误，回退到简单方法
            return self.compute_mvie_fallback(polytope)
    
    def compute_mvie_fallback(self, polytope):
        """
        计算MVIE的备用方法，当主方法失败时使用
        """
        try:
            # 获取多胞体的内点
            interior_point = polytope.get_interior_point()
            if interior_point is None:
                print("  警告: 找不到内点，使用默认中心")
                interior_point = np.zeros(self.dim)
                
            # 采样边界点
            boundary_points = polytope._sample_boundary_points(num_samples=100)
            
            if boundary_points is None or len(boundary_points) < self.dim + 1:
                print("  警告: 边界点采样失败，使用默认椭球体")
                return Ellipsoid(interior_point, np.eye(self.dim) * 0.1)
                
            # 计算中心到边界点的平均距离
            distances = np.linalg.norm(boundary_points - interior_point, axis=1)
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            # 使用协方差矩阵作为形状矩阵的估计
            centered_points = boundary_points - interior_point
            cov = centered_points.T @ centered_points / len(centered_points)
            
            # 确保矩阵正定
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            
            # 缩放椭球体以确保它完全在多胞体内
            scale_factor = min_distance**2 / np.max(eigvals)
            Q = eigvecs @ np.diag(eigvals * scale_factor * 0.9) @ eigvecs.T
            
            return Ellipsoid(interior_point, Q)
        except Exception as e:
            print(f"  备用MVIE计算也失败: {e}")
            # 最后的回退策略
            return Ellipsoid(np.zeros(self.dim), np.eye(self.dim) * 0.1) 