import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import warnings

class ConvexPolytope:
    def __init__(self, halfspaces=None, points=None):
        """
        创建一个凸多胞体
        可以使用半空间表示或点表示
        参数:
            halfspaces: 半空间表示，每行为 [a_1, a_2, ..., a_n, b]，表示 a·x + b <= 0
            points: 点表示，凸包的顶点
        """
        self.halfspaces = np.array(halfspaces) if halfspaces is not None else None
        self.points = np.array(points) if points is not None else None
        self.dimension = (halfspaces.shape[1] - 1) if halfspaces is not None else \
                        (points.shape[1] if points is not None else 0)
        self.interior_point = None  # 缓存内点计算结果
        
    def to_mesh(self):
        """
        将多胞体转换为三角网格，用于可视化
        返回Open3D TriangleMesh对象
        """
        try:
            import open3d as o3d
            
            # 尝试获取顶点表示
            vertices = self.get_vertices()
            
            # 如果没有顶点或顶点数量不足，尝试从半空间计算
            if vertices is None or len(vertices) < 4:
                try:
                    vertices = self.compute_vertices_from_halfspaces()
                    if vertices is None:
                        print("警告: 计算多胞体顶点失败")
                        vertices = []
                except Exception as e:
                    print(f"计算多胞体顶点失败: {e}")
                    vertices = []
                
                # 如果仍然没有足够的顶点，尝试采样边界点
                if len(vertices) < 4:
                    try:
                        boundary_points = self._sample_boundary_points(num_samples=500)
                        if boundary_points is not None and len(boundary_points) >= 4:
                            vertices = boundary_points
                        else:
                            print("警告: 无法生成足够的边界点以创建网格")
                    except Exception as e:
                        print(f"采样边界点失败: {e}")
            
            # 确保顶点是列表或数组类型且不为None
            if vertices is None:
                vertices = []
            
            # 转换为numpy数组
            vertices = np.array(vertices)
            
            # 检查顶点数量是否足够
            if len(vertices) < 4:
                print(f"警告: 顶点数量不足，仅有 {len(vertices)} 个，使用近似网格")
                
                # 获取内部点作为中心
                center = self.get_interior_point()
                if center is None:
                    center = np.zeros(3)
                
                # 创建一个小球作为替代
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                sphere.translate(center)
                sphere.paint_uniform_color([0, 1, 0])  # 绿色
                sphere.compute_vertex_normals()
                return sphere
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            
            # 尝试使用几种不同的方法创建三角网格
            mesh = None
            
            # 方法1: Alpha Shape
            try:
                # 根据点云密度自适应选择alpha值
                if len(vertices) >= 10:
                    dists = []
                    for i in range(min(10, len(vertices))):
                        for j in range(i+1, min(11, len(vertices))):
                            dists.append(np.linalg.norm(vertices[i] - vertices[j]))
                    avg_dist = np.mean(dists) if dists else 1.0
                else:
                    avg_dist = 1.0
                
                alpha = max(0.5, avg_dist * 2.0)
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                    mesh.compute_vertex_normals()
                    return mesh
            except Exception as e:
                print(f"Alpha shape网格生成失败: {e}")
            
            # 方法2: 凸包
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
                    mesh.compute_vertex_normals()
                    return mesh
            except Exception as e:
                print(f"凸包网格生成失败: {e}")
            
            # 方法3: 如果前两种方法失败，使用球形配置创建近似网格
            if mesh is None or not hasattr(mesh, 'triangles') or len(mesh.triangles) == 0:
                try:
                    # 计算中心点和到中心点的平均距离
                    center = np.mean(vertices, axis=0)
                    dists = np.linalg.norm(vertices - center, axis=1)
                    avg_radius = np.mean(dists) if len(dists) > 0 else 1.0
                    
                    # 创建球体并略微收缩多胞体
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=avg_radius*0.9)
                    sphere.translate(center)
                    sphere.paint_uniform_color([0, 1, 0])  # 绿色
                    sphere.compute_vertex_normals()
                    return sphere
                except Exception as e:
                    print(f"球形近似网格生成失败: {e}")
                    
                    # 最后的回退策略：返回默认球
                    default_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                    if len(vertices) > 0:
                        default_sphere.translate(vertices[0])  # 使用第一个顶点作为中心
                    default_sphere.paint_uniform_color([0, 1, 0])
                    default_sphere.compute_vertex_normals()
                    return default_sphere
            
            return mesh
            
        except ImportError:
            print("警告: 未安装Open3D库，无法创建可视化网格")
            return None
        except Exception as e:
            print(f"多胞体网格生成出错: {e}")
            return None
            
    def get_vertices(self):
        """获取多胞体的顶点表示"""
        if self.points is not None:
            return self.points
        elif self.halfspaces is not None:
            return self.compute_vertices_from_halfspaces()
        return None
            
    def _sample_boundary_points(self, num_samples=1000):
        """
        在多胞体边界上均匀采样点
        参数:
            num_samples: 采样点数量
        返回:
            边界点列表
        """
        if self.points is None and self.halfspaces is None:
            return None
            
        # 获取内点
        interior_point = self.get_interior_point()
        if interior_point is None:
            print("无法找到内点，边界采样失败")
            return None
            
        # 均匀采样方向
        # 使用黄金螺旋点序列在球面上生成均匀分布的方向
        indices = np.arange(0, num_samples, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_samples)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        directions = np.vstack((x, y, z)).T
        
        # 确保方向向量的维度与多胞体维度一致
        if self.dimension != 3:
            # 如果维度不是3，则需要调整
            if self.dimension == 2:
                directions = directions[:, :2]  # 只取x和y
            else:
                # 生成相应维度的均匀方向向量
                directions = np.random.randn(num_samples, self.dimension)
                # 归一化
                norms = np.linalg.norm(directions, axis=1)
                directions = directions / norms[:, np.newaxis]
        
        # 沿每个方向射线找到边界点
        boundary_points = []
        
        if self.halfspaces is not None:
            # 使用半空间表示
            for direction in directions:
                # 计算与每个半空间的交点
                intersections = []
                
                for i, hs in enumerate(self.halfspaces):
                    a = hs[:-1]  # 法向量
                    b = -hs[-1]  # 偏移量
                    
                    # 计算射线与平面的交点参数 t
                    # p + t*d 与平面 a·x = b 的交点
                    denum = np.dot(direction, a)
                    
                    # 如果射线与面平行，则跳过
                    if abs(denum) < 1e-10:
                        continue
                        
                    # 计算交点参数
                    t = (b - np.dot(interior_point, a)) / denum
                    
                    # 只考虑正方向的交点
                    if t > 0:
                        intersections.append((t, i))
                
                # 如果找到交点，取最近的
                if intersections:
                    t_min = min(intersections)[0]
                    point = interior_point + t_min * direction
                    boundary_points.append(point)
        
        elif self.points is not None:
            # 使用点表示
            try:
                hull = ConvexHull(self.points)
                vertices = self.points[hull.vertices]
                
                # 对每个方向，找到最远的顶点（支撑点）
                for direction in directions:
                    # 计算方向与所有顶点的点积
                    projections = np.dot(vertices, direction)
                    # 找到最大投影
                    max_idx = np.argmax(projections)
                    boundary_points.append(vertices[max_idx])
            except:
                # 如果凸包计算失败，使用射线投射法
                for direction in directions:
                    # 找到最远的点
                    max_dist = -float('inf')
                    farthest_point = None
                    
                    for point in self.points:
                        # 计算从内点到当前点的向量
                        vec = point - interior_point
                        # 计算在方向上的投影
                        proj = np.dot(vec, direction)
                        
                        if proj > max_dist:
                            max_dist = proj
                            farthest_point = point
                    
                    if farthest_point is not None:
                        boundary_points.append(farthest_point)
        
        return np.array(boundary_points) if boundary_points else None
    
    def get_halfspaces(self):
        """获取多胞体的半空间表示"""
        if self.halfspaces is not None:
            return self.halfspaces
        elif self.points is not None:
            try:
                # 使用凸包计算半空间表示
                hull = ConvexHull(self.points)
                # 从凸包的方程中提取半空间表示
                halfspaces = np.zeros((len(hull.equations), self.dimension + 1))
                halfspaces[:,:-1] = hull.equations[:,:-1]  # 法向量
                halfspaces[:,-1] = hull.equations[:,-1]    # 偏移量
                return halfspaces
            except Exception as e:
                print(f"计算半空间表示出错: {e}")
                
        return None
    
    def get_interior_point(self):
        """
        获取多胞体内部的一个点
        使用多种方法尝试查找内点
        
        返回:
            内点坐标，如果找不到则返回None
        """
        # 如果已经有内点属性，直接返回
        if hasattr(self, 'interior_point') and self.interior_point is not None:
            return self.interior_point
            
        # 如果顶点可用，使用顶点的凸组合
        if self.vertices is not None and len(self.vertices) > 0:
            interior_point = np.mean(self.vertices, axis=0)
            if self.contains(interior_point):
                self.interior_point = interior_point
                return interior_point
                
        # 尝试找到Chebyshev中心（最大内接球的中心）
        interior_point = self._compute_chebyshev_center()
        if interior_point is not None and self.contains(interior_point):
            self.interior_point = interior_point
            return interior_point
            
        # 尝试线性规划寻找内点
        try:
            if self.halfspaces is not None and len(self.halfspaces) > 0:
                # 求解线性规划问题: 最大化自由度t，满足Ax + t*1 <= b
                A = self.halfspaces[:, :-1]
                b = -self.halfspaces[:, -1]  # 注意符号变换
                
                # 过滤掉范数太小的约束
                norms = np.linalg.norm(A, axis=1)
                valid_indices = norms > 1e-10
                if np.sum(valid_indices) > 0:
                    A = A[valid_indices]
                    b = b[valid_indices]
                    
                    # 线性规划问题的约束矩阵和右端
                    m, n = A.shape
                    A_lp = np.hstack([A, np.ones((m, 1))])
                    
                    # 目标：最大化t
                    c = np.zeros(n + 1)
                    c[-1] = -1  # 最大化t
                    
                    # 求解线性规划
                    from scipy.optimize import linprog
                    
                    # 使用高级求解器
                    # 尝试不同的方法，提高求解成功率
                    for method in ['highs', 'highs-ds', 'highs-ipm', 'interior-point']:
                        try:
                            result = linprog(c, A_ub=A_lp, b_ub=b, method=method)
                            if result.success and result.x[-1] > -1e-6:  # t值大于-1e-6表示可行
                                interior_point = result.x[:-1]
                                if self.contains(interior_point):
                                    self.interior_point = interior_point
                                    return interior_point
                        except Exception as e:
                            print(f"方法{method}求解出错: {e}")
                            continue
        except Exception as e:
            print(f"线性规划寻找内点出错: {e}")
        
        # 随机采样寻找内点
        interior_point = self._random_sampling(max_attempts=10000)
        if interior_point is not None:
            self.interior_point = interior_point
            return interior_point
        
        # 如果以上方法都失败，尝试使用特别方法
        try:
            # 1. 如果有顶点，使用顶点的平均值（通常会落在内部）
            if hasattr(self, 'vertices') and self.vertices is not None and len(self.vertices) > 0:
                # 使用加权平均，避免某些极端顶点的影响
                weights = np.ones(len(self.vertices)) / len(self.vertices)
                interior_candidate = np.sum(self.vertices * weights.reshape(-1, 1), axis=0)
                
                # 检查这个候选点是否在多胞体内
                if self.contains(interior_candidate):
                    self.interior_point = interior_candidate
                    return interior_candidate
                
                # 如果不在内部，向原点方向收缩
                for scale in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]:
                    centroid = np.mean(self.vertices, axis=0)
                    interior_candidate = scale * centroid
                    if self.contains(interior_candidate):
                        self.interior_point = interior_candidate
                        return interior_candidate
                
            # 2. 如果有半空间，尝试使用均匀收缩
            if self.halfspaces is not None and len(self.halfspaces) > 0:
                # 记录归一化的半空间表示
                A = self.halfspaces[:, :-1]
                b = -self.halfspaces[:, -1]  # 注意符号变换
                
                # 过滤掉范数接近零的行
                norms = np.linalg.norm(A, axis=1)
                valid_indices = norms > 1e-10
                if np.sum(valid_indices) > 0:
                    A = A[valid_indices]
                    b = b[valid_indices]
                    
                    # 计算原点在半空间中的位置
                    margins = b - A @ np.zeros(A.shape[1])
                    
                    # 如果原点在所有半空间内，返回原点
                    if np.all(margins >= 0):
                        interior_point = np.zeros(A.shape[1])
                        self.interior_point = interior_point
                        return interior_point
                    
                    # 如果原点不在多胞体内，尝试找到一个偏移的点
                    # 计算所有半空间的平均法向量（指向多胞体外部）
                    avg_normal = np.mean(A, axis=0)
                    avg_normal_norm = np.linalg.norm(avg_normal)
                    
                    if avg_normal_norm > 1e-10:
                        # 往平均法向量的反方向移动
                        for scale in [2.0, 5.0, 10.0, 20.0, 50.0]:
                            interior_candidate = -scale * avg_normal / avg_normal_norm
                            if self.contains(interior_candidate):
                                self.interior_point = interior_candidate
                                return interior_candidate
                                
        except Exception as e:
            print(f"特别内点查找方法出错: {e}")
        
        # 所有方法都失败，使用默认值
        import warnings
        warnings.warn("未能找到多胞体内点，返回默认点")
        
        # 返回原点作为默认值
        default_point = np.zeros(self.dimension)
        return default_point
    
    def _compute_chebyshev_center(self):
        """计算多胞体的Chebyshev中心"""
        if self.halfspaces is None or len(self.halfspaces) < self.dimension + 1:
            return None
            
        try:
            A = self.halfspaces[:, :-1]  # 法向量
            b = -self.halfspaces[:, -1]  # 偏移量
            
            # 规范化法向量
            norms = np.linalg.norm(A, axis=1)
            valid_indices = norms > 1e-10
            
            if np.sum(valid_indices) <= self.dimension:
                return None  # 不足以确定中心
                
            A_norm = A[valid_indices] / norms[valid_indices, np.newaxis]
            b_norm = b[valid_indices] / norms[valid_indices]
            
            # Chebyshev中心线性规划问题
            # 最大化与所有半空间的最小距离
            n = A_norm.shape[1]  # 空间维度
            
            # 目标函数: [x_1, x_2, ..., x_n, r]，最大化r
            c = np.zeros(n+1)
            c[-1] = -1  # 最大化r
            
            # 不等式约束: A_i·x + ||A_i||·r ≤ b_i
            # 重写为 A_i·x + r ≤ b_i (因为A_i已归一化)
            A_lp = np.column_stack([A_norm, np.ones(len(A_norm))])
            
            # 使用线性规划解决
            # 注意：使用了SciPy的新接口
            result = linprog(c, A_ub=A_lp, b_ub=b_norm, bounds=(None, None), method='highs')
            
            if result.success:
                center = result.x[:-1]  # 提取中心坐标（不包括半径r）
                return center
            else:
                return None
        except Exception as e:
            print(f"计算Chebyshev中心出错: {e}")
            return None
    
    def _random_sampling(self, max_attempts=1000):
        """通过随机采样寻找内点"""
        if self.halfspaces is None and self.points is None:
            return None
            
        # 估计多胞体范围
        if self.points is not None:
            min_coords = np.min(self.points, axis=0)
            max_coords = np.max(self.points, axis=0)
        else:
            # 如果没有点表示，使用默认范围
            min_coords = np.array([-10.0] * self.dimension)
            max_coords = np.array([10.0] * self.dimension)
            
        # 增加搜索范围
        range_size = max_coords - min_coords
        min_coords = min_coords - 0.1 * range_size
        max_coords = max_coords + 0.1 * range_size
        
        # 随机采样
        for _ in range(max_attempts):
            # 生成随机点
            point = min_coords + np.random.random(self.dimension) * (max_coords - min_coords)
            
            # 检查点是否在多胞体内
            if self.contains(point):
                return point
                
        # 如果随机采样失败，尝试顶点凸组合
        if self.points is not None:
            n_points = len(self.points)
            
            # 简单平均
            interior = np.mean(self.points, axis=0)
                
            if self.contains(interior):
                return interior
                
            # 如果简单平均不在内部，尝试更复杂的凸组合
            for _ in range(max_attempts):
                weights = np.random.random(n_points)
                weights = weights / np.sum(weights)
                
                interior = np.zeros(self.points[0].shape)
                for i, p in enumerate(self.points):
                    interior += weights[i] * p
                    
                if self.contains(interior):
                    return interior
                    
        return None
    
    def contains(self, point):
        """检查点是否在凸多胞体内"""
        point = np.array(point)
        
        if self.halfspaces is not None:
            # 使用半空间表示检查
            for hs in self.halfspaces:
                a = hs[:-1]
                b = hs[-1]  # b in ax + b <= 0
                if np.dot(a, point) + b > 1e-10:  # 添加容错
                    return False
            return True
        elif self.points is not None:
            # 尝试使用点表示检查
            try:
                hull = ConvexHull(self.points)
                test_point = np.array([point])
                new_hull = ConvexHull(np.vstack([self.points, test_point]))
                return len(new_hull.vertices) == len(hull.vertices)
            except:
                # 如果凸包计算失败，使用另一种方法
                try:
                    # 判断点是否在所有顶点的凸组合内
                    from scipy.optimize import nnls
                    
                    # 求解 min ||Ax - b||^2 subject to x >= 0, sum(x) = 1
                    # 其中 A 是顶点矩阵，b 是点
                    A = self.points.T
                    b = point
                    
                    # 先求解非负最小二乘问题
                    x, residual = nnls(A, b)
                    
                    # 归一化权重
                    if np.sum(x) > 0:
                        x = x / np.sum(x)
                        
                    # 计算近似点
                    approx_point = A @ x
                    
                    # 如果残差很小，认为点在内部
                    return np.linalg.norm(approx_point - b) < 1e-6
                except:
                    return False
        return False
    
    def compute_vertices_from_halfspaces(self):
        """
        从半空间表示计算多胞体顶点
        
        返回:
            成功时返回顶点列表，失败时返回None
        """
        if self.halfspaces is None or len(self.halfspaces) < self.dimension + 1:
            print("警告: 半空间约束不足，无法计算顶点")
            return None
            
        # 1. 首先找到一个内点
        interior_point = self.get_interior_point()
        
        if interior_point is None:
            print("警告: 无法找到内点，无法计算顶点")
            return None
            
        # 2. 使用射线法计算顶点
        try:
            vertices = self._compute_vertices_by_ray_casting(interior_point)
            
            if vertices is not None and len(vertices) > 0:
                self.vertices = vertices
                return vertices
        except Exception as e:
            print(f"射线法计算顶点出错: {e}")
            
        # 3. 如果射线法失败，尝试通过面的交点计算
        try:
            A = self.halfspaces[:, :-1]
            b = -self.halfspaces[:, -1]
            
            # 首先检查并过滤冗余约束
            A_filtered, b_filtered = self._filter_redundant_halfspaces(A, b)
            
            # 如果过滤后的约束数大于维度，尝试计算顶点
            if len(b_filtered) > self.dimension:
                vertices = self._compute_vertices_from_face_intersections(A_filtered, b_filtered)
                if vertices is not None and len(vertices) > 0:
                    self.vertices = vertices
                    return vertices
        except Exception as e:
            print(f"面交点法计算顶点出错: {e}")
            
        # 4. 如果以上方法都失败，使用采样法
        try:
            # 采样多胞体边界上的点
            boundary_points = self._sample_boundary_points(num_samples=5000)
            
            if boundary_points is not None and len(boundary_points) > 0:
                # 使用凸包算法提取顶点
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(boundary_points)
                    vertices = boundary_points[hull.vertices]
                    self.vertices = vertices
                    return vertices
                except Exception as e:
                    print(f"凸包计算出错: {e}")
                    # 直接返回采样点
                    self.vertices = np.array(boundary_points)
                    return self.vertices
        except Exception as e:
            print(f"采样边界点出错: {e}")
            
        # 所有方法都失败，返回一个默认的边界框
        print("警告: 所有顶点计算方法都失败，使用默认边界框")
        
        # 创建一个默认的边界框
        if self.dimension == 3:
            # 创建一个简单的立方体
            corners = [
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
            ]
            self.vertices = np.array(corners) * 10.0  # 一个较大的立方体
            return self.vertices
        elif self.dimension == 2:
            # 创建一个简单的正方形
            corners = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
            self.vertices = np.array(corners) * 10.0
            return self.vertices
        else:
            # 任意维度的超立方体
            import itertools
            corners = list(itertools.product([-1, 1], repeat=self.dimension))
            self.vertices = np.array(corners) * 10.0
            return self.vertices
    
    def _filter_redundant_halfspaces(self, A, b):
        """过滤冗余或近似重复的半空间"""
        n = len(A)
        keep_indices = []
        
        for i in range(n):
            # 默认保留第一个约束
            if i == 0:
                keep_indices.append(i)
                continue
                
            # 检查当前约束与之前保留的约束是否近似平行
            is_redundant = False
            for j in keep_indices:
                cos_angle = np.abs(np.dot(A[i], A[j]))
                
                # 如果夹角余弦接近1，检查偏移量差异
                if cos_angle > 0.99:
                    # 检查哪个约束更紧
                    if b[i] <= b[j]:
                        # i约束更紧或相等，可能替换j
                        is_redundant = False
                    else:
                        # j约束更紧，i是冗余的
                        is_redundant = True
                    break
            
            if not is_redundant:
                keep_indices.append(i)
                
        return keep_indices
            
    def _compute_vertices_by_ray_casting(self, interior_point):
        """
        使用射线投射法计算顶点
        从内点向各个方向发射射线，找到与多胞体边界的交点
        
        参数:
            interior_point: 多胞体内的一点
            
        返回:
            顶点列表
        """
        if self.halfspaces is None:
            return None
            
        # 生成均匀分布在单位球面上的方向
        num_directions = min(5000, 100 * self.dimension)  # 适应维度的方向数量
        directions = []
        
        if self.dimension == 3:
            # 使用螺旋点法在球面上生成更均匀的点
            golden_ratio = (1 + 5 ** 0.5) / 2
            indices = np.arange(num_directions)
            theta = 2 * np.pi * indices / golden_ratio
            phi = np.arccos(1 - 2 * (indices + 0.5) / num_directions)
            
            # 转换为笛卡尔坐标
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            directions = np.vstack([x, y, z]).T
        elif self.dimension == 2:
            # 在二维中，均匀分布角度
            angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # 在任意维度中，使用正态分布采样
            # 多次采样，确保方向足够多且分布均匀
            while len(directions) < num_directions:
                # 生成随机方向
                direction = np.random.normal(size=self.dimension)
                norm = np.linalg.norm(direction)
                
                # 避免接近零向量
                if norm > 1e-10:
                    directions.append(direction / norm)
            
            directions = np.array(directions)
            
        # 对每个方向，求解射线与多胞体的交点
        intersection_points = []
        A = self.halfspaces[:, :-1]
        b = -self.halfspaces[:, -1]
        
        for direction in directions:
            # 计算射线 p(t) = interior_point + t * direction 与每个半空间的交点
            # 对于半空间 a^T x + b <= 0，相交时有 a^T (interior_point + t * direction) + b = 0
            # 求解 t: t = -(a^T interior_point + b) / (a^T direction)
            
            numerator = -(A @ interior_point + b)
            denominator = A @ direction
            
            # 只考虑向前的交点 (t > 0) 且避免除以零
            valid_indices = (denominator < -1e-10) | (denominator > 1e-10)
            
            if np.sum(valid_indices) > 0:
                t_values = np.zeros_like(numerator)
                t_values[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
                
                # 找到最小的正t值（最近的交点）
                positive_t = t_values[t_values > 1e-10]
                
                if len(positive_t) > 0:
                    min_t = np.min(positive_t)
                    intersection_point = interior_point + min_t * direction
                    
                    # 验证交点是否在多胞体上（满足所有半空间约束）
                    if self._is_vertex(intersection_point):
                        # 如果是顶点，将其添加到列表
                        intersection_points.append(intersection_point)
        
        # 过滤可能的重复顶点
        if len(intersection_points) > 0:
            from scipy.spatial import cKDTree
            
            # 使用KD树去除接近重复的点
            tree = cKDTree(intersection_points)
            unique_indices = []
            
            # 自适应容差
            tolerance = 1e-4 * np.max([np.linalg.norm(p) for p in intersection_points])
            tolerance = max(tolerance, 1e-6)  # 确保最小容差
            
            for i, point in enumerate(intersection_points):
                # 查找给定容差范围内的邻居
                indices = tree.query_ball_point(point, tolerance)
                # 只保留索引最小的点
                if i == min(indices):
                    unique_indices.append(i)
            
            return np.array([intersection_points[i] for i in unique_indices])
        
        return None
    
    def _compute_vertices_from_face_intersections(self, A, b):
        """通过计算面交点来找到顶点"""
        from itertools import combinations
        
        vertices = []
        n = len(A)
        
        # 遍历所有可能的d个面的组合
        for face_indices in combinations(range(n), self.dimension):
            # 提取这d个面的法向量和偏移量
            A_sub = A[list(face_indices)]
            b_sub = b[list(face_indices)]
            
            # 检查这些面是否线性独立（非奇异）
            if np.linalg.matrix_rank(A_sub) != self.dimension:
                continue
                
            # 求解线性方程组A_sub * x = b_sub
            try:
                vertex = np.linalg.solve(A_sub, b_sub)
                
                # 检查这个点是否在所有半空间内
                if self.contains(vertex):
                    # 检查是否已存在（避免数值误差导致的重复）
                    is_duplicate = False
                    for v in vertices:
                        if np.linalg.norm(vertex - v) < 1e-6:
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        vertices.append(vertex)
            except:
                # 如果线性系统无解或奇异，跳过
                continue
        
        return np.array(vertices) if vertices else None
    
    def _is_vertex(self, point):
        """检查点是否为多胞体的真实顶点"""
        # 确定在点上的面（即满足等式a·x+b=0的面）
        on_face_indices = []
        
        for i, hs in enumerate(self.halfspaces):
            a = hs[:-1]
            b = hs[-1]
            
            # 计算点到面的距离
            dist = np.abs(np.dot(a, point) + b) / np.linalg.norm(a)
            
            # 如果距离很小，认为点在这个面上
            if dist < 1e-8:
                on_face_indices.append(i)
        
        # 如果点至少在d个独立面的交点上，它就是顶点
        # 检查法向量的线性独立性
        if len(on_face_indices) >= self.dimension:
            # 提取这些面的法向量
            face_normals = self.halfspaces[on_face_indices, :-1]
            
            # 检查秩
            rank = np.linalg.matrix_rank(face_normals)
            return rank >= self.dimension
            
        return False 