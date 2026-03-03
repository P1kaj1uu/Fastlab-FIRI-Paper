import open3d as o3d
import numpy as np
import math
import os
import pickle
from scipy.optimize import linprog, minimize
from scipy.linalg import null_space, sqrtm, inv
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import time
from scipy.spatial import KDTree
import itertools

class ObstacleGenerator:
    def __init__(self, space_size=(10, 10, 10)):
        self.space_size = space_size
        self.obstacles = []
        self.inflated_obstacles = []

    def generate_random_obstacle(self, inflation=1.0):
        obstacle_type = np.random.choice(['sphere', 'cylinder', 'box'])
        position = np.random.rand(3) * self.space_size

        if obstacle_type == 'sphere':
            radius = np.random.uniform(0.5, 2)
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            inflated = o3d.geometry.TriangleMesh.create_sphere(radius=radius + inflation)

        elif obstacle_type == 'cylinder':
            radius = np.random.uniform(0.3, 1.5)
            height = np.random.uniform(1, 3)
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
            inflated = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius + inflation, height=height + inflation)

        elif obstacle_type == 'box':
            size = np.random.uniform(0.5, 2, 3)
            mesh = o3d.geometry.TriangleMesh.create_box(width=size[0],
                                                        height=size[1],
                                                        depth=size[2])
            inflated = o3d.geometry.TriangleMesh.create_box(
                width=size[0] + inflation,
                height=size[1] + inflation,
                depth=size[2] + inflation)

        mesh.translate(position)
        mesh.compute_vertex_normals()
        inflated.translate(position)
        inflated.compute_vertex_normals()
        return mesh, inflated

    def generate_strategic_obstacles(self, num_obstacles=30, start=None, goal=None):
        """
        生成战略性障碍物，确保起点和终点之间至少有一个障碍物，强制路径需要避障
        """
        obstacles = []
        inflated_obstacles = []
        
        # 记录已生成的障碍物中心位置
        obstacle_centers = []
        
        # 首先，确保在起点和终点之间放置一个障碍物
        if start is not None and goal is not None:
            # 计算起点到终点的路径
            direction = goal - start
            path_length = np.linalg.norm(direction)
            unit_direction = direction / path_length
            
            # 在路径的中段放置一个大型障碍物，确保路径必须绕行
            mid_point = start + 0.5 * direction
            
            # 为了确保障碍物不会完全阻挡路径，我们在中点附近随机偏移
            # 但确保仍在起点到终点的路线上
            offset = np.random.uniform(-0.3, 0.3, 3)
            # 使偏移垂直于主方向
            offset = offset - np.dot(offset, unit_direction) * unit_direction
            # 限制偏移大小
            if np.linalg.norm(offset) > 0.5:
                offset = offset / np.linalg.norm(offset) * 0.5
            
            strategic_position = mid_point + offset
            
            # 创建一个略大的障碍物
            obstacle_type = np.random.choice(['sphere', 'cylinder', 'box'])
            if obstacle_type == 'sphere':
                radius = np.random.uniform(1.5, 2.5)  # 较大的半径
                obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                inf_obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius + 1.0)
            elif obstacle_type == 'cylinder':
                radius = np.random.uniform(1.2, 2.0)
                height = np.random.uniform(2.0, 3.5)
                obs = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
                inf_obs = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=radius + 1.0, height=height + 1.0)
            else:  # box
                size = np.random.uniform(1.5, 2.5, 3)
                obs = o3d.geometry.TriangleMesh.create_box(width=size[0],
                                                         height=size[1],
                                                         depth=size[2])
                inf_obs = o3d.geometry.TriangleMesh.create_box(
                    width=size[0] + 1.0, height=size[1] + 1.0, depth=size[2] + 1.0)
            
            # 放置障碍物
            obs.translate(strategic_position)
            obs.compute_vertex_normals()
            inf_obs.translate(strategic_position)
            inf_obs.compute_vertex_normals()
            
            obstacles.append(obs)
            inflated_obstacles.append(inf_obs)
            obstacle_centers.append(strategic_position)
            
            print(f"策略性障碍物放置在 {strategic_position}，确保路径必须绕行")
        
        # 然后生成其他随机障碍物，但避免它们离起点或终点太近
        safe_radius_start = 2.0  # 起点周围的安全区域
        safe_radius_goal = 2.0   # 终点周围的安全区域
        
        # 限制路径绕行的空间范围，防止从障碍物外侧绕路
        path_corridor_width = 3.0  # 路径走廊的宽度
        
        # 生成其余障碍物
        remaining = num_obstacles - len(obstacles)
        for _ in range(remaining):
            # 生成随机位置
            position = np.random.rand(3) * self.space_size
            
            # 检查是否离起点或终点太近
            if start is not None and np.linalg.norm(position - start) < safe_radius_start:
                continue
            if goal is not None and np.linalg.norm(position - goal) < safe_radius_goal:
                continue
            
            # 检查是否在设定的路径走廊范围内
            if start is not None and goal is not None:
                # 计算点到起点-终点线段的距离
                v = goal - start
                v_length = np.linalg.norm(v)
                v_unit = v / v_length
                
                # 点到线的投影
                t = np.dot(position - start, v_unit)
                t = np.clip(t, 0, v_length)  # 确保投影点在线段上
                
                # 投影点
                proj = start + t * v_unit
                
                # 点到线的距离
                dist_to_line = np.linalg.norm(position - proj)
                
                # 如果障碍物在路径走廊之外且不是故意放置的额外障碍物，则跳过
                # 添加一些概率使部分障碍物可以在走廊外（为了增加环境多样性）
                outside_corridor = dist_to_line > path_corridor_width
                allow_outside = np.random.random() < 0.3  # 30%的概率允许在走廊外
                
                if outside_corridor and not allow_outside:
                    continue
                    
            # 生成障碍物
            obstacle_type = np.random.choice(['sphere', 'cylinder', 'box'])
            if obstacle_type == 'sphere':
                radius = np.random.uniform(0.5, 2.0)
                obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                inf_obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius + 1.0)
            elif obstacle_type == 'cylinder':
                radius = np.random.uniform(0.3, 1.5)
                height = np.random.uniform(1.0, 3.0)
                obs = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
                inf_obs = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=radius + 1.0, height=height + 1.0)
            else:
                size = np.random.uniform(0.5, 2.0, 3)
                obs = o3d.geometry.TriangleMesh.create_box(width=size[0],
                                                         height=size[1],
                                                         depth=size[2])
                inf_obs = o3d.geometry.TriangleMesh.create_box(
                    width=size[0] + 1.0, height=size[1] + 1.0, depth=size[2] + 1.0)
            
            # 放置障碍物
            obs.translate(position)
            obs.compute_vertex_normals()
            inf_obs.translate(position)
            inf_obs.compute_vertex_normals()
            
            obstacles.append(obs)
            inflated_obstacles.append(inf_obs)
            obstacle_centers.append(position)
            
            print(f"障碍物中心: {np.mean(np.asarray(obs.vertices), axis=0)}")
        
        return obstacles, inflated_obstacles


# FIRI算法的新增组件
class Ellipsoid:
    def __init__(self, center, Q=None):
        """
        创建一个椭球体
        E = {x | (x-c)^T Q^(-1) (x-c) <= 1}
        参数:
            center: 椭球体中心
            Q: 半正定矩阵，定义椭球体形状
        """
        self.center = np.array(center, dtype=float)
        self.dim = len(center)
        if Q is None:
            self.Q = np.eye(self.dim)  # 默认为单位球
        else:
            self.Q = np.array(Q, dtype=float)
        
        # 确保Q是半正定矩阵
        eigvals = np.linalg.eigvals(self.Q)
        if not np.all(eigvals >= 1e-10):
            print("警告: Q不是正定矩阵，将被调整为正定矩阵")
            min_eig = min(eigvals)
            if min_eig < 1e-10:
                self.Q = self.Q + (abs(min_eig) + 1e-6) * np.eye(self.dim)
        
        # 计算Q的逆矩阵，用于其他方法
        try:
            # 使用SVD分解计算逆矩阵，提高数值稳定性
            u, s, vh = np.linalg.svd(self.Q)
            
            # 过滤太小的奇异值
            s_inv = np.where(s > 1e-10, 1.0/s, 0.0)
            
            # 计算稳定的逆矩阵
            self.Q_inv = (vh.T * s_inv) @ u.T
        except:
            print("警告: 无法计算Q的逆矩阵，使用单位矩阵")
            self.Q_inv = np.eye(self.dim)

    def volume(self):
        """计算椭球体体积"""
        try:
            # 使用SVD分解计算行列式，提高数值稳定性
            u, s, vh = np.linalg.svd(self.Q)
            
            # 行列式是奇异值的乘积
            det_Q = np.prod(s)
            
            if det_Q <= 0:
                return 0.0
                
            return (4/3) * np.pi * np.sqrt(det_Q)
        except:
            return 0.0
    
    def contains(self, point):
        """检查点是否在椭球体内"""
        vec = point - self.center
        try:
            # 使用预计算的Q_inv
            return vec.T @ self.Q_inv @ vec <= 1
        except:
            # 如果Q不可逆，默认返回False
            return False
    
    def transform_point(self, point):
        """将点从世界坐标变换到椭球体标准坐标系（椭球体->单位球）"""
        try:
            # 使用SVD分解进行稳定的变换
            u, s, vh = np.linalg.svd(self.Q)
            
            # 过滤过小的奇异值，并计算平方根
            s_sqrt = np.sqrt(np.where(s > 1e-10, s, 1e-10))
            s_sqrt_inv = 1.0 / s_sqrt
            
            # 计算Q^(1/2)和Q^(-1/2)
            Q_sqrt = u @ np.diag(s_sqrt) @ vh
            Q_sqrt_inv = vh.T @ np.diag(s_sqrt_inv) @ u.T
            
            # 应用变换 x' = Q^(-1/2)(x - c)
            return Q_sqrt_inv @ (np.array(point) - self.center)
        except Exception as e:
            print(f"变换点时出错: {e}")
            return np.array(point) - self.center
    
    def inverse_transform_point(self, point):
        """将点从椭球体标准坐标系变换回世界坐标（单位球->椭球体）"""
        try:
            # 使用SVD分解进行稳定的变换
            u, s, vh = np.linalg.svd(self.Q)
            
            # 过滤过小的奇异值，并计算平方根
            s_sqrt = np.sqrt(np.where(s > 1e-10, s, 1e-10))
            
            # 计算Q^(1/2)
            Q_sqrt = u @ np.diag(s_sqrt) @ vh
            
            # 应用逆变换: x = Q^(1/2)y + c
            return self.center + Q_sqrt @ np.array(point)
        except Exception as e:
            print(f"逆变换点时出错: {e}")
            # 返回一个默认逆变换
            return self.center + np.array(point)
    
    def transform_halfspace(self, halfspace):
        """
        将半空间从原始坐标系变换到椭球体标准坐标系
        按照论文式(15)实现变换: {x | a^T x + b ≤ 0} 变换为 {y | a'^T y + b' ≤ 0}
        """
        a = halfspace[:-1]  # 法向量
        b = halfspace[-1]   # 偏移量
        
        # 确保Q_inv存在
        if not hasattr(self, 'Q_inv') or self.Q_inv is None:
            # 使用SVD稳定计算Q^(-1/2)
            U, s, Vh = np.linalg.svd(self.Q, full_matrices=False)
            
            # 更合理的正则化 - 使用比例缩放截断
            max_s = np.max(s)
            threshold = max_s * 1e-10
            s_inv = np.where(s > threshold, 1.0/np.sqrt(s), 0.0)
            
            # 计算Q^(-1/2)
            Q_inv_sqrt = U @ np.diag(s_inv) @ Vh
            self.Q_inv_sqrt = Q_inv_sqrt
            self.Q_inv = Q_inv_sqrt.T @ Q_inv_sqrt
        
        # 按照论文式(15)进行变换
        # a' = Q^(1/2) a / ||Q^(1/2) a||
        # b' = b + a^T c / ||Q^(1/2) a||
        
        # 计算 Q^(1/2) a
        Q_sqrt_a = self.Q_inv_sqrt @ a
        
        # 计算范数
        norm = np.linalg.norm(Q_sqrt_a)
        
        if norm < 1e-10:
            # 处理范数接近零的情况
            print("警告: 半空间变换中出现数值不稳定性")
            return np.zeros_like(halfspace)
        
        # 按照公式计算变换后的半空间参数
        a_prime = Q_sqrt_a / norm
        b_prime = (b + np.dot(a, self.center)) / norm
        
        # 返回变换后的半空间
        transformed_halfspace = np.zeros_like(halfspace)
        transformed_halfspace[:-1] = a_prime
        transformed_halfspace[-1] = b_prime
        
        return transformed_halfspace
    
    def inverse_transform_halfspace(self, halfspace):
        """
        将半空间从椭球体标准坐标系变换回原始坐标系
        变换公式需要与transform_halfspace方法保持一致
        """
        a_std = halfspace[:-1]  # 标准空间中的法向量
        b_std = halfspace[-1]   # 标准空间中的偏移量
        
        # 确保Q矩阵已经分解
        if not hasattr(self, 'Q_sqrt'):
            # 使用SVD计算Q^(1/2)
            U, s, Vh = np.linalg.svd(self.Q, full_matrices=False)
            
            # 稳定的平方根计算
            max_s = np.max(s)
            threshold = max_s * 1e-10
            s_sqrt = np.sqrt(np.where(s > threshold, s, threshold))
            
            # 计算Q^(1/2)
            self.Q_sqrt = U @ np.diag(s_sqrt) @ Vh
        
        # 根据论文式(15)的逆变换
        # 从 a'^T y + b' ≤ 0 逆变换为 a^T x + b ≤ 0
        
        # 计算 a = Q^(-1/2) a'
        a_original = self.Q_sqrt @ a_std
        
        # 计算 b = b' * ||Q^(1/2) a|| - a^T c
        norm = np.linalg.norm(a_original)
        
        if norm < 1e-10:
            # 处理数值不稳定情况
            print("警告: 半空间逆变换中出现数值不稳定性")
            return np.zeros_like(halfspace)
        
        # 计算原始空间的偏移量，保持一致性
        b_original = b_std * norm - np.dot(a_original, self.center)
        
        # 返回原始空间的半空间表示
        original_halfspace = np.zeros_like(halfspace)
        original_halfspace[:-1] = a_original
        original_halfspace[-1] = b_original
        
        return original_halfspace
    
    def to_mesh(self):
        """将椭球体转换为Open3D网格用于可视化"""
        try:
            # 创建单位球
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            
            # 创建变换矩阵
            # 从椭球体的二次型到仿射变换
            Q_sqrt = sqrtm(self.Q)
            
            # 应用变换到每个顶点
            vertices = np.asarray(sphere.vertices)
            transformed_vertices = np.array([Q_sqrt @ v + self.center for v in vertices])
            sphere.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            
            # 更新法线
            sphere.compute_vertex_normals()
            return sphere
        except Exception as e:
            print(f"创建椭球体网格时出错: {e}")
            # 返回一个简单的球体作为备用
            return o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)


class ConvexPolytope:
    def __init__(self, halfspaces=None, points=None):
        """
        初始化凸多面体
        halfspaces: 半空间表示 Ax <= b，存储为[A, b]形式
        points: 顶点表示
        """
        self.halfspaces = halfspaces  # 半空间表示
        self.points = points  # 顶点表示
        self.interior_point = None  # 内部点
        
        # 设置维度
        if halfspaces is not None:
            self.dim = halfspaces.shape[1] - 1
        elif points is not None and len(points) > 0:
            self.dim = points.shape[1]
        else:
            self.dim = 3  # 默认为3D

    def to_mesh(self):
        """
        将多胞体转换为可视化的网格
        使用顶点表示创建凸包，如果顶点不可用则使用采样
        """
        try:
            import open3d as o3d
            from scipy.spatial import ConvexHull
            
            vertices = None
            
            # 检查是否有顶点表示
            if self.points is not None and len(self.points) > 3:
                vertices = self.points
            else:
                # 尝试从半空间计算顶点
                vertices = self.compute_vertices_from_halfspaces()
                
                # 如果计算失败，使用采样创建边界点
                if vertices is None or len(vertices) < 4:
                    print("使用采样创建顶点...")
                    vertices = self._sample_boundary_points()
            
            # 确保有足够的顶点
            if vertices is None or len(vertices) < 4:
                # 创建一个默认立方体
                print("使用默认立方体...")
                center = self.get_interior_point()
                if center is None:
                    center = np.zeros(self.dim)
                
                # 创建边长为2的立方体
                mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
                mesh.translate(center - np.array([1.0, 1.0, 1.0]))
                return mesh
            
            # 计算凸包
            try:
                hull = ConvexHull(vertices)
                
                # 创建点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices)
                
                # 从点云创建凸包
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5)
                
                # 如果凸包创建失败，使用DelaunayTriangulation
                if len(np.asarray(mesh.triangles)) < 1:
                    triangles = []
                    for simplex in hull.simplices:
                        triangles.append(simplex)
                    
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(triangles)
                
                mesh.compute_vertex_normals()
                return mesh
                
            except Exception as e:
                print(f"凸包计算错误: {e}")
                # 创建备用形状
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                center = self.get_interior_point()
                if center is not None:
                    mesh.translate(center)
                return mesh
                
        except Exception as e:
            print(f"多胞体网格创建失败: {e}")
            # 返回一个小球作为备用
            try:
                import open3d as o3d
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                return mesh
            except:
                raise ValueError("无法创建默认网格")
                
    def _sample_boundary_points(self, num_samples=1000):
        """
        通过采样边界创建点集
        """
        if self.halfspaces is None or len(self.halfspaces) < self.dim + 1:
            return None
            
        try:
            # 计算边界框
            bound = 10.0
            interior = self.get_interior_point()
            if interior is not None:
                # 使用内部点为中心创建更大的边界框
                bound = max(10.0, np.max(np.abs(interior)) * 2)
                
            # 在边界框中均匀采样
            points = []
            valid_count = 0
            
            # 生成初始随机点
            samples = np.random.uniform(-bound, bound, size=(num_samples*5, self.dim))
            
            for point in samples:
                if valid_count >= num_samples:
                    break
                    
                # 检查点是否在多胞体内
                if self.contains(point):
                    points.append(point)
                    valid_count += 1
            
            if len(points) < 4:
                return None
                
            return np.array(points)
            
        except Exception as e:
            print(f"边界采样错误: {e}")
            return None
    
    def get_halfspaces(self):
        """
        获取半空间表示为A和b，方便MVIE计算
        """
        if self.halfspaces is None or len(self.halfspaces) == 0:
            return None, None
            
        # 转换为A和b格式
        A = self.halfspaces
        b = -A[:, -1]  # 偏移量取负
        A = A[:, :-1]  # 法向量部分
        
        return A, b
        
    def get_interior_point(self):
        """
        获取多胞体内部点，如果没有预计算则尝试计算
        """
        if self.interior_point is not None:
            return self.interior_point
            
        # 尝试计算内部点
        try:
            # 如果有顶点表示，使用几何中心
            if self.points is not None and len(self.points) > 0:
                self.interior_point = np.mean(self.points, axis=0)
                # 验证内部点是否真的在内部
                if self.contains(self.interior_point):
                    return self.interior_point
                    
            # 如果有半空间表示，使用切比雪夫中心
            if self.halfspaces is not None and len(self.halfspaces) > 0:
                interior = self._compute_chebyshev_center()
                if interior is not None and self.contains(interior):
                    self.interior_point = interior
                    return self.interior_point
                
            # 还可以尝试其他方法，比如随机采样
            interior = self._random_sampling()
            if interior is not None:
                self.interior_point = interior
                return self.interior_point
                
        except Exception as e:
            print(f"计算内部点出错: {e}")
            
        return None
        
    def _compute_chebyshev_center(self):
        """
        计算切比雪夫中心 - 多胞体内可容纳最大球的中心
        """
        try:
            import scipy.optimize as opt
            
            # 获取半空间表示
            A = self.halfspaces[:, :-1]  # 法向量
            b = -self.halfspaces[:, -1]  # 偏移量
            
            # 规范化法向量
            norms = np.linalg.norm(A, axis=1)
            valid_indices = norms > 1e-10
            
            A = A[valid_indices]
            b = b[valid_indices]
            norms = norms[valid_indices]
            
            # 规范化法向量，使其长度为1
            A_norm = A / norms[:, np.newaxis]
            b_norm = b / norms
            
            # 问题维度
            n = A.shape[1]
            
            # 目标函数：最大化球半径
            c = np.zeros(n + 1)
            c[-1] = -1  # 最大化半径（最小化-r）
            
            # 约束：||x - p||₂ ≤ r, for all p on boundary
            # 等价于：aᵀx + r ≤ b, for all a, b defining halfspaces
            A_lp = np.hstack([A_norm, np.ones((A_norm.shape[0], 1))])
            
            # 求解优化问题
            bounds = [(None, None)] * n + [(0, None)]  # 半径非负
            res = opt.linprog(
                c,
                A_ub=A_lp,
                b_ub=b_norm,
                bounds=bounds,
                method='highs'
            )
            
            if res.success:
                return res.x[:-1]  # 中心点
            else:
                return None
                
        except Exception as e:
            print(f"计算切比雪夫中心出错: {e}")
            return None
            
    def _random_sampling(self, max_attempts=1000):
        """
        通过随机采样找内部点
        """
        if self.points is None or len(self.points) == 0:
            # 如果没有顶点，尝试从半空间生成边界框
            if self.halfspaces is not None:
                try:
                    # 一个保守的估计边界盒
                    bound = 10.0  # 假设多胞体在 [-10, 10]^n 的范围内
                    dim = self.halfspaces.shape[1] - 1
                    
                    for _ in range(max_attempts):
                        # 生成随机点
                        point = np.random.uniform(-bound, bound, dim)
                        if self.contains(point):
                            return point
                            
                    # 如果随机采样失败，尝试网格采样
                    grid_size = 5
                    grid = np.linspace(-bound, bound, grid_size)
                    for idx in itertools.product(range(grid_size), repeat=dim):
                        point = np.array([grid[i] for i in idx])
                        if self.contains(point):
                            return point
                            
                except Exception as e:
                    print(f"随机采样内部点出错: {e}")
            
            return None
            
        # 如果有顶点，可以使用凸组合
        try:
            # 使用顶点的凸组合
            n_points = len(self.points)
            weights = np.ones(n_points) / n_points  # 均匀权重
            interior = np.zeros(self.points[0].shape)
            
            for i, p in enumerate(self.points):
                interior += weights[i] * p
                
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
        except:
            pass
            
        return None
    
    def contains(self, point):
        """检查点是否在凸多胞体内"""
        if self.halfspaces is not None:
            # 使用半空间表示检查
            for hs in self.halfspaces:
                a = hs[:-1]
                b = -hs[-1]  # 注意b的符号
                if np.dot(a, point) > b:
                    return False
            return True
        else:
            # 尝试使用点表示检查
            try:
                hull = ConvexHull(self.points)
                test_point = np.array([point])
                new_hull = ConvexHull(np.vstack([self.points, test_point]))
                return len(new_hull.vertices) == len(hull.vertices)
            except:
                return False
    
    def compute_vertices_from_halfspaces(self):
        """优化的多胞体顶点计算方法，减少冗余计算"""
        if self.halfspaces is None or len(self.halfspaces) < 4:
            raise ValueError("多胞体需要至少4个有效半空间")
        
        # 首先查找一个内点
        interior_point = self.sample_interior_point()
        if interior_point is None:
            print("警告: 使用默认内点")
            interior_point = np.array([5.0, 5.0, 5.0])
        
        # 预处理：规范化半空间表示
        A = self.halfspaces[:, :-1]  # 法向量
        b = -self.halfspaces[:, -1]  # 偏移量
        
        # 规范化法向量
        norms = np.linalg.norm(A, axis=1)
        valid_indices = norms > 1e-10
        
        if np.sum(valid_indices) < 4:
            print("警告: 有效半空间不足4个")
            return None
        
        A_norm = A[valid_indices] / norms[valid_indices, np.newaxis]
        b_norm = b[valid_indices] / norms[valid_indices]
        
        # 优化策略1: 快速过滤冗余或近似平行的半空间
        # 计算法向量夹角余弦，过滤近似重复的约束
        n = len(A_norm)
        keep_indices = []
        
        for i in range(n):
            # 默认保留第一个约束
            if i == 0:
                keep_indices.append(i)
                continue
                
            # 检查当前约束与之前保留的约束是否近似平行
            is_redundant = False
            for j in keep_indices:
                cos_angle = np.abs(np.dot(A_norm[i], A_norm[j]))
                
                # 如果夹角余弦接近1，检查偏移量差异
                if cos_angle > 0.99:
                    # 检查哪个约束更紧
                    if b_norm[i] <= b_norm[j]:
                        # i约束更紧或相等，可能替换j
                        is_redundant = False
                        # 考虑移除j并保留i (在实践中更复杂，这里简化处理)
                    else:
                        # j约束更紧，i是冗余的
                        is_redundant = True
                    break
            
            if not is_redundant:
                keep_indices.append(i)
        
        # 使用过滤后的半空间
        filtered_A = A_norm[keep_indices]
        filtered_b = b_norm[keep_indices]
        
        if len(filtered_A) < 4:
            print(f"警告: 过滤后只剩 {len(filtered_A)} 个半空间约束")
            # 回退到原始集合
            filtered_A = A_norm
            filtered_b = b_norm
        
        # 使用射线投射法计算顶点
        vertices = self._compute_vertices_by_ray_casting(interior_point)
        
        if vertices is not None and len(vertices) >= 4:
            self.points = vertices
            return vertices
        else:
            print("警告: 顶点计算失败")
            return None
            
    def _compute_vertices_by_ray_casting(self, interior_point):
        """使用射线投射法计算多胞体顶点"""
        # 在各方向上投射射线，找到与多胞体边界的交点
        vertices = []
        
        # 生成均匀分布在单位球面上的方向向量
        num_dirs = 20  # 方向数量
        
        # 黄金螺旋点采样 - 在球面上更均匀地分布点
        indices = np.arange(0, num_dirs, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_dirs)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        directions = np.vstack((x, y, z)).T
        
        for direction in directions:
            # 归一化方向向量
            direction = direction / np.linalg.norm(direction)
            
            # 沿该方向找到边界点
            t_values = []
            
            for i in range(len(self.halfspaces)):
                a = self.halfspaces[i, :-1]  # 法向量
                b = -self.halfspaces[i, -1]  # 偏移量
                
                # 计算射线与平面的交点参数
                # (p + t*d)·a = b, 其中p是内点，d是方向
                denum = np.dot(direction, a)
                
                # 忽略与射线近似平行的平面
                if abs(denum) < 1e-10:
                    continue
                
                # 计算交点参数
                t = (b - np.dot(interior_point, a)) / denum
                
                # 只考虑正方向的交点（从内点向外）
                if t > 0:
                    t_values.append((t, i))
            
            # 如果没有找到交点，跳过
            if not t_values:
                continue
                
            # 找到最近的交点
            t_values.sort()
            t_min, idx = t_values[0]
            
            # 计算顶点坐标
            vertex = interior_point + t_min * direction
            
            # 检查是否已经存在
            is_duplicate = False
            for v in vertices:
                if np.linalg.norm(vertex - v) < 1e-6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                vertices.append(vertex)
        
        return np.array(vertices) if vertices else None
    
    def sample_interior_point(self):
        """采样多胞体内点"""
        if self.halfspaces is None or len(self.halfspaces) == 0:
            return np.array([5.0, 5.0, 5.0])
            
        try:
            # 使用线性规划寻找Chebyshev中心
            from scipy.optimize import linprog
            
            A = self.halfspaces[:, :-1]  # 法向量
            b = -self.halfspaces[:, -1]  # 偏移量
            
            # 规范化法向量
            norms = np.linalg.norm(A, axis=1)
            valid_indices = norms > 1e-10
            
            if np.sum(valid_indices) > 0:
                A_norm = A[valid_indices] / norms[valid_indices, np.newaxis]
                b_norm = b[valid_indices] / norms[valid_indices]
                
                # 构建线性规划模型，最大化到各个面的最小距离
                dim = A.shape[1]
                c = np.zeros(dim + 1)
                c[-1] = -1  # 最大化r
                
                # 约束：a_i^T x + r <= b_i
                A_lp = np.hstack([A_norm, np.ones((len(A_norm), 1))])
                
                # 求解LP
                bounds = [(None, None)] * dim + [(0, None)]  # r >= 0
                try:
                    res = linprog(c, A_ub=A_lp, b_ub=b_norm, bounds=bounds, method='highs')
                    
                    if res.success:
                        center = res.x[:-1]
                        radius = res.x[-1]
                        
                        # 确保计算出的点真的在多胞体内
                        if self.contains(center) and radius > 1e-6:
                            return center
                except:
                    pass
            
            # 如果线性规划失败，尝试简单的启发式方法
            # 估计边界框中心
            min_bounds = np.full(self.halfspaces.shape[1] - 1, -10.0)
            max_bounds = np.full(self.halfspaces.shape[1] - 1, 10.0)
            
            # 尝试从半空间中估计边界
            for i in range(len(A)):
                a = A[i]
                b_val = b[i]
                
                # 如果法向量主要沿某一轴
                main_axis = np.argmax(np.abs(a))
                if abs(a[main_axis]) > 0.8 * np.linalg.norm(a):
                    # 这是一个近似轴向约束
                    bound = b_val / a[main_axis]
                    if a[main_axis] > 0:  # 上界
                        max_bounds[main_axis] = min(max_bounds[main_axis], bound)
                    else:  # 下界
                        min_bounds[main_axis] = max(min_bounds[main_axis], bound)
            
            # 确保边界有效
            for i in range(len(min_bounds)):
                if min_bounds[i] >= max_bounds[i]:
                    min_bounds[i] = -5.0
                    max_bounds[i] = 5.0
            
            # 尝试边界框中心
            center = (min_bounds + max_bounds) / 2
            if self.contains(center):
                return center
            
            # 如果中心点不在内部，使用随机采样
            for _ in range(50):
                point = min_bounds + np.random.random(len(min_bounds)) * (max_bounds - min_bounds)
                if self.contains(point):
                    return point
            
            # 返回默认值
            return np.array([5.0, 5.0, 5.0])
                
        except Exception as e:
            print(f"内点计算错误: {e}")
            return np.array([5.0, 5.0, 5.0])
            
    def contains(self, point):
        """检查点是否在多胞体内部"""
        if self.halfspaces is None:
            return False
            
        for hs in self.halfspaces:
            a = hs[:-1]  # 法向量
            b = -hs[-1]  # 偏移量
            
            # 检查 a·x ≤ b
            if np.dot(a, point) > b + 1e-8:  # 考虑数值误差
                return False
                
        return True


# 引入SOCP优化的MVIE计算
class MVIE_SOCP:
    """
    使用二阶锥规划(SOCP)方法计算最大体积内接椭球
    基于论文: "SOCP-based algorithm for the minimum volume enclosing ellipsoid"
    """
    def __init__(self, dimension=3):
        """初始化MVIE计算器"""
        self.dim = dimension
        self.max_iterations = 100
        self.eps = 1e-8
        
    def compute(self, polytope):
        """使用SOCP模型计算MVIE"""
        # 获取多胞体的半空间表示
        A, b = polytope.get_halfspaces()
        
        # 检查多胞体是否为空
        if A is None or b is None or A.shape[0] < self.dim + 1:
            raise ValueError("多胞体没有有效的半空间表示")
        
        # 获取多胞体中的一个内部点作为初始椭球中心
        center = polytope.get_interior_point()
        if center is None:
            # 如果无法找到内部点，使用多胞体的空间质心或顶点平均
            try:
                if polytope.points is not None and len(polytope.points) > 0:
                    center = np.mean(polytope.points, axis=0)
                else:
                    center = np.zeros(self.dim)
                print("警告: 找不到内部点，使用备用点", center)
            except:
                center = np.zeros(self.dim)
                print("警告: 多胞体处理出错，使用原点作为中心")
        
        # 尝试使用不同的求解方法
        methods = [
            self._solve_affine_scaling,  # 首选: 论文中的Affine Scaling方法
            self._solve_cvxpy,           # 备选: CVXPY通用求解器
            self._solve_khachiyan        # 后备: Khachiyan迭代算法
        ]
        
        for method in methods:
            try:
                print(f"  尝试使用{method.__name__[7:]}方法求解MVIE...")
                E, center_opt = method(A, b, center)
                if E is not None:
                    # 构造椭球体
                    Q = E @ E.T
                    
                    # 验证矩阵是否有效
                    if self._is_valid_matrix(Q):
                        # 检查椭球体体积
                        ellipsoid = Ellipsoid(center_opt, Q)
                        vol = ellipsoid.volume()
                        if vol > 0 and vol < 1e12 and not np.isnan(vol) and not np.isinf(vol):
                            return ellipsoid
                    
                    print("  求解结果无效，尝试下一个方法")
            except Exception as e:
                print(f"  {method.__name__[7:]}方法失败: {e}")
        
        # 所有方法都失败，返回默认椭球
        print("  所有MVIE方法均失败，使用默认椭球")
        return Ellipsoid(center, np.eye(self.dim))
    
    def _solve_affine_scaling(self, A, b, center_init, max_iter=100, tol=1e-6):
        """
        使用Affine Scaling方法求解MVIE，按照论文第V-B节
        论文表示此方法比标准SOCP求解器快几个数量级
        """
        m, n = A.shape  # m个约束，n维问题
        
        # 初始解: 使用单位矩阵作为E
        E = np.eye(n)
        center = center_init.copy()
        
        # 初始化拉格朗日乘子
        lambda_vec = np.ones(m) / m
        
        # 迭代求解
        for iter_idx in range(max_iter):
            # 1. 计算每个约束的违反程度
            AE = np.zeros((m, n))
            for i in range(m):
                AE[i] = A[i] @ E
            
            norms = np.linalg.norm(AE, axis=1)
            margins = b - A @ center
            violations = norms - margins
            
            # 2. 检查收敛
            max_violation = np.max(violations)
            if max_violation < tol:
                break
            
            # 3. 更新拉格朗日乘子 (按照论文算法1)
            # 计算相对违反度
            rel_violations = violations / (norms + 1e-10)
            
            # 更新lambda - 使用指数更新规则
            step_size = 0.5  # 步长参数
            lambda_vec *= np.exp(step_size * rel_violations)
            lambda_vec /= np.sum(lambda_vec)  # 归一化
            
            # 4. 计算新的E矩阵 (按照论文式18)
            M = np.zeros((n, n))
            for i in range(m):
                ai = A[i].reshape(-1, 1)  # 列向量
                M += lambda_vec[i] * (ai @ ai.T) / (norms[i] + 1e-10)
            
            # 进行Cholesky分解
            try:
                L = np.linalg.cholesky(M)
                E = np.linalg.inv(L.T)
            except:
                # 如果Cholesky分解失败，使用特征分解
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.maximum(eigvals, 1e-10)  # 确保所有特征值为正
                E = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # 5. 更新中心点 (使用内点法)
            # 构造中心点更新的线性系统
            A_tilde = A / (norms + 1e-10).reshape(-1, 1)
            
            # 解线性系统A_tilde^T lambda = 0的最小二乘解
            try:
                center_update = np.linalg.lstsq(A_tilde.T, b * lambda_vec, rcond=None)[0]
                # 限制中心点的移动幅度
                center = center * 0.7 + center_update * 0.3
            except:
                # 如果线性系统求解失败，使用小步长更新
                center_step = np.zeros(n)
                for i in range(m):
                    center_step += lambda_vec[i] * A[i] * (margins[i] / (norms[i] + 1e-10))
                center += 0.1 * center_step
        
        # 检查最终解的有效性
        if iter_idx == max_iter - 1:
            print(f"  Affine Scaling方法未收敛，迭代次数: {max_iter}")
        
        return E, center
    
    def _solve_cvxpy(self, A, b, center_init):
        """
        使用CVXPY求解MVIE标准问题
        """
        try:
            import cvxpy as cp
            
            # 定义SOCP问题变量
            E_var = cp.Variable((self.dim, self.dim), symmetric=True)
            center_var = cp.Variable(self.dim)
            
            # 目标函数: 最大化log(det(E))
            objective = cp.Maximize(cp.log_det(E_var))
            
            # 约束条件
            constraints = []
            
            for i in range(A.shape[0]):
                a_i = A[i]  # 半空间法向量
                b_i = b[i]  # 半空间偏移量
                
                # 添加SOCP约束: ||E^T a_i||_2 + a_i^T d <= b_i
                constraints.append(cp.norm(E_var @ a_i) + a_i @ center_var <= b_i)
            
            # 添加E必须是正定矩阵的约束
            constraints.append(E_var >> 0)
            
            # 求解问题
            prob = cp.Problem(objective, constraints)
            
            try:
                # 首先尝试使用SCS求解器 (通常更快)
                prob.solve(solver=cp.SCS)
            except:
                # 如果SCS失败，尝试使用ECOS
                prob.solve(solver=cp.ECOS)
                
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"  CVXPY求解状态: {prob.status}")
                return None, None
            
            # 获取最优解
            E_opt = E_var.value
            center_opt = center_var.value
            
            # 检查数值稳定性
            if not self._is_valid_matrix(E_opt @ E_opt.T):
                return None, None
                
            return E_opt, center_opt
            
        except Exception as e:
            print(f"  CVXPY求解错误: {e}")
            return None, None
    
    def _solve_khachiyan(self, A, b, center_init, tol=1e-6):
        """
        使用Khachiyan算法求解MVIE
        这是一个迭代算法，适用于顶点表示的多胞体
        """
        # 采样多胞体边界点
        try:
            # 从多胞体边界采样点
            boundary_points = []
            
            # 计算边界框
            if center_init is not None:
                center = center_init
            else:
                center = np.zeros(self.dim)
                
            # 在各个方向上采样
            for _ in range(max(50, 5 * self.dim)):
                # 随机方向
                direction = np.random.randn(self.dim)
                direction = direction / np.linalg.norm(direction)
                
                # 二分搜索边界点
                low = 0.0
                high = 100.0  # 大的初始值
                
                for _ in range(20):  # 二分搜索次数
                    mid = (low + high) / 2
                    point = center + mid * direction
                    
                    # 检查点是否在多胞体内
                    inside = True
                    for i in range(len(A)):
                        if np.dot(A[i], point) > b[i]:
                            inside = False
                            break
                    
                    if inside:
                        low = mid
                    else:
                        high = mid
                
                # 添加边界点
                boundary_point = center + low * 0.99 * direction
                boundary_points.append(boundary_point)
            
            if len(boundary_points) > self.dim:
                center, Q = self._min_vol_ellipsoid(np.array(boundary_points), tol)
                # 从Q推导出E
                _, s, Vh = np.linalg.svd(Q)
                E = Vh.T @ np.diag(np.sqrt(s))
                return E, center
                
        except Exception as e:
            print(f"  Khachiyan求解错误: {e}")
            
        return None, None
    
    def _is_valid_matrix(self, Q):
        """检查矩阵是否是有效的正定矩阵"""
        try:
            # 检查条件数
            eigvals = np.linalg.eigvals(Q)
            if np.any(eigvals <= 0) or np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
                return False
                
            # 检查条件数
            condition = np.max(eigvals) / np.min(eigvals)
            if condition > 1e10:
                print(f"  矩阵条件数过大: {condition:.2e}")
                return False
                
            return True
        except:
            return False
    
    def _min_vol_ellipsoid(self, points, tol=0.001):
        """
        实现Khachiyan算法计算包含所有点的最小体积椭球体
        """
        points = np.asarray(points)
        N, d = points.shape
        
        # 初始椭球为单位球
        Q = np.eye(d)
        center = np.mean(points, axis=0)
        
        # 迭代优化
        iter_count = 0
        max_iter = 100
        
        while iter_count < max_iter:
            # 计算每个点到中心的马氏距离
            diff = points - center
            dist = np.sum(diff @ np.linalg.inv(Q) * diff, axis=1)
            
            # 找到最远点
            j = np.argmax(dist)
            max_dist = dist[j]
            
            # 收敛检查
            if max_dist <= d + tol:
                break
                
            # 更新椭球
            beta = (max_dist - d) / ((max_dist) * (d + 1))
            beta = min(beta, 1.0)  # 限制步长
            
            # 更新中心
            new_center = (1 - beta) * center + beta * points[j]
            
            # 更新形状矩阵
            w = points[j] - center
            Q = (1 - beta) * Q + beta * (d + 1) * np.outer(w, w)
            
            # 更新中心
            center = new_center
            
            iter_count += 1
        
        # 检查矩阵的有效性
        if not self._is_valid_matrix(Q):
            # 无效矩阵，使用备用方法
            Q = np.eye(d)
            
        return center, Q


# 添加2D MVE算法 (针对2D情况的优化)
class MVIE_2D:
    """
    二维平面中最大内接椭圆的线性时间算法
    基于Steiner内切椭圆计算
    """
    def compute(self, polygon):
        """
        计算2D多边形的最大内接椭圆
        参数:
            polygon: 包含points或halfspaces的多边形对象
        返回:
            Ellipsoid对象表示的2D椭圆(z坐标为0)
        """
        # 检查维度
        if polygon.dim != 2:
            raise ValueError("MVIE_2D只适用于二维多边形")
        
        try:
            # 确保有点表示
            if polygon.points is None and polygon.halfspaces is not None:
                polygon.compute_vertices_from_halfspaces()
            
            if polygon.points is None or len(polygon.points) < 3:
                raise ValueError("多边形没有足够的顶点")
            
            # 计算凸包以确保点是按顺序排列的
            from scipy.spatial import ConvexHull
            hull = ConvexHull(polygon.points)
            vertices = polygon.points[hull.vertices]
            
            # 计算三角剖分的加权中心
            triangles = []
            for i in range(1, len(vertices) - 1):
                triangles.append([vertices[0], vertices[i], vertices[i+1]])
            
            # 计算各三角形的面积和中心
            centers = []
            areas = []
            
            for triangle in triangles:
                # 三角形顶点
                a, b, c = triangle
                
                # 计算面积 (叉积的一半)
                area = 0.5 * abs(np.cross(b - a, c - a))
                
                # 计算重心
                center = (a + b + c) / 3
                
                areas.append(area)
                centers.append(center)
            
            # 使用面积加权的方式计算内切椭圆中心
            total_area = sum(areas)
            if total_area < 1e-10:
                center = np.mean(vertices, axis=0)
            else:
                center = sum(c * a for c, a in zip(centers, areas)) / total_area
            
            # 计算到各边的最小距离
            min_dist = float('inf')
            for i in range(len(vertices)):
                j = (i + 1) % len(vertices)
                p1, p2 = vertices[i], vertices[j]
                
                # 计算点到线段的距离
                v = p2 - p1
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-10:
                    continue
                
                v_unit = v / v_norm
                # 垂直向量
                perp = np.array([-v_unit[1], v_unit[0]])
                
                # 计算到当前边的距离
                dist = abs(np.dot(center - p1, perp))
                min_dist = min(min_dist, dist)
            
            # 构建椭圆 (近似为圆，半径为最小距离)
            Q = np.eye(2) * (min_dist ** 2)
            
            # 转换为三维空间中的椭球体 (z坐标为0)
            center_3d = np.array([center[0], center[1], 0.0])
            Q_3d = np.eye(3)
            Q_3d[:2, :2] = Q
            
            return Ellipsoid(center_3d, Q_3d)
            
        except Exception as e:
            print(f"2D MVIE计算错误: {e}")
            # 返回默认椭圆
            return Ellipsoid(np.array([5.0, 5.0, 0.0]), np.eye(3) * 0.5)


class FIRI:
    def __init__(self, obstacles, dimension=3):
        """初始化FIRI算法"""
        self.obstacles = obstacles
        self.dim = dimension
        # 添加MVIE计算器实例
        self.mvie_socp = MVIE_SOCP(dimension)
        self.mvie_2d = MVIE_2D() if dimension == 2 else None
    
    def compute_safe_region(self, seed_points, initial_ellipsoid=None, 
                           max_iterations=3, volume_threshold=0.01):
        """
        计算一组种子点的安全区域
        使用FIRI算法不断扩大椭球体直到收敛
        返回结果为多胞体和内接椭球体
        
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
        
        # 初始化椭球体
        if initial_ellipsoid is None:
            # 使用种子点的中心作为初始椭球体中心
            center = np.mean(seed_points, axis=0)
            # 使用一个小半径的单位矩阵作为初始椭球体
            Q = np.eye(self.dim) * 0.1
            current_ellipsoid = Ellipsoid(center, Q)
        else:
            current_ellipsoid = initial_ellipsoid
        
        # 迭代求解，直到收敛或达到最大迭代次数
        prev_volume = current_ellipsoid.volume()
        current_polytope = None
        
        for iter_num in range(max_iterations):
            print(f"FIRI迭代 {iter_num+1}/{max_iterations}...")
            
            # 1. 执行限制性膨胀，创建包含所有种子点的凸多胞体
            print("  执行限制性膨胀...")
            current_polytope = self.restrictive_inflation(current_ellipsoid, seed_points)
            
            # 2. 计算多胞体内的最大内接椭球 - 使用SOCP方法
            try:
                print("  使用SOCP方法计算MVIE...")
                new_ellipsoid = self.mvie_socp.compute(current_polytope)
                current_volume = new_ellipsoid.volume()
                print(f"  当前椭球体体积: {current_volume:.6f}")
                
                # 检查是否收敛
                volume_increase = (current_volume - prev_volume) / prev_volume
                print(f"  体积增长比例: {volume_increase:.2%}")
                
                if volume_increase < volume_threshold:
                    print(f"  已收敛，停止迭代")
                    current_ellipsoid = new_ellipsoid
                    break
                
                # 更新椭球体和体积
                current_ellipsoid = new_ellipsoid
                prev_volume = current_volume
                
            except Exception as e:
                print(f"  计算MVIE出错: {e}")
                # 如果SOCP失败，使用备用方法
                try:
                    # 尝试使用备用方法计算MVIE
                    print("  尝试使用备用方法计算MVIE...")
                    new_ellipsoid = self.compute_mvie_fallback(current_polytope)
                    current_ellipsoid = new_ellipsoid
                except:
                    print("  备用方法也失败，使用当前椭球体")
                break
        
        # 返回最终的多胞体和椭球体
        return current_polytope, current_ellipsoid
    
    def restrictive_inflation(self, ellipsoid, seed_points):
        """
        实现论文中的限制性膨胀算法，确保种子点完全包含
        输入：
            - ellipsoid: 初始椭球体
            - seed_points: 需要包含的种子点列表
        输出：
            - 一个包含所有种子点的凸多胞体
        """
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
                obs_vertices = np.asarray(obs.vertices)
                obs_center = np.mean(obs_vertices, axis=0)
                dists = np.linalg.norm(obs_vertices - obs_center, axis=1)
                obs_radius = np.max(dists)
            except:
                try:
                    obs_center = obs['center']
                    obs_radius = obs['radius']
                except:
                    print("  警告: 无法处理的障碍物类型，跳过")
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
        
        # 设置最大障碍物数量限制
        max_obstacles = min(20, len(valid_obstacles))
        valid_obstacles = valid_obstacles[:max_obstacles]
        
        # 为每个有效障碍物生成精确的半空间
        for obs_center, obs_radius, _ in valid_obstacles:
            # 变换障碍物中心到标准空间
            obs_center_std = ellipsoid.transform_point(obs_center)
            
            # 根据论文IV-B节，解决最小范数问题
            # 对于每个障碍物，找到离它最近的标准空间点y_min
            # 然后生成半空间 a^T y + b <= 0，其中 a = y_min - obs_center_std, b = -||a||^2/2
            
            # 计算障碍物在标准空间中的半径近似
            try:
                # 使用特征值计算最大伸缩比例
                _, s, _ = np.linalg.svd(ellipsoid.Q_inv)
                scale_factor = np.sqrt(np.max(s))
                obs_radius_std = obs_radius * scale_factor
            except:
                # 简单放大作为备用
                obs_radius_std = obs_radius * 1.5
            
            # 计算安全边距 - 基于障碍物大小和接近度
            base_margin = 0.5 * (1 + min(1.0, obs_radius / 5.0))
            
            # 以下是论文中最小范数问题的求解
            # 对于每个种子点，计算其到障碍物表面的最近点
            for seed_std in transformed_seeds:
                # 计算种子点到障碍物中心的向量
                direction = seed_std - obs_center_std
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm < 1e-10:
                    continue  # 跳过与障碍物中心重合的点
                
                # 规范化方向
                direction = direction / direction_norm
                
                # 障碍物表面点 = 中心 + 半径*方向
                surface_point = obs_center_std + direction * obs_radius_std
                
                # 计算从表面点到种子点的向量 (这是最小范数向量)
                a = seed_std - surface_point
                a_norm = np.linalg.norm(a)
                
                if a_norm < 1e-10:
                    continue  # 跳过在表面上的点
                
                # 创建半空间：确保a指向远离障碍物的方向
                halfspace = np.zeros(self.dim + 1)
                halfspace[:-1] = a / a_norm  # 规范化法向量
                
                # 设置偏移量：确保种子点在半空间内部
                # 计算内积确定相对位置
                offset = -np.dot(a/a_norm, seed_std) + base_margin
                halfspace[-1] = offset
                
                # 验证所有种子点是否在此半空间内
                all_inside = True
                for test_seed in transformed_seeds:
                    if np.dot(halfspace[:-1], test_seed) + halfspace[-1] > 0:
                        all_inside = False
                        break
                
                if all_inside:
                    standard_halfspaces.append(halfspace)
        
        # 如果没有足够的半空间约束，添加边界约束
        if len(standard_halfspaces) < self.dim + 1:
            print(f"  警告: 只有 {len(standard_halfspaces)} 个有效半空间约束，添加边界约束")
            # 添加边界约束，限制在 [-15, 15] 的立方体内
            for i in range(self.dim):
                for sign in [-1, 1]:
                    hs = np.zeros(self.dim + 1)
                    hs[i] = sign
                    hs[-1] = sign * 15.0
                    standard_halfspaces.append(hs)
        
        # 创建标准空间中的多胞体
        standard_polytope = ConvexPolytope(halfspaces=np.array(standard_halfspaces))
        
        # 将标准空间中的多胞体变换回原始空间
        original_halfspaces = []
        for hs in standard_halfspaces:
            # 逆变换半空间
            original_hs = ellipsoid.inverse_transform_halfspace(hs)
            original_halfspaces.append(original_hs)
        
        # 创建原始空间中的多胞体
        original_polytope = ConvexPolytope(halfspaces=np.array(original_halfspaces))
        
        # 计算顶点表示，以备后续使用
        try:
            vertices = original_polytope.compute_vertices_from_halfspaces()
            if vertices is not None and len(vertices) >= 4:
                original_polytope.points = np.array(vertices)
        except Exception as e:
            print(f"  顶点计算错误: {e}")
        
        return original_polytope
    
    # 保留原有函数但重定向到新实现
    def compute_mvie(self, polytope):
        """使用SOCP方法计算MVIE (保留向后兼容性)"""
        try:
            # 如果是2D情况且有专用求解器，使用它
            if self.dim == 2 and self.mvie_2d is not None:
                return self.mvie_2d.compute(polytope)
            else:
                # 使用SOCP求解器
                return self.mvie_socp.compute(polytope)
        except Exception as e:
            print(f"  MVIE计算失败: {e}")
            # 回退到原始计算方法
            print("  使用备用方法计算MVIE...")
            return self.compute_mvie_fallback(polytope)
    
    def compute_mvie_fallback(self, polytope):
        """
        MVIE计算的备用方法，使用简单的球形近似
        """
        print("  使用备用方法计算MVIE")
        
        # 尝试估计多胞体的边界
        min_coords = np.full(self.dim, -float('inf'))
        max_coords = np.full(self.dim, float('inf'))
        
        # 从半空间估计边界
        if polytope.halfspaces is not None and len(polytope.halfspaces) > 0:
            try:
                A = polytope.halfspaces[:, :-1]  # 法向量
                b = -polytope.halfspaces[:, -1]  # 偏移量
                
                for i in range(len(A)):
                    normal = A[i]
                    offset = b[i]
                    
                    # 检查是否是坐标轴方向的约束
                    axis_dir = np.argmax(np.abs(normal))
                    if np.abs(normal[axis_dir]) > 0.8 * np.linalg.norm(normal):
                        # 近似为坐标轴方向的约束
                        value = offset / normal[axis_dir]
                        if normal[axis_dir] > 0:  # 上界
                            max_coords[axis_dir] = min(max_coords[axis_dir], value)
                        else:  # 下界
                            min_coords[axis_dir] = max(min_coords[axis_dir], value)
            except Exception as e:
                print(f"  估计边界盒时出错: {e}")
        
        # 检查是否有有效边界
        valid_min = min_coords > -float('inf')
        valid_max = max_coords < float('inf')
        
        # 确定中心点和半径
        center = np.array([5.0, 5.0, 5.0])  # 默认中心
        radius = 1.0  # 默认半径
        
        if np.any(valid_min & valid_max):
            # 处理有效约束的维度
            valid_dims = valid_min & valid_max
            center_valid = (min_coords[valid_dims] + max_coords[valid_dims]) / 2
            
            # 更新中心点的有效维度
            for i, valid in enumerate(valid_dims):
                if valid:
                    center[i] = center_valid[i - np.sum(valid_dims[:i])]
            
            # 计算可能的半径
            radii = (max_coords[valid_dims] - min_coords[valid_dims]) / 2
            radius = min(np.min(radii), 2.0)  # 限制最大半径
        
        # 检查中心点是否在多胞体内
        if polytope.contains(center):
            # 尝试增大半径
            for factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
                test_radius = radius * factor
                is_valid = True
                
                # 检查椭球体是否完全在多胞体内
                for hs in polytope.halfspaces:
                    normal = hs[:-1]
                    offset = -hs[-1]
                    
                    # 计算中心到平面的距离
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm < 1e-10:
                        continue
                    
                    dist = (np.dot(normal, center) - offset) / normal_norm
                    if dist < -test_radius:  # 完全在半空间外
                        is_valid = False
                        break
                
                if is_valid:
                    radius = test_radius
                    break
        else:
            # 尝试各种备用中心点
            candidates = [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [7.0, 7.0, 7.0],
                [3.0, 7.0, 5.0],
                [7.0, 3.0, 5.0],
                [5.0, 5.0, 3.0],
                [5.0, 5.0, 7.0]
            ]
            
            for candidate in candidates:
                if polytope.contains(np.array(candidate)):
                    center = np.array(candidate)
                    break
        
        # 创建椭球体
        Q = np.eye(self.dim) * (radius ** 2)
        return Ellipsoid(center, Q)


# 将FIRIConfig类的定义移动到这里，在FIRIPlanner之前
class FIRIConfig:
    """FIRI算法配置类，支持自适应参数调整"""
    def __init__(self, space_size=(10, 10, 10)):
        # 基本参数
        self.space_size = space_size
        self.base_safety_margin = 0.5
        self.path_samples = 20
        self.seed_density = 3
        self.use_file_cache = True  # 控制是否使用文件缓存
        
        # 性能监控
        self.timing = {}
        self.iteration_counts = {}
        
        # 自适应参数
        self._adaptive_params = {}
        self.update_adaptive_params()
    
    def update_adaptive_params(self, obstacle_count=None, path_length=None, 
                               complexity_estimate=None):
        """根据场景复杂度更新自适应参数"""
        # 默认值
        if obstacle_count is None:
            obstacle_count = 10
        if path_length is None:
            path_length = 15.0
        if complexity_estimate is None:
            complexity_estimate = 1.0  # 标准复杂度
        
        # 调整安全边距
        self._adaptive_params['safety_margin'] = self.base_safety_margin * (
            1.0 + 0.2 * min(3.0, complexity_estimate))
        
        # 调整种子点密度
        self._adaptive_params['seed_density'] = max(3, min(10, 
            int(self.seed_density * complexity_estimate)))
        
        # 调整路径采样
        self._adaptive_params['path_samples'] = max(20, min(50, 
            int(self.path_samples * complexity_estimate)))
    
    def get_param(self, name):
        """获取参数值，优先使用自适应参数"""
        if name in self._adaptive_params:
            return self._adaptive_params[name]
        return getattr(self, name, None)
    
    def record_timing(self, operation, time_ms):
        """记录计算时间"""
        if operation not in self.timing:
            self.timing[operation] = []
        self.timing[operation].append(time_ms)


# 路径规划部分（修改为使用FIRI生成的安全区域）
class FIRIPlanner:
    def __init__(self, obstacles, space_size=(10, 10, 10)):
        self.obstacles = obstacles
        self.space_size = space_size
        self.safe_regions = []
        self.firi = FIRI(obstacles, dimension=3)
        # 新增配置对象
        self.config = FIRIConfig(space_size)
        
        # 构建KD-Tree加速空间查询
        self._build_obstacle_kdtree()
    
    def _build_obstacle_kdtree(self):
        """构建障碍物顶点的KD-Tree用于快速查询"""
        try:
            # 收集所有障碍物顶点
            all_vertices = []
            self.obstacle_vertex_map = {}  # 映射顶点到障碍物索引
            
            for i, obs in enumerate(self.obstacles):
                try:
                    vertices = np.asarray(obs.vertices)
                    start_idx = len(all_vertices)
                    all_vertices.extend(vertices)
                    end_idx = len(all_vertices)
                    
                    # 记录每个障碍物对应的顶点索引范围
                    self.obstacle_vertex_map[i] = (start_idx, end_idx)
                except:
                    print(f"警告: 障碍物 {i} 无法提取顶点")
            
            # 如果有足够的顶点，构建KD-Tree
            if len(all_vertices) > 0:
                self.vertex_array = np.array(all_vertices)
                self.obstacle_kdtree = KDTree(self.vertex_array)
                print(f"已构建KD-Tree: {len(all_vertices)}个顶点")
                self.use_kdtree = True
            else:
                self.use_kdtree = False
                print("警告: 无法构建KD-Tree，将使用传统碰撞检测")
        except Exception as e:
            print(f"构建KD-Tree时出错: {e}")
            self.use_kdtree = False
    
    def generate_safe_regions(self, start, goal, num_waypoints=4):
        """
        从起点到终点生成安全区域
        参数:
            start: 起点坐标
            goal: 终点坐标
            num_waypoints: 路径点数量
        返回:
            安全区域列表，每个元素是(多胞体, 椭球体)的元组
        """
        print("生成安全区域...")
        # 初始路径：直线采样
        path_points = []
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            point = start * (1 - t) + goal * t
            path_points.append(point)
        
        # 保存初始调整前路径
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/adjusted_path.pkl', 'wb') as f:
            pickle.dump(np.array(path_points), f)
        
        # 采样更多种子点
        self.safe_regions = []
        
        # 为每段路径创建安全区域
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            # 创建种子点 (路径中点和额外的偏移点)
            segment_seeds = [p1, p2, (p1 + p2) / 2]
            
            # 添加额外的种子点 - 沿垂直方向偏移
            direction = p2 - p1
            dir_norm = np.linalg.norm(direction)
            
            if dir_norm > 1e-6:
                # 规范化方向
                direction = direction / dir_norm
                
                # 创建垂直方向向量
                perp = np.cross(direction, np.array([0, 0, 1]))
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(direction, np.array([1, 0, 0]))
                
                if np.linalg.norm(perp) > 1e-6:
                    perp = perp / np.linalg.norm(perp)
                    
                    # 添加偏移种子点
                    mid = (p1 + p2) / 2
                    offset_dist = min(0.5, dir_norm / 4)  # 适当的偏移距离
                    segment_seeds.append(mid + perp * offset_dist)
                    segment_seeds.append(mid - perp * offset_dist)
            
            print(f"为路径段 {i} 计算安全区域 (包含 {len(segment_seeds)} 个种子点)...")
            
            try:
                # 使用FIRI计算安全区域
                polytope, ellipsoid = self.firi.compute_safe_region(
                    seed_points=segment_seeds, 
                    max_iterations=2  # 减少迭代次数以提高速度
                )
                
                # 计算安全区域的质量评估
                volume = ellipsoid.volume()
                print(f"  安全区域 {i} 椭球体体积: {volume:.6f}")
                
                # 保存安全区域
                self.safe_regions.append((polytope, ellipsoid))
                
                # 缓存到文件
                if self.config.use_file_cache:
                    with open(f'temp/safe_region_{i}.pkl', 'wb') as f:
                        pickle.dump((polytope, ellipsoid), f)
                
            except Exception as e:
                print(f"  计算路径段 {i} 的安全区域时出错: {e}")
                # 创建一个默认的安全区域作为替代
                center = (p1 + p2) / 2
                Q = np.eye(3) * 0.5
                default_ellipsoid = Ellipsoid(center, Q)
                
                # 尝试创建一个多胞体
                try:
                    # 创建一个立方体多胞体
                    halfspaces = []
                    for dim in range(3):
                        for sign in [-1, 1]:
                            hs = np.zeros(4)
                            hs[dim] = sign
                            hs[3] = -sign * (center[dim] + sign * 1.0)
                            halfspaces.append(hs)
                    
                    default_polytope = ConvexPolytope(halfspaces=np.array(halfspaces))
                except:
                    default_polytope = None
                
                self.safe_regions.append((default_polytope, default_ellipsoid))
        
        return self.safe_regions
    
    def plan_path(self, start, goal, smoothing=True):
        """
        规划从起点到终点的路径
        使用安全区域和碰撞检测
        参数:
            start: 起点
            goal: 终点
            smoothing: 是否平滑路径
        返回:
            路径点列表
        """
        print("规划路径...")
        
        # 如果还没有安全区域，生成它们
        if not self.safe_regions:
            self.generate_safe_regions(start, goal)
        
        # 初始路径：直接使用安全区域
        initial_path = [start]
        
        # 添加每个安全区域的中心点作为路径点，确保它们在有效范围内
        for region in self.safe_regions:
            polytope, ellipsoid = region
            if ellipsoid is not None:
                # 获取椭球体中心点的副本
                center = np.copy(ellipsoid.center)
                
                # 确保中心点在合理范围内 (0-10)
                for i in range(3):
                    center[i] = max(0.0, min(10.0, center[i]))
                    
                # 检查中心点是否有异常值
                if np.any(np.isnan(center)) or np.any(np.isinf(center)) or np.any(np.abs(center) > 100):
                    # 如果有异常值，使用起点和终点的中点代替
                    center = (start + goal) / 2
                
                initial_path.append(center)
        
        # 添加终点
        if len(initial_path) == 0 or not np.array_equal(initial_path[-1], goal):
            initial_path.append(goal)
        
        # 转换为NumPy数组
        final_path = np.array(initial_path)
        
        # 再次检查所有路径点是否在合理的空间范围内
        for i in range(len(final_path)):
            for j in range(3):
                if final_path[i,j] < 0 or final_path[i,j] > 10 or np.isnan(final_path[i,j]) or np.isinf(final_path[i,j]):
                    # 修正超出范围的坐标
                    final_path[i,j] = max(0.0, min(10.0, 5.0))  # 默认使用中间值5
        
        print(f"初始路径点: {final_path}")
        
        # 碰撞检测和路径调整
        path_safety = self.check_path_safety(final_path)
        
        # 记录路径安全信息到文件
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/path_safety.txt', 'w') as f:
            f.write(f"path_points: {len(final_path)}\n")
            f.write(f"collision_segments: {path_safety['collision_count']}\n")
            f.write(f"collision_indices: {path_safety['collision_indices']}\n")
            f.write(f"path_safety: {'Safe' if path_safety['is_safe'] else 'Unsafe'}\n")
            f.write(f"max_angle: {path_safety['max_angle']:.2f}° avg_angle: {path_safety['avg_angle']:.2f}° angles>90°: {path_safety['sharp_turns']}\n")
        
        # 如果路径不安全，尝试重新规划
        if not path_safety['is_safe']:
            print(f"发现碰撞! 尝试重新规划路径...")
            replanning_attempts = 3
            
            for attempt in range(replanning_attempts):
                print(f"重新规划尝试 {attempt+1}/{replanning_attempts}")
                try:
                    # 获取碰撞段索引
                    collision_indices = path_safety['collision_indices']
                    updated_path = list(final_path)
                    
                    for idx in collision_indices:
                        if idx < len(updated_path) - 1:
                            # 查找碰撞段的替代路径
                            p1 = updated_path[idx]
                            p2 = updated_path[idx + 1]
                            mid = (p1 + p2) / 2
                            
                            # 添加偏移点以避开障碍物
                            for offset_dir in [[1,0,0], [0,1,0], [0,0,1], 
                                             [-1,0,0], [0,-1,0], [0,0,-1]]:
                                offset_point = mid + np.array(offset_dir) * 1.0
                                
                                # 确保点在空间范围内
                                for k in range(3):
                                    offset_point[k] = max(0.0, min(10.0, offset_point[k]))
                                
                                # 检查偏移点是否安全
                                if not self.check_point_collision(offset_point):
                                    # 在原始点之间插入偏移点
                                    updated_path.insert(idx + 1, offset_point)
                                    break
                    
                    # 检查新路径
                    updated_path = np.array(updated_path)
                    
                    # 确保路径中没有重复点或非常接近的点
                    filtered_path = [updated_path[0]]
                    for i in range(1, len(updated_path)):
                        if np.linalg.norm(updated_path[i] - filtered_path[-1]) > 0.1:
                            filtered_path.append(updated_path[i])
                    
                    if len(filtered_path) > 1:
                        updated_path = np.array(filtered_path)
                        new_path_safety = self.check_path_safety(updated_path)
                        if new_path_safety['is_safe']:
                            print("找到安全路径!")
                            final_path = updated_path
                            break
                        else:
                            print(f"重新规划后仍有 {new_path_safety['collision_count']} 处碰撞")
                except Exception as e:
                    print(f"重新规划时出错: {e}")
            
            # 如果重新规划仍然不安全，使用直接路径
            if not path_safety['is_safe']:
                print("使用直接路径...")
                # 创建一条简单路径，通过在起点和终点之间直接构建
                direct_path = []
                
                # 添加起点
                direct_path.append(start)
                
                # 添加几个中间点（使用抛物线避开障碍物）
                for t in np.linspace(0.25, 0.75, 3):
                    point = start * (1-t) + goal * t
                    # 向上偏移以避开障碍物
                    point[2] += 2.0 * np.sin(np.pi * t)  # 抛物线路径
                    direct_path.append(point)
                
                # 添加终点
                direct_path.append(goal)
                
                final_path = np.array(direct_path)
        
        # 路径平滑
        if smoothing and len(final_path) > 2:
            try:
                smoothed_path = self.smooth_path(final_path)
                # 再次检查平滑后的路径安全性
                smooth_safety = self.check_path_safety(smoothed_path)
                if smooth_safety['is_safe']:
                    final_path = smoothed_path
                    print("路径平滑成功且安全")
                else:
                    print("平滑后的路径不安全，使用原始路径")
            except Exception as e:
                print(f"路径平滑出错: {e}")
        
        print(f"最终路径点: {final_path}")
        
        # 保存最终路径到文件
        with open('temp/final_path.pkl', 'wb') as f:
            pickle.dump(final_path, f)
            
        return final_path
    
    def check_path_safety(self, path):
        """
        检查路径安全性
        参数:
            path: 路径点数组
        返回:
            包含安全信息的字典
        """
        # 路径平滑度指标
        angles = []
        collision_indices = []
        collision_count = 0
        is_safe = True
        
        # 检查每个路径段的碰撞
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            
            # 检查此段是否有碰撞
            has_collision = self.check_segment_collision(p1, p2)
            
            if has_collision:
                collision_indices.append(i)
                collision_count += 1
                is_safe = False
                
        # 计算路径平滑度
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                # 计算相邻段之间的角度
                v1 = path[i] - path[i-1]
                v2 = path[i+1] - path[i]
                
                # 确保两个向量都不是零向量
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    # 计算夹角（弧度）
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
        
        # 汇总路径安全信息
        result = {
            'is_safe': is_safe,
            'collision_count': collision_count,
            'collision_indices': collision_indices,
            'avg_angle': np.mean(angles) if angles else 0,
            'max_angle': np.max(angles) if angles else 0,
            'sharp_turns': sum(1 for a in angles if a > 90)
        }
        
        return result
    
    def check_segment_collision(self, p1, p2, samples=10):
        """
        检查线段是否与障碍物碰撞
        使用采样点和KD-Tree
        """
        if not self.use_kdtree:
            # 如果没有KD-Tree，使用传统方法
            for t in np.linspace(0, 1, samples):
                point = p1 * (1-t) + p2 * t
                if self.check_point_collision(point):
                    return True
            return False
        
        # 确定采样点数量（基于段长度）
        segment_length = np.linalg.norm(p2 - p1)
        num_samples = max(5, int(segment_length * 5))
        
        # 沿线段采样点
        for t in np.linspace(0, 1, num_samples):
            point = p1 * (1-t) + p2 * t
            
            # 查询KD-Tree获取最近点和距离
            if self.check_point_collision(point):
                return True
                
        return False
    
    def check_point_collision(self, point, safe_distance=0.05):
        """检查点是否与障碍物碰撞"""
        if self.use_kdtree:
            # 使用KD-Tree查询最近点
            distances, indices = self.obstacle_kdtree.query([point], k=1)
            return distances[0] <= safe_distance
        else:
            # 传统碰撞检测
            for obs in self.obstacles:
                try:
                    # 尝试提取中心和半径
                    center = np.mean(np.asarray(obs.vertices), axis=0)
                    verts = np.asarray(obs.vertices)
                    radius = np.max(np.linalg.norm(verts - center, axis=1))
                    
                    # 计算点到中心的距离
                    dist = np.linalg.norm(point - center)
                    if dist <= radius + safe_distance:
                        return True
                except:
                    # 如果异常，尝试使用字典表示
                    try:
                        center = obs['center']
                        radius = obs['radius']
                        dist = np.linalg.norm(point - center)
                        if dist <= radius + safe_distance:
                            return True
                    except:
                        continue
            
            return False
    
    def smooth_path(self, path, window_size=3, iterations=2):
        """
        平滑路径，使用移动平均
        保持起点和终点不变
        """
        if len(path) <= 2:
            return path
            
        smoothed = np.copy(path)
        
        for _ in range(iterations):
            original = np.copy(smoothed)
            
            # 对中间点平滑处理
            for i in range(1, len(path) - 1):
                # 确定窗口范围
                start = max(0, i - window_size // 2)
                end = min(len(path), i + window_size // 2 + 1)
                
                # 计算移动平均
                window_points = original[start:end]
                smoothed[i] = np.mean(window_points, axis=0)
        
        # 保持起点和终点不变
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        
        return smoothed


# 可视化函数
def visualize_firi_results(obstacles, safe_regions, path=None):
    """
    可视化FIRI算法结果
    修复了元组解包问题和空值处理
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)
    
    # 添加障碍物
    for obs in obstacles:
        obs.paint_uniform_color([0.7, 0.1, 0.1])  # 红色
        vis.add_geometry(obs)
    
    # 添加安全区域
    region_colors = [[0.0, 0.5, 0.8], [0.0, 0.8, 0.5]]  # 蓝色和绿色交替
    
    # 检查safe_regions的类型，确保它是可迭代的
    if safe_regions and isinstance(safe_regions, list):
        for i, region in enumerate(safe_regions):
            # 处理不同类型的安全区域数据结构
            try:
                if isinstance(region, tuple) and len(region) == 2:
                    polytope, ellipsoid = region
                else:
                    # 如果region不是元组或元组长度不为2，则跳过
                    print(f"跳过无效的安全区域 {i}: {type(region)}")
                    continue
                
                # 显示椭球体
                if ellipsoid is not None:
                    try:
                        ellipsoid_mesh = ellipsoid.to_mesh()
                        ellipsoid_mesh.paint_uniform_color(region_colors[i % 2])
                        vis.add_geometry(ellipsoid_mesh)
                    except Exception as e:
                        print(f"椭球体可视化错误: {e}")
                
                # 显示多胞体（半透明）
                if polytope is not None:
                    try:
                        polytope_mesh = polytope.to_mesh()
                        polytope_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
                        polytope_mesh.compute_vertex_normals()
                        # 设置透明度
                        polytope_mesh.compute_triangle_normals()
                        vis.add_geometry(polytope_mesh)
                    except Exception as e:
                        print(f"无法创建多胞体网格: {e}")
            except Exception as e:
                print(f"处理安全区域 {i} 时出错: {e}")
    
    # 添加路径
    if path is not None and len(path) > 1:
        lines = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(path)
        lines.points = points
        lines.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path)-1)])
        lines.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        vis.add_geometry(lines)
        
        # 添加路径点
        for point in path:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # 减小球体大小
            sphere.translate(point)
            sphere.paint_uniform_color([1.0, 0.5, 0.0])  # 橙色
            vis.add_geometry(sphere)
        
        # 保存路径点和线
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/path_points.pkl', 'wb') as f:
            pickle.dump(np.asarray(path), f)
    
    # 设置渲染参数
    render_opt = vis.get_render_option()
    render_opt.line_width = 5.0
    render_opt.background_color = np.array([0.9, 0.9, 0.9])
    render_opt.light_on = True
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_lookat([5, 5, 5])
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()


def visualize_path_only(obstacles, path, start, goal):
    """
    只显示路径、障碍物和起终点的简化可视化函数
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)
    
    # 添加障碍物
    for obs in obstacles:
        obs.paint_uniform_color([0.7, 0.1, 0.1])  # 红色
        vis.add_geometry(obs)
    
    # 添加路径线
    if path is not None and len(path) > 1:
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(path)
        lines.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path)-1)])
        lines.paint_uniform_color([0.0, 0.8, 0.0])  # 绿色
        vis.add_geometry(lines)
    
    # 添加起点（蓝色）
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
    vis.add_geometry(start_sphere)
    
    # 添加终点（黄色）
    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    goal_sphere.translate(goal)
    goal_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
    vis.add_geometry(goal_sphere)
    
    # 设置渲染参数
    render_opt = vis.get_render_option()
    render_opt.line_width = 8.0  # 加粗线条
    render_opt.background_color = np.array([0.9, 0.9, 0.9])  # 淡灰色背景
    render_opt.light_on = True
    render_opt.point_size = 10.0
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_lookat([5, 5, 5])
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()


# 主程序
if __name__ == "__main__":
    # 清理上一次运行产生的临时文件
    if os.path.exists('temp'):
        for file in os.listdir('temp'):
            if file.endswith('.pkl') or file.endswith('.txt'):
                try:
                    os.remove(os.path.join('temp', file))
                except Exception as e:
                    print(f"无法删除文件 {file}: {e}")
    else:
        os.makedirs('temp')
    
    print("清理上一次运行产生的临时文件...")
    
    # 初始化空间和障碍物
    space_size = (10, 10, 10)
    generator = ObstacleGenerator(space_size=space_size)
    
    # 设置起点和终点
    start = np.array([1.0, 1.0, 1.0])
    goal = np.array([9.0, 9.0, 9.0])
    
    # 使用战略性障碍物生成
    obstacles, inflated_obs = generator.generate_strategic_obstacles(
        num_obstacles=20, start=start, goal=goal)
    
    # 使用FIRI规划路径
    planner = FIRIPlanner(inflated_obs, space_size)
    safe_regions = planner.generate_safe_regions(start, goal, num_waypoints=4)
    path = planner.plan_path(start, goal)
    
    # 可视化详细结果
    print("可视化FIRI详细结果...")
    visualize_firi_results(obstacles, safe_regions, path)
    
    # 可视化简化结果（只显示路径、障碍物和起终点）
    print("可视化简化结果（只显示路径和障碍物）...")
    visualize_path_only(obstacles, path, start, goal)
