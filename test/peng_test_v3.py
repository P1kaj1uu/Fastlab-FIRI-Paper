import open3d as o3d
import numpy as np
import math
import os
import pickle
from scipy.optimize import linprog, minimize
from scipy.linalg import null_space, sqrtm, inv
from scipy.spatial import ConvexHull, KDTree
from scipy.interpolate import splprep, splev
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import time
import itertools
import copy
import argparse
import scipy.interpolate

# 提前定义ObstacleSet和Obstacle类，避免未定义错误
class Obstacle:
    """
    Enhanced obstacle representation supporting different shapes
    """
    def __init__(self, center, radius, obstacle_type='sphere', dimensions=None):
        self.center = np.array(center)
        self.radius = radius
        self.obstacle_type = obstacle_type  # 'sphere', 'cylinder', or 'box'
        self.dimensions = dimensions  # Additional dimensions for non-spherical obstacles [width, height, depth] or [radius, height]
        
    def __str__(self):
        if self.obstacle_type == 'sphere':
            return f"Obstacle(type={self.obstacle_type}, center={self.center}, radius={self.radius})"
        elif self.obstacle_type == 'cylinder':
            return f"Obstacle(type={self.obstacle_type}, center={self.center}, radius={self.dimensions[0]}, height={self.dimensions[1]})"
        else:  # box
            return f"Obstacle(type={self.obstacle_type}, center={self.center}, dimensions={self.dimensions})"
        
    def is_point_in_collision(self, point, safety_margin=0):
        """Check if a point collides with this obstacle"""
        point = np.array(point)
        if self.obstacle_type == 'sphere':
            distance = np.linalg.norm(point - self.center)
            return distance <= self.radius + safety_margin
        elif self.obstacle_type == 'cylinder':
            # Decompose into xy-distance and z-distance
            xy_center = self.center[:2]
            xy_point = point[:2]
            xy_distance = np.linalg.norm(xy_point - xy_center)
            z_distance = abs(point[2] - self.center[2])
            
            cylinder_radius = self.dimensions[0]
            cylinder_height = self.dimensions[1]
            
            # Check if point is within cylinder radius and height
            return (xy_distance <= cylinder_radius + safety_margin and 
                    z_distance <= cylinder_height/2 + safety_margin)
        else:  # box
            # Check if point is within box dimensions
            half_width = self.dimensions[0]/2
            half_height = self.dimensions[1]/2
            half_depth = self.dimensions[2]/2
            
            return (abs(point[0] - self.center[0]) <= half_width + safety_margin and
                    abs(point[1] - self.center[1]) <= half_height + safety_margin and
                    abs(point[2] - self.center[2]) <= half_depth + safety_margin)
        
    def is_segment_in_collision(self, p1, p2, safety_margin=0, samples=10):
        """Check if a line segment collides with this obstacle"""
        # Check endpoints
        if self.is_point_in_collision(p1, safety_margin) or self.is_point_in_collision(p2, safety_margin):
            return True
            
        # Sample points along the segment
        for t in np.linspace(0, 1, samples):
            point = p1 * (1-t) + p2 * t
            if self.is_point_in_collision(point, safety_margin):
                return True
                
        return False
        
    def to_mesh(self):
        """Convert obstacle to Open3D mesh for visualization"""
        import open3d as o3d
        
        if self.obstacle_type == 'sphere':
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
            mesh.translate(self.center)
        elif self.obstacle_type == 'cylinder':
            cylinder_radius = self.dimensions[0]
            cylinder_height = self.dimensions[1]
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_height)
            # By default, cylinder's center is at the center of its axis
            mesh.translate(self.center)
        else:  # box
            width, height, depth = self.dimensions
            mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
            # By default, box's origin is at one corner, translate to center
            mesh.translate(self.center - np.array([width/2, height/2, depth/2]))
            
        mesh.compute_vertex_normals()
        return mesh

class ObstacleSet:
    def __init__(self):
        self.obstacle_list = []
        
    def add_obstacle(self, obstacle):
        self.obstacle_list.append(obstacle)
        
    def is_point_in_collision(self, point, safety_margin=0):
        for obstacle in self.obstacle_list:
            if obstacle.is_point_in_collision(point, safety_margin):
                return True
        return False
        
    def is_segment_in_collision(self, p1, p2, safety_margin=0, samples=10):
        for obstacle in self.obstacle_list:
            if obstacle.is_segment_in_collision(p1, p2, safety_margin, samples):
                return True
        return False
        
    def __iter__(self):
        return iter(self.obstacle_list)
        
    def __len__(self):
        return len(self.obstacle_list)

###############################
# 障碍物生成器
###############################
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
            inflated = o3d.geometry.TriangleMesh.create_cylinder(radius=radius + inflation, height=height + inflation)
        elif obstacle_type == 'box':
            size = np.random.uniform(0.5, 2.0, 3)
            mesh = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
            inflated = o3d.geometry.TriangleMesh.create_box(width=size[0] + inflation, height=size[1] + inflation, depth=size[2] + inflation)

        mesh.translate(position)
        mesh.compute_vertex_normals()
        inflated.translate(position)
        inflated.compute_vertex_normals()
        return mesh, inflated

    def generate_strategic_obstacles(self, num_obstacles=30, start=None, goal=None):
        """
        生成战略性障碍物，确保起点和终点之间至少有一个障碍物，
        强制路径需要避障
        """
        obstacles = []
        inflated_obstacles = []
        obstacle_centers = []

        # 保证起点到终点之间有一个障碍物
        if start is not None and goal is not None:
            direction = goal - start
            path_length = np.linalg.norm(direction)
            unit_direction = direction / path_length
            mid_point = start + 0.5 * direction
            offset = np.random.uniform(-0.3, 0.3, 3)
            offset = offset - np.dot(offset, unit_direction) * unit_direction
            if np.linalg.norm(offset) > 0.5:
                offset = offset / np.linalg.norm(offset) * 0.5
            strategic_position = mid_point + offset
            obstacle_type = np.random.choice(['sphere', 'cylinder', 'box'])
            if obstacle_type == 'sphere':
                radius = np.random.uniform(1.5, 2.5)
                obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                inf_obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius + 1.0)
            elif obstacle_type == 'cylinder':
                radius = np.random.uniform(1.2, 2.0)
                height = np.random.uniform(2.0, 3.5)
                obs = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
                inf_obs = o3d.geometry.TriangleMesh.create_cylinder(radius=radius + 1.0, height=height + 1.0)
            else:  # box
                size = np.random.uniform(1.5, 2.5, 3)
                obs = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
                inf_obs = o3d.geometry.TriangleMesh.create_box(width=size[0] + 1.0, height=size[1] + 1.0, depth=size[2] + 1.0)
            obs.translate(strategic_position)
            obs.compute_vertex_normals()
            inf_obs.translate(strategic_position)
            inf_obs.compute_vertex_normals()
            obstacles.append(obs)
            inflated_obstacles.append(inf_obs)
            obstacle_centers.append(strategic_position)
            print(f"策略性障碍物放置在 {strategic_position}，确保路径必须绕行")
        
        safe_radius_start = 2.0
        safe_radius_goal = 2.0
        path_corridor_width = 3.0
        remaining = num_obstacles - len(obstacles)
        for _ in range(remaining):
            position = np.random.rand(3) * self.space_size
            if start is not None and np.linalg.norm(position - start) < safe_radius_start:
                continue
            if goal is not None and np.linalg.norm(position - goal) < safe_radius_goal:
                continue
            if start is not None and goal is not None:
                v = goal - start
                v_length = np.linalg.norm(v)
                v_unit = v / v_length
                t = np.dot(position - start, v_unit)
                t = np.clip(t, 0, v_length)
                proj = start + t * v_unit
                dist_to_line = np.linalg.norm(position - proj)
                outside_corridor = dist_to_line > path_corridor_width
                allow_outside = np.random.random() < 0.3
                if outside_corridor and not allow_outside:
                    continue
            obstacle_type = np.random.choice(['sphere', 'cylinder', 'box'])
            if obstacle_type == 'sphere':
                radius = np.random.uniform(0.5, 2.0)
                obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                inf_obs = o3d.geometry.TriangleMesh.create_sphere(radius=radius + 1.0)
            elif obstacle_type == 'cylinder':
                radius = np.random.uniform(0.3, 1.5)
                height = np.random.uniform(1.0, 3.0)
                obs = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
                inf_obs = o3d.geometry.TriangleMesh.create_cylinder(radius=radius + 1.0, height=height + 1.0)
            else:
                size = np.random.uniform(0.5, 2.0, 3)
                obs = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
                inf_obs = o3d.geometry.TriangleMesh.create_box(width=size[0] + 1.0, height=size[1] + 1.0, depth=size[2] + 1.0)
            obs.translate(position)
            obs.compute_vertex_normals()
            inf_obs.translate(position)
            inf_obs.compute_vertex_normals()
            obstacles.append(obs)
            inflated_obstacles.append(inf_obs)
            obstacle_centers.append(position)
            print(f"障碍物中心: {np.mean(np.asarray(obs.vertices), axis=0)}")
        return obstacles, inflated_obstacles

#####################################
# 内接椭球和半空间变换工具
#####################################
class Ellipsoid:
    def __init__(self, center, Q=None, axes_lengths=None):
        self.center = np.array(center)
        self.dim = len(center)
        
        if Q is not None:
            self.Q = np.array(Q)
            self.volume_calc()
        elif axes_lengths is not None:
            # 如果提供了轴长度，则构造对角矩阵
            D = np.diag(1.0 / (axes_lengths ** 2))
            self.Q = D
            self.volume_calc()
        else:
            # 默认为单位球
            self.Q = np.eye(self.dim)
            self.volume_calc()
            
    def volume_calc(self):
        # 计算椭球体体积
        try:
            det_Q = np.linalg.det(self.Q)
            if det_Q > 0:
                volume = np.pi ** (self.dim / 2) / (np.sqrt(det_Q) * math.gamma(self.dim / 2 + 1))
                self.volume = volume
            else:
                self.volume = 0.0
        except:
            self.volume = 0.0
    
    def contains(self, point):
        # 检查点是否在椭球体内
        p = np.array(point) - self.center
        return np.dot(np.dot(p, self.Q), p) <= 1.0

    def transform_point(self, point):
        """将点从世界坐标变换到椭球标准坐标系 (即映射到单位球)"""
        try:
            u, s, vh = np.linalg.svd(self.Q)
            s_sqrt = np.sqrt(np.where(s > 1e-10, s, 1e-10))
            s_sqrt_inv = 1.0 / s_sqrt
            Q_sqrt = u @ np.diag(s_sqrt) @ vh
            Q_sqrt_inv = vh.T @ np.diag(s_sqrt_inv) @ u.T
            return Q_sqrt_inv @ (np.array(point) - self.center)
        except Exception as e:
            print(f"变换点出错: {e}")
            return np.array(point) - self.center

    def inverse_transform_point(self, point):
        """将点从单位球变换回椭球体 (逆变换)"""
        try:
            u, s, vh = np.linalg.svd(self.Q)
            s_sqrt = np.sqrt(np.where(s > 1e-10, s, 1e-10))
            Q_sqrt = u @ np.diag(s_sqrt) @ vh
            return self.center + Q_sqrt @ np.array(point)
        except Exception as e:
            print(f"逆变换点出错: {e}")
            return self.center + np.array(point)

    def transform_halfspace(self, halfspace):
        """
        将半空间 {x | a^T x + b <= 0} 变换到椭球标准坐标系中，
        按论文(15)的公式：a' = Q^(1/2) a / ||Q^(1/2) a||, b' = (b + a^T c)/||Q^(1/2) a||
        """
        a = halfspace[:-1]
        b = halfspace[-1]
        # 保证提前计算好 Q^(-1/2)
        if not hasattr(self, 'Q_inv_sqrt'):
            U, s, Vh = np.linalg.svd(self.Q, full_matrices=False)
            s_inv_sqrt = np.where(s > 1e-10, 1.0/np.sqrt(s), 0.0)
            self.Q_inv_sqrt = U @ np.diag(s_inv_sqrt) @ Vh
        Q_sqrt_a = self.Q_inv_sqrt @ a
        norm_val = np.linalg.norm(Q_sqrt_a)
        if norm_val < 1e-10:
            print("警告: 半空间变换数值不稳定")
            return np.zeros_like(halfspace)
        a_prime = Q_sqrt_a / norm_val
        b_prime = (b + np.dot(a, self.center)) / norm_val
        transformed_halfspace = np.zeros_like(halfspace)
        transformed_halfspace[:-1] = a_prime
        transformed_halfspace[-1] = b_prime
        return transformed_halfspace

    def inverse_transform_halfspace(self, halfspace):
        """
        将椭球标准坐标系中的半空间逆变换回原始坐标系，
        与 transform_halfspace 保持一致
        """
        a_std = halfspace[:-1]
        b_std = halfspace[-1]
        if not hasattr(self, 'Q_sqrt'):
            U, s, Vh = np.linalg.svd(self.Q, full_matrices=False)
            s_sqrt = np.sqrt(np.where(s > 1e-10, s, 1e-10))
            self.Q_sqrt = U @ np.diag(s_sqrt) @ Vh
        a_original = self.Q_sqrt @ a_std
        norm_val = np.linalg.norm(a_original)
        if norm_val < 1e-10:
            print("警告: 半空间逆变换数值不稳定")
            return np.zeros_like(halfspace)
        b_original = b_std * norm_val - np.dot(a_original, self.center)
        original_halfspace = np.zeros_like(halfspace)
        original_halfspace[:-1] = a_original
        original_halfspace[-1] = b_original
        return original_halfspace

    def to_mesh(self):
        """将椭球体转换为Open3D网格以便可视化"""
        try:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            Q_sqrt = sqrtm(self.Q)
            vertices = np.asarray(sphere.vertices)
            transformed_vertices = np.array([Q_sqrt @ v + self.center for v in vertices])
            sphere.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            sphere.compute_vertex_normals()
            return sphere
        except Exception as e:
            print(f"创建椭球网格出错: {e}")
            return o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)

#####################################
# 凸多胞体及顶点/半空间转换工具
#####################################
class ConvexPolytope:
    def __init__(self, halfspaces=None, points=None):
        """
        初始化凸多胞体
        halfspaces: 半空间约束, 列表形式 [{'normal': normal_vector, 'distance': distance}]
                    normal_vector是指向半空间外部的法向量
                    distance是原点到平面的有向距离
        points: 顶点集合
        """
        self.halfspaces = []
        self.points = None
        
        if halfspaces is not None:
            if isinstance(halfspaces, list):
                self.halfspaces = halfspaces
            elif isinstance(halfspaces, np.ndarray):
                # 如果是numpy数组形式，转换为字典列表
                m, n = halfspaces.shape
                for i in range(m):
                    self.halfspaces.append({
                        'normal': halfspaces[i, :-1],
                        'distance': halfspaces[i, -1]
                    })
        
        if points is not None:
            self.points = np.array(points)
    
    def is_inside(self, point):
        """检查点是否在多面体内部"""
        if not self.halfspaces:
            return True
        
        for hs in self.halfspaces:
            normal = hs['normal']
            distance = hs['distance']
            if np.dot(normal, point) > distance:
                return False
        return True
    
    def contains_origin(self):
        """检查多面体是否包含原点"""
        origin = np.zeros(len(self.halfspaces[0]['normal']))
        return self.is_inside(origin)
    
    def compute_chebyshev_center(self):
        """计算多面体的Chebyshev中心"""
        if not self.halfspaces:
            return np.zeros(3)
        
        # 提取法向量和距离
        A = np.array([hs['normal'] for hs in self.halfspaces])
        b = np.array([hs['distance'] for hs in self.halfspaces])
        
        m, n = A.shape
        
        # 使用线性规划，最大化半径
        c = np.zeros(n+1)
        c[-1] = -1  # 最大化半径
        
        # 构建约束矩阵, Ax - r||A_i|| <= b
        G = np.zeros((m, n+1))
        G[:, :-1] = A
        for i in range(m):
            G[i, -1] = -np.linalg.norm(A[i])
        
        try:
            from scipy.optimize import linprog
            res = linprog(c, A_ub=G, b_ub=b, bounds=(None, None))
            if res.success:
                return res.x[:-1]  # 返回中心点
        except:
            pass
        
        # 如果线性规划失败，返回多面体顶点的平均值
        if self.points is not None and len(self.points) > 0:
            return np.mean(self.points, axis=0)
        
        # 如果都失败，返回默认中心点
        return np.ones(len(self.halfspaces[0]['normal'])) * 5.0

    def to_mesh(self):
        """将多胞体转换为Open3D网格进行可视化"""
        try:
            vertices = None
            if self.points is not None and len(self.points) > 3:
                vertices = self.points
            else:
                vertices = self.compute_vertices_from_halfspaces()
                if vertices is None or len(vertices) < 4:
                    print("使用采样创建顶点...")
                    vertices = self._sample_boundary_points()
            if vertices is None or len(vertices) < 4:
                print("使用默认立方体...")
                center = self.get_interior_point()
                if center is None:
                    center = np.zeros(self.dim)
                mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
                mesh.translate(center - np.array([1.0, 1.0, 1.0]))
                return mesh
            hull = ConvexHull(vertices)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5)
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
            print(f"多胞体网格创建失败: {e}")
            try:
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                return mesh
            except:
                raise ValueError("无法创建默认网格")
                
    def _sample_boundary_points(self, num_samples=1000):
        if self.halfspaces is None or len(self.halfspaces) < self.dim + 1:
            return None
        try:
            bound = 10.0
            interior = self.get_interior_point()
            if interior is not None:
                bound = max(10.0, np.max(np.abs(interior)) * 2)
            points = []
            valid_count = 0
            samples = np.random.uniform(-bound, bound, size=(num_samples*5, self.dim))
            for point in samples:
                if valid_count >= num_samples:
                    break
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
        if self.halfspaces is None or len(self.halfspaces) == 0:
            return None, None
        A = self.halfspaces
        b = -A[:, -1]
        A = A[:, :-1]
        return A, b

    def get_interior_point(self):
        if self.interior_point is not None:
            return self.interior_point
        try:
            if self.points is not None and len(self.points) > 0:
                self.interior_point = np.mean(self.points, axis=0)
                if self.contains(self.interior_point):
                    return self.interior_point
            if self.halfspaces is not None and len(self.halfspaces) > 0:
                interior = self._compute_chebyshev_center()
                if interior is not None and self.contains(interior):
                    self.interior_point = interior
                    return self.interior_point
            interior = self._random_sampling()
            if interior is not None:
                self.interior_point = interior
                return self.interior_point
        except Exception as e:
            print(f"计算内部点出错: {e}")
        return None

    def _compute_chebyshev_center(self):
        try:
            from scipy.optimize import linprog
            A = self.halfspaces[:, :-1]
            b = -self.halfspaces[:, -1]
            norms = np.linalg.norm(A, axis=1)
            valid_indices = norms > 1e-10
            A_norm = A[valid_indices] / norms[valid_indices, np.newaxis]
            b_norm = b[valid_indices] / norms[valid_indices]
            n = A.shape[1]
            c = np.zeros(n + 1)
            c[-1] = -1
            A_lp = np.hstack([A_norm, np.ones((A_norm.shape[0], 1))])
            bounds = [(None, None)] * n + [(0, None)]
            res = linprog(c, A_ub=A_lp, b_ub=b_norm, bounds=bounds, method='highs')
            if res.success:
                return res.x[:-1]
            else:
                return None
        except Exception as e:
            print(f"计算切比雪夫中心出错: {e}")
            return None

    def _random_sampling(self, max_attempts=1000):
        if self.points is None or len(self.points) == 0:
            if self.halfspaces is not None:
                try:
                    bound = 10.0
                    dim = self.halfspaces.shape[1] - 1
                    for _ in range(max_attempts):
                        point = np.random.uniform(-bound, bound, dim)
                        if self.contains(point):
                            return point
                    grid_size = 5
                    grid = np.linspace(-bound, bound, grid_size)
                    for idx in itertools.product(range(grid_size), repeat=dim):
                        point = np.array([grid[i] for i in idx])
                        if self.contains(point):
                            return point
                except Exception as e:
                    print(f"随机采样内部点出错: {e}")
            return None
        try:
            n_points = len(self.points)
            weights = np.ones(n_points) / n_points
            interior = np.zeros(self.points[0].shape)
            for i, p in enumerate(self.points):
                interior += weights[i] * p
            if self.contains(interior):
                return interior
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
        if self.halfspaces is not None:
            for hs in self.halfspaces:
                a = hs[:-1]
                b = -hs[-1]
                if np.dot(a, point) > b + 1e-8:
                    return False
            return True
        else:
            try:
                hull = ConvexHull(self.points)
                test_point = np.array([point])
                new_hull = ConvexHull(np.vstack([self.points, test_point]))
                return len(new_hull.vertices) == len(hull.vertices)
            except:
                return False

    def compute_vertices_from_halfspaces(self):
        if self.halfspaces is None or len(self.halfspaces) < 4:
            raise ValueError("多胞体需要至少4个有效半空间")
        interior_point = self.sample_interior_point()
        if interior_point is None:
            print("警告: 使用默认内点")
            interior_point = np.array([5.0, 5.0, 5.0])
        A = self.halfspaces[:, :-1]
        b = -self.halfspaces[:, -1]
        norms = np.linalg.norm(A, axis=1)
        valid_indices = norms > 1e-10
        if np.sum(valid_indices) < 4:
            print("警告: 有效半空间不足4个")
            return None
        A_norm = A[valid_indices] / norms[valid_indices, np.newaxis]
        b_norm = b[valid_indices] / norms[valid_indices]
        n = len(A_norm)
        keep_indices = []
        for i in range(n):
            if i == 0:
                keep_indices.append(i)
                continue
            is_redundant = False
            for j in keep_indices:
                cos_angle = np.abs(np.dot(A_norm[i], A_norm[j]))
                if cos_angle > 0.99:
                    if b_norm[i] <= b_norm[j]:
                        is_redundant = False
                    else:
                        is_redundant = True
                    break
            if not is_redundant:
                keep_indices.append(i)
        filtered_A = A_norm[keep_indices]
        filtered_b = b_norm[keep_indices]
        if len(filtered_A) < 4:
            print(f"警告: 过滤后只剩 {len(filtered_A)} 个半空间约束")
            filtered_A = A_norm
            filtered_b = b_norm
        vertices = self._compute_vertices_by_ray_casting(interior_point)
        if vertices is not None and len(vertices) >= 4:
            self.points = vertices
            return vertices
        else:
            print("警告: 顶点计算失败")
            return None

    def _compute_vertices_by_ray_casting(self, interior_point):
        vertices = []
        num_dirs = 20
        indices = np.arange(0, num_dirs, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_dirs)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        directions = np.vstack((x, y, z)).T
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            t_values = []
            for i in range(len(self.halfspaces)):
                a = self.halfspaces[i, :-1]
                b = -self.halfspaces[i, -1]
                denum = np.dot(direction, a)
                if abs(denum) < 1e-10:
                    continue
                t = (b - np.dot(interior_point, a)) / denum
                if t > 0:
                    t_values.append((t, i))
            if not t_values:
                continue
            t_values.sort()
            t_min, idx = t_values[0]
            vertex = interior_point + t_min * direction
            duplicate = False
            for v in vertices:
                if np.linalg.norm(vertex - v) < 1e-6:
                    duplicate = True
                    break
            if not duplicate:
                vertices.append(vertex)
        return np.array(vertices) if vertices else None

    def sample_interior_point(self):
        if self.halfspaces is None or len(self.halfspaces) == 0:
            return np.array([5.0, 5.0, 5.0])
        try:
            from scipy.optimize import linprog
            A = self.halfspaces[:, :-1]
            b = -self.halfspaces[:, -1]
            norms = np.linalg.norm(A, axis=1)
            valid_indices = norms > 1e-10
            if np.sum(valid_indices) > 0:
                A_norm = A[valid_indices] / norms[valid_indices, np.newaxis]
                b_norm = b[valid_indices] / norms[valid_indices]
                dim = A.shape[1]
                c = np.zeros(dim + 1)
                c[-1] = -1
                A_lp = np.hstack([A_norm, np.ones((len(A_norm), 1))])
                bounds = [(None, None)] * dim + [(0, None)]
                res = linprog(c, A_ub=A_lp, b_ub=b_norm, bounds=bounds, method='highs')
                if res.success:
                    center = res.x[:-1]
                    radius = res.x[-1]
                    if self.contains(center) and radius > 1e-6:
                        return center
            min_bounds = np.full(self.halfspaces.shape[1] - 1, -10.0)
            max_bounds = np.full(self.halfspaces.shape[1] - 1, 10.0)
            for i in range(len(A)):
                a = A[i]
                b_val = b[i]
                main_axis = np.argmax(np.abs(a))
                if abs(a[main_axis]) > 0.8 * np.linalg.norm(a):
                    bound = b_val / a[main_axis]
                    if a[main_axis] > 0:
                        max_bounds[main_axis] = min(max_bounds[main_axis], bound)
                    else:
                        min_bounds[main_axis] = max(min_bounds[main_axis], bound)
            for i in range(len(min_bounds)):
                if min_bounds[i] >= max_bounds[i]:
                    min_bounds[i] = -5.0
                    max_bounds[i] = 5.0
            center = (min_bounds + max_bounds) / 2
            if self.contains(center):
                return center
            for _ in range(50):
                point = min_bounds + np.random.random(len(min_bounds)) * (max_bounds - min_bounds)
                if self.contains(point):
                    return point
            return np.array([5.0, 5.0, 5.0])
        except Exception as e:
            print(f"内点计算错误: {e}")
            return np.array([5.0, 5.0, 5.0])

##########################################
# MVIE计算器（使用SOCP、CVXPY和Khachiyan等方法）
##########################################
class MVIE_SOCP:
    """
    使用SOCP方法计算最大体积内接椭球
    采用多方法备用策略，提高鲁棒性
    """
    def __init__(self, dimension=3):
        self.dim = dimension
        self.max_iterations = 100
        self.eps = 1e-8

    def compute(self, polytope):
        A, b = polytope.get_halfspaces()
        if A is None or b is None or A.shape[0] < self.dim + 1:
            raise ValueError("多胞体没有有效的半空间表示")
        center = polytope.get_interior_point()
        if center is None:
            try:
                if polytope.points is not None and len(polytope.points) > 0:
                    center = np.mean(polytope.points, axis=0)
                else:
                    center = np.zeros(self.dim)
                print("警告: 找不到内部点，使用备用点", center)
            except:
                center = np.zeros(self.dim)
                print("警告: 多胞体处理出错，使用原点作为中心")
        methods = [self._solve_affine_scaling, self._solve_cvxpy, self._solve_khachiyan]
        for method in methods:
            try:
                print(f"  尝试使用{method.__name__[7:]}方法求解MVIE...")
                E, center_opt = method(A, b, center)
                if E is not None:
                    Q = E @ E.T
                    if self._is_valid_matrix(Q):
                        ellipsoid = Ellipsoid(center_opt, Q)
                        vol = ellipsoid.volume()
                        if vol > 0 and vol < 1e12 and not np.isnan(vol) and not np.isinf(vol):
                            return ellipsoid
                    print("  求解结果无效，尝试下一个方法")
            except Exception as e:
                print(f"  {method.__name__[7:]}方法失败: {e}")
        print("  所有MVIE方法均失败，使用默认椭球")
        return Ellipsoid(center, np.eye(self.dim))

    def _solve_affine_scaling(self, A, b, center_init, max_iter=100, tol=1e-6):
        m, n = A.shape
        E = np.eye(n)
        center = center_init.copy()
        lambda_vec = np.ones(m) / m
        for iter_idx in range(max_iter):
            AE = np.zeros((m, n))
            for i in range(m):
                AE[i] = A[i] @ E
            norms = np.linalg.norm(AE, axis=1)
            margins = b - A @ center
            violations = norms - margins
            max_violation = np.max(violations)
            if max_violation < tol:
                break
            rel_violations = violations / (norms + 1e-10)
            step_size = 0.5
            lambda_vec *= np.exp(step_size * rel_violations)
            lambda_vec /= np.sum(lambda_vec)
            M = np.zeros((n, n))
            for i in range(m):
                ai = A[i].reshape(-1, 1)
                M += lambda_vec[i] * (ai @ ai.T) / (norms[i] + 1e-10)
            try:
                L = np.linalg.cholesky(M)
                E = np.linalg.inv(L.T)
            except:
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.maximum(eigvals, 1e-10)
                E = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            A_tilde = A / (norms + 1e-10).reshape(-1, 1)
            try:
                center_update = np.linalg.lstsq(A_tilde.T, b * lambda_vec, rcond=None)[0]
                center = center * 0.7 + center_update * 0.3
            except:
                center_step = np.zeros(n)
                for i in range(m):
                    center_step += lambda_vec[i] * A[i] * (margins[i] / (norms[i] + 1e-10))
                center += 0.1 * center_step
        if iter_idx == max_iter - 1:
            print(f"  Affine Scaling方法未收敛，迭代次数: {max_iter}")
        return E, center

    def _solve_cvxpy(self, A, b, center_init):
        try:
            import cvxpy as cp
            E_var = cp.Variable((self.dim, self.dim), symmetric=True)
            center_var = cp.Variable(self.dim)
            objective = cp.Maximize(cp.log_det(E_var))
            constraints = []
            for i in range(A.shape[0]):
                a_i = A[i]
                b_i = b[i]
                constraints.append(cp.norm(E_var @ a_i) + a_i @ center_var <= b_i)
            constraints.append(E_var >> 0)
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.SCS)
            except:
                prob.solve(solver=cp.ECOS)
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"  CVXPY求解状态: {prob.status}")
                return None, None
            E_opt = E_var.value
            center_opt = center_var.value
            if not self._is_valid_matrix(E_opt @ E_opt.T):
                return None, None
            return E_opt, center_opt
        except Exception as e:
            print(f"  CVXPY求解错误: {e}")
            return None, None

    def _solve_khachiyan(self, A, b, center_init, tol=1e-6):
        try:
            boundary_points = []
            center = center_init if center_init is not None else np.zeros(self.dim)
            for _ in range(max(50, 5 * self.dim)):
                direction = np.random.randn(self.dim)
                direction = direction / np.linalg.norm(direction)
                low = 0.0
                high = 100.0
                for _ in range(20):
                    mid = (low + high) / 2
                    point = center + mid * direction
                    inside = True
                    for i in range(len(A)):
                        if np.dot(A[i], point) > b[i]:
                            inside = False
                            break
                    if inside:
                        low = mid
                    else:
                        high = mid
                boundary_point = center + low * 0.99 * direction
                boundary_points.append(boundary_point)
            if len(boundary_points) > self.dim:
                center, Q = self._min_vol_ellipsoid(np.array(boundary_points), tol)
                _, s, Vh = np.linalg.svd(Q)
                E = Vh.T @ np.diag(np.sqrt(s))
                return E, center
        except Exception as e:
            print(f"  Khachiyan求解错误: {e}")
        return None, None

    def _min_vol_ellipsoid(self, points, tol=0.001):
        points = np.asarray(points)
        N, d = points.shape
        Q = np.eye(d)
        center = np.mean(points, axis=0)
        iter_count = 0
        max_iter = 100
        while iter_count < max_iter:
            diff = points - center
            dist = np.sum(diff @ np.linalg.inv(Q) * diff, axis=1)
            j = np.argmax(dist)
            max_dist = dist[j]
            if max_dist <= d + tol:
                break
            beta = (max_dist - d) / (max_dist * (d + 1))
            beta = min(beta, 1.0)
            new_center = (1 - beta) * center + beta * points[j]
            w = points[j] - center
            Q = (1 - beta) * Q + beta * (d + 1) * np.outer(w, w)
            center = new_center
            iter_count += 1
        if not self._is_valid_matrix(Q):
            Q = np.eye(d)
        return center, Q

    def _is_valid_matrix(self, Q):
        try:
            eigvals = np.linalg.eigvals(Q)
            if np.any(eigvals <= 0) or np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
                return False
            condition = np.max(eigvals) / np.min(eigvals)
            if condition > 1e10:
                print(f"  矩阵条件数过大: {condition:.2e}")
                return False
            return True
        except:
            return False

    def compute_mvie(self, polytope):
        """计算多面体内的最大体积椭球体(MVIE)"""
        print("  使用SOCP方法计算MVIE...")
        try:
            # 使用不同的方法尝试求解最大内切椭球体
            return self.compute_mvie_affine_scaling(polytope)
        except Exception as e:
            print(f"  SOCP方法失败: {e}")
            # 如果失败，退回到简单方法
            return self.compute_mvie_simple(polytope)

    def compute_mvie_affine_scaling(self, polytope):
        """使用Affine Scaling方法计算MVIE"""
        print("  尝试使用affine_scaling方法求解MVIE...")
        if len(polytope.halfspaces) < 4:
            raise ValueError("多面体约束太少，需要至少4个半空间约束")
        
        # 提取法向量和距离
        halfspaces = polytope.halfspaces
        A = np.array([hs['normal'] for hs in halfspaces])
        b = np.array([hs['distance'] for hs in halfspaces])
        
        # 初始化中心点和矩阵
        n = A.shape[1]  # 维度
        m = A.shape[0]  # 约束数量
        
        # 初始椭球体为单位球在多面体内部的某个点
        center = np.ones(n) * 5.0  # 初始中心点，通常在(0,10)范围内
        E = np.eye(n)  # 初始协方差矩阵
        
        # Affine Scaling优化算法参数
        max_iter = 100
        tol = 1e-6
        
        # 迭代求解
        for it in range(max_iter):
            # 计算当前点到各个约束的距离
            dist = b - np.dot(A, center)
            
            # 检查当前点是否在内部
            if np.any(dist <= 0):
                # 如果当前点在多面体外部，则移动到内部
                center = np.ones(n) * 5.0
                E = np.eye(n) * 0.1
                continue
            
            # 计算距离归一化矩阵
            D = np.diag(1.0 / dist)
            
            # 计算优化矩阵
            M = np.dot(np.dot(A.T, D), A)
            
            # 对矩阵进行特征值分解
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                # 使用最小特征值对应的特征向量作为更新方向
                idx = np.argmin(eigvals)
                v = eigvecs[:, idx]
                
                # 计算最大步长，避免越过边界
                Av = np.dot(A, v)
                pos_idx = Av > 0
                if np.any(pos_idx):
                    alpha = np.min(dist[pos_idx] / Av[pos_idx]) * 0.95
                else:
                    alpha = 1.0
                
                # 更新中心点
                center_new = center + alpha * v
                
                # 更新椭球体模型
                E_new = np.linalg.inv(M)
                
                # 检查收敛性
                if np.linalg.norm(center - center_new) < tol:
                    break
                
                center = center_new
                E = E_new
            except Exception as e:
                print(f"  矩阵特征值分解失败: {e}")
                break
        
        print(f"  Affine Scaling方法{'收敛' if it < max_iter-1 else '未收敛'}，迭代次数: {it+1}")
        
        # 构造椭球体
        try:
            # 检查矩阵是否正定
            eigvals = np.linalg.eigvalsh(E)
            if np.any(eigvals <= 0):
                # 如果不是正定矩阵，使用对角矩阵代替
                E = np.diag(np.abs(np.diag(E)))
            
            # 创建椭球体
            ellipsoid = Ellipsoid(center=center, Q=E)
            return ellipsoid
        except Exception as e:
            print(f"  构造椭球体失败: {e}")
            # 返回一个默认的小椭球体
            return Ellipsoid(center=center, axes_lengths=np.ones(n) * 0.1)

    def compute_mvie_simple(self, polytope):
        """使用简单方法计算MVIE（备用方法）"""
        print("  使用简单方法计算MVIE...")
        
        # 提取多面体的法向量和距离
        halfspaces = polytope.halfspaces
        
        if len(halfspaces) < 1:
            # 如果没有约束，使用默认椭球体
            center = np.array([5.0, 5.0, 5.0])
            return Ellipsoid(center=center, axes_lengths=np.array([1.0, 1.0, 1.0]))
        
        # 估计中心点
        A = np.array([hs['normal'] for hs in halfspaces])
        b = np.array([hs['distance'] for hs in halfspaces])
        
        # 尝试计算重心作为中心点
        try:
            n = A.shape[1]  # 维度
            center = np.zeros(n)
            for i in range(len(halfspaces)):
                center += halfspaces[i]['normal'] * halfspaces[i]['distance']
            center = center / len(halfspaces)
        except:
            # 如果失败，使用默认中心点
            center = np.array([5.0, 5.0, 5.0])
        
        # 计算到各个约束的距离
        dists = []
        for i in range(len(halfspaces)):
            normal = halfspaces[i]['normal']
            distance = halfspaces[i]['distance']
            # 点到平面的距离公式
            dist = (distance - np.dot(normal, center)) / np.linalg.norm(normal)
            dists.append(max(0, dist))
        
        # 使用最小距离作为球体半径
        if dists:
            radius = min(dists)
        else:
            radius = 0.1
        
        # 确保中心点在边界内(0,10)
        for i in range(len(center)):
            center[i] = max(0.0 + radius, min(10.0 - radius, center[i]))
        
        # 创建默认椭球体
        return Ellipsoid(center=center, axes_lengths=np.ones(3) * radius)

##########################################
# 针对二维情况的MVIE（Steiner内切椭圆）
##########################################
class MVIE_2D:
    def compute(self, polygon):
        if polygon.dim != 2:
            raise ValueError("MVIE_2D只适用于二维多边形")
        try:
            if polygon.points is None and polygon.halfspaces is not None:
                polygon.compute_vertices_from_halfspaces()
            if polygon.points is None or len(polygon.points) < 3:
                raise ValueError("多边形没有足够的顶点")
            from scipy.spatial import ConvexHull
            hull = ConvexHull(polygon.points)
            vertices = polygon.points[hull.vertices]
            triangles = []
            for i in range(1, len(vertices) - 1):
                triangles.append([vertices[0], vertices[i], vertices[i+1]])
            centers = []
            areas = []
            for triangle in triangles:
                a, b, c = triangle
                area = 0.5 * abs(np.cross(b - a, c - a))
                center = (a + b + c) / 3
                areas.append(area)
                centers.append(center)
            total_area = sum(areas)
            if total_area < 1e-10:
                center = np.mean(vertices, axis=0)
            else:
                center = sum(c * a for c, a in zip(centers, areas)) / total_area
            min_dist = float('inf')
            for i in range(len(vertices)):
                j = (i + 1) % len(vertices)
                p1, p2 = vertices[i], vertices[j]
                v = p2 - p1
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-10:
                    continue
                v_unit = v / v_norm
                perp = np.array([-v_unit[1], v_unit[0]])
                dist = abs(np.dot(center - p1, perp))
                min_dist = min(min_dist, dist)
            Q = np.eye(2) * (min_dist ** 2)
            center_3d = np.array([center[0], center[1], 0.0])
            Q_3d = np.eye(3)
            Q_3d[:2, :2] = Q
            return Ellipsoid(center_3d, Q_3d)
        except Exception as e:
            print(f"2D MVIE计算错误: {e}")
            return Ellipsoid(np.array([5.0, 5.0, 0.0]), np.eye(3) * 0.5)

##########################################
# FIRI算法核心部分：安全区域生成与椭球迭代
##########################################
class FIRI:
    def __init__(self, obstacles, safety_margin=0.5):
        self.obstacles = obstacles
        self.safety_margin = safety_margin
        self.dim = 3
        self.safe_regions = []
        
        # 用于计算的中心点
        self.center_point = np.array([5.0, 5.0, 5.0])  # 默认中心点
        
        # 初始化sorted_obstacles，根据离中心点的距离排序
        self.sorted_obstacles = self.sort_obstacles_by_distance()
        
        # 尝试构建KD树（如果可能）
        self.use_kdtree = False
        try:
            points = []
            for obs in obstacles:
                if hasattr(obs, 'vertices') and obs.vertices is not None and len(obs.vertices) > 0:
                    points.extend(obs.vertices)
            if len(points) > 0:
                self.obstacle_points = np.array(points)
                self.obstacle_kdtree = KDTree(self.obstacle_points)
                self.use_kdtree = True
            else:
                print("警告: 无法提取足够的顶点构建KD-Tree")
        except Exception as e:
            print(f"警告: 无法构建KD-Tree，将使用传统碰撞检测")
    
    def sort_obstacles_by_distance(self):
        """根据障碍物到中心点的距离排序"""
        distances = []
        
        for obs in self.obstacles:
            # 获取障碍物中心
            if hasattr(obs, 'center'):
                center = obs.center
            elif hasattr(obs, 'vertices') and obs.vertices is not None and len(obs.vertices) > 0:
                center = np.mean(np.asarray(obs.vertices), axis=0)
            else:
                try:
                    center = obs['center']
                except:
                    # 如果无法获取中心，则使用默认位置
                    center = np.array([5.0, 5.0, 5.0])
            
            # 计算到中心点的距离
            distance = np.linalg.norm(center - self.center_point)
            distances.append((obs, distance))
        
        # 按距离排序
        sorted_obstacles = [obs for obs, _ in sorted(distances, key=lambda x: x[1])]
        return sorted_obstacles
    
    def check_point_collision(self, point, safe_distance=0.0):
        """检查点是否与障碍物碰撞"""
        for obstacle in self.obstacles:
            # 处理不同类型的障碍物
            if hasattr(obstacle, 'obstacle_type'):
                if obstacle.obstacle_type == 'sphere':
                    # 球体碰撞检测
                    distance = np.linalg.norm(point - obstacle.center)
                    if distance <= obstacle.radius + safe_distance:
                        return True
                elif obstacle.obstacle_type == 'cylinder':
                    # 圆柱体碰撞检测
                    # 简化为2D距离检测 + 高度检测
                    point_2d = point[:2]
                    center_2d = obstacle.center[:2]
                    distance_2d = np.linalg.norm(point_2d - center_2d)
                    
                    height = obstacle.dimensions[1]  # 高度
                    half_height = height / 2
                    
                    # 检查高度范围
                    if (point[2] >= obstacle.center[2] - half_height - safe_distance and 
                        point[2] <= obstacle.center[2] + half_height + safe_distance and
                        distance_2d <= obstacle.radius + safe_distance):
                        return True
                elif obstacle.obstacle_type == 'box':
                    # 立方体碰撞检测
                    dimensions = obstacle.dimensions
                    half_sizes = dimensions / 2
                    
                    # 检查点是否在扩展了安全距离的边界框内
                    if (point[0] >= obstacle.center[0] - half_sizes[0] - safe_distance and
                        point[0] <= obstacle.center[0] + half_sizes[0] + safe_distance and
                        point[1] >= obstacle.center[1] - half_sizes[1] - safe_distance and
                        point[1] <= obstacle.center[1] + half_sizes[1] + safe_distance and
                        point[2] >= obstacle.center[2] - half_sizes[2] - safe_distance and
                        point[2] <= obstacle.center[2] + half_sizes[2] + safe_distance):
                        return True
                else:
                    # 未知类型，使用球形近似
                    distance = np.linalg.norm(point - obstacle.center)
                    if distance <= obstacle.radius + safe_distance:
                        return True
            else:
                # 默认为球形
                distance = np.linalg.norm(point - obstacle.center)
                if distance <= obstacle.radius + safe_distance:
                    return True
        return False
    
    def check_segment_collision(self, p1, p2, samples=10):
        """检查线段是否与障碍物碰撞"""
        # 生成线段上的采样点
        t_values = np.linspace(0, 1, samples)
        for t in t_values:
            point = p1 * (1-t) + p2 * t
            if self.check_point_collision(point, safe_distance=0.2):
                return True
            
        # 对于圆柱形和长方体障碍物，检查线段与障碍物表面的交点
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'obstacle_type'):
                # 获取线段向量
                segment_direction = p2 - p1
                segment_length = np.linalg.norm(segment_direction)
                if segment_length < 1e-6:  # 线段长度几乎为0
                    continue
                    
                normalized_direction = segment_direction / segment_length
                
                if obstacle.obstacle_type == 'cylinder':
                    # 圆柱体碰撞检测（使用简化模型：无限长圆柱体，然后检查高度）
                    # 这里使用线段到圆柱轴线的最短距离
                    cylinder_axis = np.array([0, 0, 1])  # 假设圆柱沿Z轴
                    p1_to_center = p1 - obstacle.center
                    p2_to_center = p2 - obstacle.center
                    
                    # 计算线段到圆柱轴的最短距离（不考虑高度）
                    # 使用叉积计算点到直线的距离
                    closest_point_1 = p1_to_center - np.dot(p1_to_center, cylinder_axis) * cylinder_axis
                    closest_point_2 = p2_to_center - np.dot(p2_to_center, cylinder_axis) * cylinder_axis
                    
                    distance_1 = np.linalg.norm(closest_point_1[:2])
                    distance_2 = np.linalg.norm(closest_point_2[:2])
                    
                    if min(distance_1, distance_2) <= obstacle.radius + 0.2:
                        # 检查高度约束
                        height = obstacle.dimensions[1]
                        half_height = height / 2
                        
                        # 检查线段是否在高度范围内
                        min_z = min(p1[2], p2[2])
                        max_z = max(p1[2], p2[2])
                        
                        if (min_z <= obstacle.center[2] + half_height + 0.2 and 
                            max_z >= obstacle.center[2] - half_height - 0.2):
                            return True
                
                elif obstacle.obstacle_type == 'box':
                    # 立方体碰撞检测
                    # 简化：检查线段是否穿过障碍物扩展的边界框
                    dimensions = obstacle.dimensions
                    half_sizes = dimensions / 2
                    
                    # 定义边界框的六个面
                    box_min = obstacle.center - half_sizes - 0.2
                    box_max = obstacle.center + half_sizes + 0.2
                    
                    # 参数化线段方程: point = p1 + t * (p2 - p1), t ∈ [0, 1]
                    # 计算线段与每个面的交点
                    for i in range(3):  # x, y, z 三个轴
                        for val in [box_min[i], box_max[i]]:
                            if abs(segment_direction[i]) < 1e-6:  # 如果线段平行于这个平面
                                continue
                                
                            # 计算交点参数 t
                            t = (val - p1[i]) / segment_direction[i]
                            
                            # 如果 t 在 [0, 1] 范围内，检查交点是否在平面的边界内
                            if 0 <= t <= 1:
                                intersection = p1 + t * segment_direction
                                
                                # 检查其他两个维度是否在边界内
                                in_bounds = True
                                for j in range(3):
                                    if j != i and (intersection[j] < box_min[j] or intersection[j] > box_max[j]):
                                        in_bounds = False
                                        break
                                        
                                if in_bounds:
                                    return True
        
        return False

    def compute_safe_region(self, seed_points, initial_ellipsoid=None, max_iterations=3, volume_threshold=0.01):
        """计算安全区域"""
        if initial_ellipsoid is None:
            center = np.mean(seed_points, axis=0)
            try:
                initial_ellipsoid = Ellipsoid(center=center, axes_lengths=np.array([1.0, 1.0, 1.0]))
            except Exception as e:
                print(f"初始化椭球体失败: {e}")
                initial_ellipsoid = Ellipsoid(center=center, Q=np.eye(3))
        
        ellipsoid = initial_ellipsoid
        volume_history = [ellipsoid.volume]
        
        for iteration in range(max_iterations):
            print(f"FIRI迭代 {iteration+1}/{max_iterations}...")
            
            try:
                # 使用限制性膨胀为当前椭球体计算多面体约束
                valid_obstacles, polytope = self.restrictive_inflation(seed_points)
                
                # 尝试计算多面体内最大体积椭球体 (MVIE)
                ellipsoid = self.compute_mvie(polytope)
                current_volume = ellipsoid.volume
                
                if iteration > 0:
                    volume_delta = (current_volume - volume_history[-1]) / volume_history[-1] * 100
                    print(f"  当前椭球体体积: {current_volume:.6f}")
                    print(f"  体积增长比例: {volume_delta:.2f}%")
                    
                    # 检查收敛
                    if abs(volume_delta) < volume_threshold:
                        print("  已收敛，停止迭代")
                        break
                else:
                    print(f"  当前椭球体体积: {current_volume:.6f}")
                    print(f"  体积增长比例: {0.00:.2f}%")
                
                volume_history.append(current_volume)
                
            except Exception as e:
                print(f"  计算失败: {e}")
                # 如果失败，返回最后一个有效结果
                if iteration > 0:
                    break
        
        return polytope, ellipsoid

    def restrictive_inflation(self, seed_points):
        # 初始化障碍物半空间约束和有效障碍物列表
        standard_halfspaces = []
        valid_obstacles = []
        
        print("  执行限制性膨胀...")
        # 处理每个障碍物
        for obstacle in self.sorted_obstacles:
            try:
                # 处理有顶点属性的复杂障碍物
                if hasattr(obstacle, 'vertices') and obstacle.vertices is not None and len(obstacle.vertices) > 0:
                    # 处理顶点数据
                    vertices = np.asarray(obstacle.vertices)
                    center = np.mean(vertices, axis=0)
                    
                    # 基于顶点计算多面体约束
                    for vertex in vertices:
                        # 计算法向量 (从中心指向顶点的方向)
                        direction = vertex - center
                        direction_norm = np.linalg.norm(direction)
                        
                        if direction_norm > 1e-6:  # 避免除以零
                            normal = direction / direction_norm
                            distance = np.dot(normal, vertex)
                            
                            # 添加半空间约束
                            halfspace = {'normal': normal, 'distance': distance, 'type': 'polytope'}
                            standard_halfspaces.append(halfspace)
                            
                    valid_obstacles.append(obstacle)
                
                # 处理新的障碍物类型
                elif hasattr(obstacle, 'obstacle_type'):
                    if obstacle.obstacle_type == 'sphere':
                        # 对于球形障碍物，我们使用中心和半径
                        center = obstacle.center
                        radius = obstacle.radius
                        
                        # 检查种子点是否离障碍物太近
                        too_close = False
                        for point in seed_points:
                            if np.linalg.norm(point - center) <= radius + self.safety_margin:
                                too_close = True
                                break
                                
                        if not too_close:
                            # 添加到有效障碍物列表
                            valid_obstacles.append(obstacle)
                            
                            # 对于球形，我们可以创建一个近似多面体 (在这里我们简化为八个方向)
                            directions = [
                                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                [1, 1, 1], [-1, -1, -1]
                            ]
                            
                            for dir in directions:
                                normal = np.array(dir) / np.linalg.norm(dir)
                                point_on_sphere = center + normal * radius
                                distance = np.dot(normal, point_on_sphere)
                                
                                # 添加半空间约束
                                halfspace = {'normal': normal, 'distance': distance, 'type': 'sphere'}
                                standard_halfspaces.append(halfspace)
                    
                    elif obstacle.obstacle_type == 'cylinder':
                        # 对于圆柱体，我们使用中心、半径和高度
                        center = obstacle.center
                        radius = obstacle.radius
                        height = obstacle.dimensions[1]  # 假设第二个维度是高度
                        half_height = height / 2
                        
                        # 检查种子点是否离障碍物太近
                        too_close = False
                        for point in seed_points:
                            # 简化为2D距离 + 高度检查
                            point_2d = point[:2]
                            center_2d = center[:2]
                            dist_2d = np.linalg.norm(point_2d - center_2d)
                            
                            if (dist_2d <= radius + self.safety_margin and 
                                point[2] >= center[2] - half_height - self.safety_margin and
                                point[2] <= center[2] + half_height + self.safety_margin):
                                too_close = True
                                break
                                
                        if not too_close:
                            # 添加到有效障碍物列表
                            valid_obstacles.append(obstacle)
                            
                            # 添加圆柱侧面约束 (在XY平面上的圆)
                            angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
                            for angle in angles:
                                normal_2d = np.array([np.cos(angle), np.sin(angle), 0])
                                point_on_cylinder = center + normal_2d * radius
                                distance = np.dot(normal_2d, point_on_cylinder)
                                
                                halfspace = {'normal': normal_2d, 'distance': distance, 'type': 'cylinder_side'}
                                standard_halfspaces.append(halfspace)
                            
                            # 添加圆柱顶面和底面约束
                            top_normal = np.array([0, 0, 1])
                            bottom_normal = np.array([0, 0, -1])
                            
                            top_point = center + np.array([0, 0, half_height])
                            bottom_point = center + np.array([0, 0, -half_height])
                            
                            top_distance = np.dot(top_normal, top_point)
                            bottom_distance = np.dot(bottom_normal, bottom_point)
                            
                            standard_halfspaces.append({'normal': top_normal, 'distance': top_distance, 'type': 'cylinder_top'})
                            standard_halfspaces.append({'normal': bottom_normal, 'distance': bottom_distance, 'type': 'cylinder_bottom'})
                    
                    elif obstacle.obstacle_type == 'box':
                        # 对于立方体，我们使用中心和尺寸
                        center = obstacle.center
                        dimensions = obstacle.dimensions
                        half_sizes = dimensions / 2
                        
                        # 检查种子点是否离障碍物太近
                        too_close = False
                        for point in seed_points:
                            if (point[0] >= center[0] - half_sizes[0] - self.safety_margin and
                                point[0] <= center[0] + half_sizes[0] + self.safety_margin and
                                point[1] >= center[1] - half_sizes[1] - self.safety_margin and
                                point[1] <= center[1] + half_sizes[1] + self.safety_margin and
                                point[2] >= center[2] - half_sizes[2] - self.safety_margin and
                                point[2] <= center[2] + half_sizes[2] + self.safety_margin):
                                too_close = True
                                break
                                
                        if not too_close:
                            # 添加到有效障碍物列表
                            valid_obstacles.append(obstacle)
                            
                            # 添加6个面的半空间约束
                            normals = [
                                [1, 0, 0], [-1, 0, 0], 
                                [0, 1, 0], [0, -1, 0], 
                                [0, 0, 1], [0, 0, -1]
                            ]
                            
                            for i, normal in enumerate(normals):
                                normal = np.array(normal)
                                axis = i // 2  # 0=X, 1=Y, 2=Z
                                sign = 1 if i % 2 == 0 else -1
                                
                                # 计算面上的点
                                point_on_face = center.copy()
                                point_on_face[axis] += sign * half_sizes[axis]
                                
                                distance = np.dot(normal, point_on_face)
                                
                                halfspace = {'normal': normal, 'distance': distance, 'type': 'box_face'}
                                standard_halfspaces.append(halfspace)
                    
                    else:
                        # 对于未知类型，记录警告，然后尝试使用默认方式处理
                        print(f"  警告: 未知障碍物类型 {obstacle.obstacle_type}，尝试默认处理")
                        center = obstacle.center
                        radius = obstacle.radius
                        valid_obstacles.append(obstacle)
                        
                        # 使用默认的球形处理方式
                        directions = [
                            [1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [-1, 0, 0], [0, -1, 0], [0, 0, -1]
                        ]
                        
                        for dir in directions:
                            normal = np.array(dir) / np.linalg.norm(dir)
                            point_on_sphere = center + normal * radius
                            distance = np.dot(normal, point_on_sphere)
                            halfspace = {'normal': normal, 'distance': distance, 'type': 'unknown'}
                            standard_halfspaces.append(halfspace)
                
                # 处理旧的障碍物格式
                elif hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
                    center = obstacle.center
                    radius = obstacle.radius
                    
                    # 检查种子点是否离障碍物太近
                    too_close = False
                    for point in seed_points:
                        if np.linalg.norm(point - center) <= radius + self.safety_margin:
                            too_close = True
                            break
                        
                    if not too_close:
                        valid_obstacles.append(obstacle)
                        
                        # 使用默认的球形处理方式
                        directions = [
                            [1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [-1, 0, 0], [0, -1, 0], [0, 0, -1]
                        ]
                        
                        for dir in directions:
                            normal = np.array(dir) / np.linalg.norm(dir)
                            point_on_sphere = center + normal * radius
                            distance = np.dot(normal, point_on_sphere)
                            halfspace = {'normal': normal, 'distance': distance, 'type': 'legacy'}
                            standard_halfspaces.append(halfspace)
            
            except Exception as e:
                print(f"  警告: 处理障碍物时出错: {e}")
        
        # 最多考虑5个最近的障碍物
        print(f"  处理了 {len(valid_obstacles)} 个有效障碍物")
        
        # 如果没有有效的半空间约束，添加边界约束
        if len(standard_halfspaces) == 0:
            print("  警告: 只有 0 个有效半空间约束，添加边界约束")
            # 添加场景边界作为约束
            for i in range(3):  # X, Y, Z
                for sign in [1, -1]:
                    normal = np.zeros(3)
                    normal[i] = sign
                    if sign > 0:
                        distance = 10.0  # 边界最大值
                    else:
                        distance = 0.0  # 边界最小值
                    halfspace = {'normal': normal, 'distance': distance, 'type': 'boundary'}
                    standard_halfspaces.append(halfspace)
        
        # 创建凸多胞体
        polytope = ConvexPolytope(standard_halfspaces)
        
        return valid_obstacles, polytope

    def compute_mvie(self, polytope):
        try:
            if self.dim == 2 and self.mvie_2d is not None:
                return self.mvie_2d.compute(polytope)
            else:
                return self.mvie_socp.compute(polytope)
        except Exception as e:
            print(f"  MVIE计算失败: {e}")
            print("  使用备用方法计算MVIE...")
            return self.compute_mvie_fallback(polytope)

    def compute_mvie_fallback(self, polytope):
        print("  使用备用方法计算MVIE")
        min_coords = np.full(self.dim, -float('inf'))
        max_coords = np.full(self.dim, float('inf'))
        if polytope.halfspaces is not None and len(polytope.halfspaces) > 0:
            try:
                A = polytope.halfspaces[:, :-1]
                b = -polytope.halfspaces[:, -1]
                for i in range(len(A)):
                    normal = A[i]
                    offset = b[i]
                    axis_dir = np.argmax(np.abs(normal))
                    if np.abs(normal[axis_dir]) > 0.8 * np.linalg.norm(normal):
                        value = offset / normal[axis_dir]
                        if normal[axis_dir] > 0:
                            max_coords[axis_dir] = min(max_coords[axis_dir], value)
                        else:
                            min_coords[axis_dir] = max(min_coords[axis_dir], value)
            except Exception as e:
                print(f"  估计边界盒时出错: {e}")
        valid_min = min_coords > -float('inf')
        valid_max = max_coords < float('inf')
        center = np.array([5.0, 5.0, 5.0])
        radius = 1.0
        if np.any(valid_min & valid_max):
            valid_dims = valid_min & valid_max
            center_valid = (min_coords[valid_dims] + max_coords[valid_dims]) / 2
            for i, valid in enumerate(valid_dims):
                if valid:
                    center[i] = center_valid[i - np.sum(valid_dims[:i])]
            radii = (max_coords[valid_dims] - min_coords[valid_dims]) / 2
            radius = min(np.min(radii), 2.0)
        if polytope.contains(center):
            for factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
                test_radius = radius * factor
                is_valid = True
                for hs in polytope.halfspaces:
                    normal = hs[:-1]
                    offset = -hs[-1]
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm < 1e-10:
                        continue
                    dist = (np.dot(normal, center) - offset) / normal_norm
                    if dist < -test_radius:
                        is_valid = False
                        break
                if is_valid:
                    radius = test_radius
                    break
        else:
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
        Q = np.eye(self.dim) * (radius ** 2)
        return Ellipsoid(center, Q)

##########################################
# FIRI算法配置和性能分析
##########################################
class FIRIConfig:
    def __init__(self, space_size=(10, 10, 10)):
        self.space_size = space_size
        self.base_safety_margin = 0.5
        self.path_samples = 20
        self.seed_density = 3
        self.use_file_cache = True
        self.timing = {}
        self.iteration_counts = {}
        self._adaptive_params = {}
        self.update_adaptive_params()
    
    def update_adaptive_params(self, obstacle_count=None, path_length=None, complexity_estimate=None):
        if obstacle_count is None:
            obstacle_count = 10
        if path_length is None:
            path_length = 15.0
        if complexity_estimate is None:
            complexity_estimate = 1.0
        self._adaptive_params['safety_margin'] = self.base_safety_margin * (1.0 + 0.2 * min(3.0, complexity_estimate))
        self._adaptive_params['seed_density'] = max(3, min(10, int(self.seed_density * complexity_estimate)))
        self._adaptive_params['path_samples'] = max(20, min(50, int(self.path_samples * complexity_estimate)))
    
    def get_param(self, name):
        if name in self._adaptive_params:
            return self._adaptive_params[name]
        return getattr(self, name, None)
    
    def record_timing(self, operation, time_ms):
        if operation not in self.timing:
            self.timing[operation] = []
        self.timing[operation].append(time_ms)

class PerformanceAnalyzer:
    def __init__(self):
        self.volume_history = []
        self.volume_growth = []
        self.computation_times = []
        self.start_time = None
        
    def start_recording(self):
        self.volume_history = []
        self.volume_growth = [] 
        self.computation_times = []
        self.start_time = time.time()
        
    def record_iteration(self, iter_num, volume, time_cost):
        self.volume_history.append(volume)
        self.computation_times.append(time_cost)
        if len(self.volume_history) > 1:
            prev_vol = self.volume_history[-2]
            self.volume_growth.append((volume - prev_vol)/prev_vol)
            
    def generate_report(self):
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.plot(self.volume_history, 'bo-')
        plt.title('Volume Growth')
        plt.xlabel('Iteration')
        plt.ylabel('Volume')
        plt.subplot(122)
        plt.plot(self.computation_times, 'r^-')
        plt.title('Computation Time per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        plt.savefig('performance_report.png')
        plt.close()

##########################################
# FIRI路径规划器：结合安全区域和碰撞检测进行路径规划
##########################################
class FIRIPlanner:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.firi = FIRI(obstacles, safety_margin=1.5)
        self.safe_regions = []
        self.use_kdtree = self.firi.use_kdtree
        if hasattr(self.firi, 'obstacle_kdtree'):
            self.obstacle_kdtree = self.firi.obstacle_kdtree
    
    def generate_safe_regions(self, start, goal):
        print("生成安全区域...")
        
        # 确保路径规划器考虑足够多的点，以增加避障成功率
        num_segments = 3
        seed_points = []
        
        # 在起点和终点之间生成种子点
        for i in range(num_segments + 1):
            t = i / num_segments
            point = start * (1 - t) + goal * t
            seed_points.append(point)
            
        # 为每个线段计算安全区域
        self.safe_regions = []
        
        for i in range(num_segments):
            segment_seeds = seed_points[i:i+2]
            
            # 添加额外的种子点
            middle_point = (segment_seeds[0] + segment_seeds[1]) / 2
            
            # 在中间点周围添加额外的种子点
            additional_seeds = []
            additional_seeds.append(middle_point)
            
            # 对中间点添加一些随机偏移，提高成功率
            for _ in range(3):
                offset = np.random.uniform(-2.0, 2.0, 3)
                random_point = middle_point + offset
                # 确保点在场景范围内
                for j in range(3):
                    random_point[j] = max(0.0, min(10.0, random_point[j]))
                additional_seeds.append(random_point)
                
            all_seeds = np.vstack([segment_seeds, additional_seeds])
            
            print(f"为路径段 {i} 计算安全区域 (包含 {len(all_seeds)} 个种子点)...")
            try:
                # 初始椭球体
                center = np.mean(all_seeds, axis=0)
                initial_ellipsoid = Ellipsoid(center=center, axes_lengths=np.array([1.0, 1.0, 1.0]))
                
                # 使用FIRI计算安全区域
                polytope, ellipsoid = self.firi.compute_safe_region(
                    all_seeds, initial_ellipsoid, max_iterations=2)
                
                print(f"  安全区域 {i} 椭球体体积: {ellipsoid.volume:.6f}")
                self.safe_regions.append((polytope, ellipsoid))
                
                # 保存安全区域数据
                if not os.path.exists('temp'):
                    os.makedirs('temp')
                with open(f'temp/safe_region_{i}.pkl', 'wb') as f:
                    pickle.dump((polytope, ellipsoid), f)
                    
            except Exception as e:
                print(f"  计算路径段 {i} 的安全区域时出错: {e}")
                self.safe_regions.append((None, None))
    
    def check_point_collision(self, point, safe_distance=0.05):
        return self.firi.check_point_collision(point, safe_distance)
    
    def check_segment_collision(self, p1, p2, samples=10):
        return self.firi.check_segment_collision(p1, p2, samples)
    
    def check_path_safety(self, path):
        angles = []
        collision_indices = []
        collision_count = 0
        is_safe = True
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            if self.check_segment_collision(p1, p2):
                collision_indices.append(i)
                collision_count += 1
                is_safe = False
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                v1 = path[i] - path[i-1]
                v2 = path[i+1] - path[i]
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
        result = {
            'is_safe': is_safe,
            'collision_count': collision_count,
            'collision_indices': collision_indices,
            'avg_angle': np.mean(angles) if angles else 0,
            'max_angle': np.max(angles) if angles else 0,
            'sharp_turns': sum(1 for a in angles if a > 90)
        }
        return result
    
    def plan_path(self, start, goal, smoothing=True):
        print("规划路径...")
        if not self.safe_regions:
            self.generate_safe_regions(start, goal)
        
        # 初始路径
        initial_path = [start]
        for region in self.safe_regions:
            polytope, ellipsoid = region
            if ellipsoid is not None:
                center = np.copy(ellipsoid.center)
                for i in range(3):
                    center[i] = max(0.0, min(10.0, center[i]))
                if np.any(np.isnan(center)) or np.any(np.isinf(center)) or np.any(np.abs(center) > 100):
                    center = (start + goal) / 2
                initial_path.append(center)
        if len(initial_path) == 0 or not np.array_equal(initial_path[-1], goal):
            initial_path.append(goal)
        
        # 改进：如果路径点太少，采用分段策略生成更多点
        if len(initial_path) < 5:
            enhanced_path = [start]
            for i in range(1, len(initial_path)):
                prev = initial_path[i-1]
                curr = initial_path[i]
                # 添加多个中间点以提高路径分辨率
                for t in np.linspace(0.2, 0.8, 3):
                    # 使用偏离直线的中间点
                    intermediate = prev * (1-t) + curr * t
                    # 增加一点随机偏移，以增加避障可能性
                    random_offset = np.random.uniform(-0.5, 0.5, 3)
                    intermediate += random_offset
                    # 确保在边界内
                    for j in range(3):
                        intermediate[j] = max(0.0, min(10.0, intermediate[j]))
                    enhanced_path.append(intermediate)
            if not np.array_equal(enhanced_path[-1], goal):
                enhanced_path.append(goal)
            initial_path = enhanced_path
        
        final_path = np.array(initial_path)
        for i in range(len(final_path)):
            for j in range(3):
                if final_path[i,j] < 0 or final_path[i,j] > 10 or np.isnan(final_path[i,j]) or np.isinf(final_path[i,j]):
                    final_path[i,j] = 5.0
        
        print(f"初始路径点: {final_path}")
        path_safety = self.check_path_safety(final_path)
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/path_safety.txt', 'w') as f:
            f.write(f"path_points: {len(final_path)}\n")
            f.write(f"collision_segments: {path_safety['collision_count']}\n")
            f.write(f"collision_indices: {path_safety['collision_indices']}\n")
            f.write(f"path_safety: {'Safe' if path_safety['is_safe'] else 'Unsafe'}\n")
            f.write(f"max_angle: {path_safety['max_angle']:.2f}° avg_angle: {path_safety['avg_angle']:.2f}° angles>90°: {path_safety['sharp_turns']}\n")
        
        # 增强避障策略
        if not path_safety['is_safe']:
            print(f"发现碰撞! 尝试重新规划路径...")
            
            # 保存最初的路径，重规划失败时恢复使用
            original_path = np.copy(final_path)
            best_collision_count = path_safety['collision_count']
            
            # 增加重规划尝试次数
            replanning_attempts = 5
            
            for attempt in range(replanning_attempts):
                print(f"重新规划尝试 {attempt+1}/{replanning_attempts}")
                try:
                    collision_indices = path_safety['collision_indices']
                    updated_path = list(final_path)
                    
                    # 对每个碰撞段尝试多种避障方向
                    for idx in collision_indices:
                        if idx < len(updated_path) - 1:
                            p1 = updated_path[idx]
                            p2 = updated_path[idx + 1]
                            
                            # 获取该段的中点
                            mid = (p1 + p2) / 2
                            
                            # 尝试不同的偏移方向和大小
                            offset_dirs = [
                                [1,0,0], [0,1,0], [0,0,1],
                                [-1,0,0], [0,-1,0], [0,0,-1],
                                [1,1,0], [1,0,1], [0,1,1],
                                [-1,-1,0], [-1,0,-1], [0,-1,-1],
                                [0.5,0.5,0.5], [-0.5,-0.5,-0.5]
                            ]
                            
                            offset_scales = [1.0, 2.0, 3.0]  # 尝试不同大小的偏移
                            
                            found_safe_point = False
                            for offset_dir in offset_dirs:
                                for scale in offset_scales:
                                    offset_point = mid + np.array(offset_dir) * scale
                                    
                                    # 确保点在场景范围内
                                    for k in range(3):
                                        offset_point[k] = max(0.0, min(10.0, offset_point[k]))
                                    
                                    # 检查新点是否安全
                                    if not self.check_point_collision(offset_point, safe_distance=0.5):
                                        # 检查连接是否安全
                                        if (not self.check_segment_collision(p1, offset_point, samples=20) and 
                                            not self.check_segment_collision(offset_point, p2, samples=20)):
                                            # 插入安全点
                                            updated_path.insert(idx + 1, offset_point)
                                            found_safe_point = True
                                            print(f"  在位置 {idx+1} 插入安全点 {offset_point}")
                                            break
                                
                                if found_safe_point:
                                    break
                            
                            # 如果所有偏移都不安全，尝试随机点
                            if not found_safe_point:
                                for _ in range(10):
                                    # 生成远离直线的随机点
                                    random_offset = np.random.uniform(-3.0, 3.0, 3)
                                    random_point = mid + random_offset
                                    
                                    # 确保点在场景范围内
                                    for k in range(3):
                                        random_point[k] = max(0.0, min(10.0, random_point[k]))
                                    
                                    if not self.check_point_collision(random_point, safe_distance=0.5):
                                        # 检查连接是否安全
                                        if (not self.check_segment_collision(p1, random_point, samples=20) and 
                                            not self.check_segment_collision(random_point, p2, samples=20)):
                                            # 插入安全点
                                            updated_path.insert(idx + 1, random_point)
                                            print(f"  在位置 {idx+1} 插入随机安全点 {random_point}")
                                            break
                    
                    # 转换为numpy数组并过滤掉太近的点
                    updated_path = np.array(updated_path)
                    filtered_path = [updated_path[0]]
                    
                    for i in range(1, len(updated_path)):
                        if np.linalg.norm(updated_path[i] - filtered_path[-1]) > 0.1:
                            filtered_path.append(updated_path[i])
                    
                    if len(filtered_path) > 1:
                        updated_path = np.array(filtered_path)
                        new_path_safety = self.check_path_safety(updated_path)
                        
                        # 如果完全安全，则使用这条路径
                        if new_path_safety['is_safe']:
                            print("找到安全路径!")
                            final_path = updated_path
                            break
                        # 如果比之前更好但仍不完全安全，保存为最佳路径
                        elif new_path_safety['collision_count'] < best_collision_count:
                            best_collision_count = new_path_safety['collision_count']
                            final_path = updated_path
                            print(f"  找到更好的路径，碰撞减少至 {best_collision_count} 处")
                        else:
                            print(f"  重新规划后仍有 {new_path_safety['collision_count']} 处碰撞")
                
                except Exception as e:
                    print(f"重新规划时出错: {e}")
            
            # 如果重规划未能完全解决碰撞，尝试使用更激进的方法
            path_safety = self.check_path_safety(final_path)
            if not path_safety['is_safe']:
                print("标准避障未能完全消除碰撞，尝试直接连接法...")
                
                # 直接连接策略：创建足够多的中间点，并尝试直接连接
                safe_paths = []
                safe_paths.append(start)
                
                # 定义空间网格用于采样
                grid_size = 5
                x_range = np.linspace(0, 10, grid_size)
                y_range = np.linspace(0, 10, grid_size)
                z_range = np.linspace(0, 10, grid_size)
                
                # 添加一些预定义的安全点，提高成功率
                predefined_safe_points = [
                    [2, 2, 3], [2, 8, 3], [8, 2, 3], [8, 8, 3],
                    [2, 5, 7], [5, 2, 7], [5, 8, 7], [8, 5, 7],
                    [5, 5, 5]
                ]
                
                # 检查并收集所有安全点
                all_safe_points = []
                
                # 首先检查预定义点
                for point in predefined_safe_points:
                    if not self.check_point_collision(point, safe_distance=0.5):
                        all_safe_points.append(np.array(point))
                
                # 然后检查网格点
                for x in x_range:
                    for y in y_range:
                        for z in z_range:
                            point = np.array([x, y, z])
                            if not self.check_point_collision(point, safe_distance=0.5):
                                all_safe_points.append(point)
                
                # 排序安全点，优先考虑离起点近的
                all_safe_points.sort(key=lambda p: np.linalg.norm(p - start))
                
                # 尝试使用安全点构建路径
                current = start
                for safe_point in all_safe_points:
                    if np.linalg.norm(current - safe_point) < 0.1:  # 跳过太近的点
                        continue
                    
                    # 检查连接是否安全
                    if not self.check_segment_collision(current, safe_point, samples=20):
                        safe_paths.append(safe_point)
                        current = safe_point
                        
                        # 检查是否可以直接连接到目标
                        if not self.check_segment_collision(current, goal, samples=20):
                            safe_paths.append(goal)
                            break
                
                # 如果未能连接到目标，添加目标
                if not np.array_equal(safe_paths[-1], goal):
                    safe_paths.append(goal)
                
                # 检查新路径安全性
                safe_path_array = np.array(safe_paths)
                safe_path_safety = self.check_path_safety(safe_path_array)
                
                if safe_path_safety['is_safe'] or safe_path_safety['collision_count'] < best_collision_count:
                    print(f"直接连接法成功! 碰撞数: {safe_path_safety['collision_count']}")
                    final_path = safe_path_array
        
        # 最后的平滑过程
        if smoothing and len(final_path) > 2:
            try:
                # 使用窗口平滑
                smoothed_path = smooth_path(final_path, window_size=3, iterations=2)
                smooth_safety = self.check_path_safety(smoothed_path)
                if smooth_safety['is_safe']:
                    final_path = smoothed_path
                    print("路径平滑成功且安全")
                else:
                    print("平滑后的路径不安全，使用原始路径")
            except Exception as e:
                print(f"路径平滑出错: {e}")
        
        print(f"最终路径点: {final_path}")
        with open('temp/final_path.pkl', 'wb') as f:
            pickle.dump(final_path, f)
        return final_path

def smooth_path(path, safe_areas=None, window_size=3, iterations=2):
    """
    Smooth the path using B-spline interpolation while staying within safe areas
    
    Parameters:
        path: Original path points
        safe_areas: List of safe areas corresponding to each path point
        window_size: Window size for smoothing (if using window smoothing)
        iterations: Number of smoothing iterations (if using window smoothing)
    """
    if len(path) < 3:
        return path
    
    # Use simple window-based smoothing
    smoothed = np.copy(path)
    for _ in range(iterations):
        original = np.copy(smoothed)
        for i in range(1, len(path) - 1):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)
            window_points = original[start_idx:end_idx]
            smoothed[i] = np.mean(window_points, axis=0)
    
    # Ensure start and end points remain unchanged
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed

##########################################
# 可视化函数
##########################################
def visualize_firi_results(obstacles, safe_regions, path=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)
    for obs in obstacles:
        obs.paint_uniform_color([0.7, 0.1, 0.1])
        vis.add_geometry(obs)
    region_colors = plt.cm.viridis(np.linspace(0,1,len(safe_regions)))[:,:3]
    if safe_regions and isinstance(safe_regions, list):
        for i, region in enumerate(safe_regions):
            try:
                if isinstance(region, tuple) and len(region) == 2:
                    polytope, ellipsoid = region
                else:
                    print(f"跳过无效的安全区域 {i}: {type(region)}")
                    continue
                if ellipsoid is not None:
                    try:
                        ellipsoid_mesh = ellipsoid.to_mesh()
                        ellipsoid_mesh.paint_uniform_color(region_colors[i % 2])
                        vis.add_geometry(ellipsoid_mesh)
                    except Exception as e:
                        print(f"椭球体可视化错误: {e}")
                if polytope is not None:
                    try:
                        polytope_mesh = polytope.to_mesh()
                        polytope_mesh.paint_uniform_color([0.8, 0.8, 0.8])
                        polytope_mesh.compute_vertex_normals()
                        polytope_mesh.compute_triangle_normals()
                        vis.add_geometry(polytope_mesh)
                    except Exception as e:
                        print(f"无法创建多胞体网格: {e}")
            except Exception as e:
                print(f"处理安全区域 {i} 时出错: {e}")
    if path is not None and len(path) > 1:
        lines = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(path)
        lines.points = points
        lines.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path)-1)])
        lines.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(lines)
        for point in path:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(point)
            sphere.paint_uniform_color([1.0, 0.5, 0.0])
            vis.add_geometry(sphere)
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open('temp/path_points.pkl', 'wb') as f:
            pickle.dump(np.asarray(path), f)
    render_opt = vis.get_render_option()
    render_opt.line_width = 5.0
    render_opt.background_color = np.array([0.9, 0.9, 0.9])
    render_opt.light_on = True
    ctr = vis.get_view_control()
    ctr.set_lookat([5, 5, 5])
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)
    vis.run()
    vis.destroy_window()

def visualize_path_only(obstacles, path, start, goal):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)
    for obs in obstacles:
        obs.paint_uniform_color([0.7, 0.1, 0.1])
        vis.add_geometry(obs)
    if path is not None and len(path) > 1:
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(path)
        lines.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path)-1)])
        lines.paint_uniform_color([0.0, 0.8, 0.0])
        vis.add_geometry(lines)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0.0, 0.0, 1.0])
    vis.add_geometry(start_sphere)
    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    goal_sphere.translate(goal)
    goal_sphere.paint_uniform_color([1.0, 1.0, 0.0])
    vis.add_geometry(goal_sphere)
    render_opt = vis.get_render_option()
    render_opt.line_width = 8.0
    render_opt.background_color = np.array([0.9, 0.9, 0.9])
    render_opt.light_on = True
    render_opt.point_size = 10.0
    ctr = vis.get_view_control()
    ctr.set_lookat([5, 5, 5])
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)
    vis.run()
    vis.destroy_window()

##########################################
# 增强可视化和结果分析
##########################################
def visualize_with_open3d(path, obstacles, start, goal, window_size=(1280, 720)):
    """
    Visualize the path planning results with Open3D
    
    Parameters:
        path: Planned path points
        obstacles: Obstacle set
        start: Start point coordinates
        goal: Goal point coordinates
        window_size: Visualization window size
    """
    import open3d as o3d
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_size[0], height=window_size[1])
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)
    
    # Add obstacles
    for obstacle in obstacles:
        if hasattr(obstacle, 'to_mesh'):
            # Use the obstacle's own mesh generation if available
            mesh = obstacle.to_mesh()
        else:
            # Fallback for old-style obstacles
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=obstacle.radius)
            mesh.translate(obstacle.center)
            mesh.compute_vertex_normals()
        
        mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red for obstacles
        vis.add_geometry(mesh)
    
    # Add start and goal points
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green for start
    vis.add_geometry(start_sphere)
    
    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    goal_sphere.translate(goal)
    goal_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for goal
    vis.add_geometry(goal_sphere)
    
    # Add path as line set
    if len(path) > 1:
        points = o3d.utility.Vector3dVector(path)
        lines = [[i, i+1] for i in range(len(path)-1)]
        colors = [[1.0, 1.0, 0.0] for _ in range(len(lines))]  # Yellow for path
        
        line_set = o3d.geometry.LineSet()
        line_set.points = points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
        
        # Add path points as small spheres
        for point in path:
            point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            point_sphere.translate(point)
            point_sphere.paint_uniform_color([1.0, 0.5, 0.0])  # Orange for path points
            vis.add_geometry(point_sphere)
    
    # Set view control
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 5.0
    opt.mesh_show_wireframe = True  # Show wireframe for better visualization of shapes
    
    # Set default viewpoint
    ctr = vis.get_view_control()
    ctr.set_lookat([5, 5, 5])
    ctr.set_front([1, 1, 1])  # View from diagonal direction
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)
    
    # Render view
    vis.run()
    
    # Capture screenshot
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    plt.axis('off')
    plt.savefig('temp/open3d_path_planning.png', bbox_inches='tight', pad_inches=0)
    
    vis.destroy_window()

def analyze_planning_results(path, obstacles, start, goal, planning_time):
    """
    Analyze path planning results and generate detailed report
    
    Parameters:
        path: Planned path points
        obstacles: List of obstacles
        start: Start coordinates
        goal: Goal coordinates
        planning_time: Planning time (seconds)
    """
    print("\n======= Path Planning Result Analysis =======")
    
    # Create results directory
    if not os.path.exists('temp'):
        os.makedirs('temp')
        
    # Calculate path length
    path_length = 0.0
    for i in range(len(path)-1):
        segment_length = np.linalg.norm(path[i+1] - path[i])
        path_length += segment_length
    
    # Calculate direct distance
    direct_distance = np.linalg.norm(goal - start)
    
    # Calculate path curvature (ratio of path length to direct distance)
    path_curvature_ratio = path_length / direct_distance if direct_distance > 0 else 1.0
    
    # Calculate path smoothness (through angle changes)
    angles = []
    sharp_turns = 0
    if len(path) > 2:
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                angles.append(angle_deg)
                if angle_deg > 90:
                    sharp_turns += 1
    
    avg_angle = np.mean(angles) if angles else 0
    max_angle = np.max(angles) if angles else 0
    smoothness_score = 10 - min(10, max_angle / 18)  # Map max angle to 0-10 score
    
    # Check if each path segment collides with obstacles
    planner = FIRIPlanner(obstacles)
    path_safety = planner.check_path_safety(path)
    
    # Generate report
    report = {
        'path_points': len(path),
        'path_length': path_length,
        'direct_distance': direct_distance,
        'path_curvature_ratio': path_curvature_ratio,
        'avg_angle': avg_angle,
        'max_angle': max_angle,
        'sharp_turns': sharp_turns,
        'smoothness_score': smoothness_score,
        'is_collision_free': path_safety['is_safe'],
        'collision_segments': path_safety['collision_count'],
        'planning_time': planning_time
    }
    
    # Print report
    print(f"Path points count: {report['path_points']}")
    print(f"Total path length: {report['path_length']:.2f} units")
    print(f"Direct distance between start and goal: {report['direct_distance']:.2f} units")
    print(f"Path curvature ratio: {report['path_curvature_ratio']:.2f} (closer to 1 is straighter)")
    print(f"Average turning angle: {report['avg_angle']:.2f}°")
    print(f"Maximum turning angle: {report['max_angle']:.2f}°")
    print(f"Sharp turns count (>90°): {report['sharp_turns']}")
    print(f"Smoothness score (0-10): {report['smoothness_score']:.2f}")
    print(f"Collision free: {'Yes' if report['is_collision_free'] else 'No'}")
    if not report['is_collision_free']:
        print(f"Collision segments count: {report['collision_segments']}")
    print(f"Planning time: {report['planning_time']:.4f} seconds")
    
    # Save to file
    with open('temp/planning_analysis.txt', 'w') as f:
        f.write("====== FIRI Path Planning Analysis Report ======\n\n")
        for key, value in report.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Visualize path and angle distribution
    plt.figure(figsize=(15, 5))
    
    # 3D path plot
    ax1 = plt.subplot(131, projection='3d')
    ax1.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], 'o-', linewidth=2)
    ax1.scatter([start[0]], [start[1]], [start[2]], c='g', s=100, label='Start')
    ax1.scatter([goal[0]], [goal[1]], [goal[2]], c='r', s=100, label='Goal')
    ax1.set_title('3D Path')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Segment length distribution
    segment_lengths = [np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]
    ax2 = plt.subplot(132)
    ax2.bar(range(len(segment_lengths)), segment_lengths)
    ax2.set_title('Segment Length Distribution')
    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Length')
    ax2.grid(True)
    
    # Angle distribution
    if angles:
        ax3 = plt.subplot(133)
        ax3.bar(range(len(angles)), angles)
        ax3.axhline(y=90, color='r', linestyle='--', label='90° Threshold')
        ax3.set_title('Turn Angle Distribution')
        ax3.set_xlabel('Turn Point Index')
        ax3.set_ylabel('Angle (°)')
        ax3.grid(True)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig('temp/path_analysis.png')
    
    # Generate comparison chart
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [direct_distance, path_length], 'o-', linewidth=2)
    plt.text(0, direct_distance, f"Direct: {direct_distance:.2f}")
    plt.text(1, path_length, f"Path: {path_length:.2f}")
    plt.title(f'Path Length Comparison (Ratio: {path_curvature_ratio:.2f})')
    plt.xticks([0, 1], ['Direct Distance', 'Actual Path'])
    plt.grid(True)
    plt.savefig('temp/path_comparison.png')
    
    return report

def clean_temp_files():
    """Clean temporary files from previous runs"""
    if not os.path.exists('temp'):
        os.makedirs('temp')
        print("Created temp directory")
    else:
        for file in os.listdir('temp'):
            file_path = os.path.join('temp', file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def generate_random_obstacles(num_obstacles=6):
    """Generate random 3D obstacles with different shapes and ensure at least one is on the path"""
    obstacles = []
    min_coords = np.array([-5, -5, -5])
    max_coords = np.array([15, 15, 5])
    
    # Ensure no obstacles near start and goal points
    start_point = np.array([0.0, 0.0, 0.0])
    goal_point = np.array([10.0, 10.0, 0.0])
    goal_safety_buffer = 3.0  # Increased from 1.0 to ensure space around start/goal
    
    # First, place a strategic obstacle in the middle of the direct path
    # This will force the planner to find a path around it
    direct_vector = goal_point - start_point
    direct_distance = np.linalg.norm(direct_vector)
    unit_direction = direct_vector / direct_distance
    
    # Position the obstacle at around the midpoint of the direct path
    mid_point = start_point + 0.5 * direct_vector
    
    # Add a small random offset to make it more interesting
    # but ensure it's still blocking the direct path
    offset = np.random.uniform(-0.3, 0.3, 3)
    # Make sure the offset is perpendicular to the path direction
    offset = offset - np.dot(offset, unit_direction) * unit_direction
    # Limit the offset magnitude
    if np.linalg.norm(offset) > 0.5:
        offset = offset / np.linalg.norm(offset) * 0.5
    
    strategic_position = mid_point + offset
    
    # Create a larger obstacle to ensure path blocking
    # Randomly choose type for strategic obstacle
    strategic_type = np.random.choice(['sphere', 'cylinder', 'box'])
    
    if strategic_type == 'sphere':
        strategic_radius = np.random.uniform(1.5, 2.0)
        strategic_obstacle = Obstacle(strategic_position, strategic_radius, 'sphere')
    elif strategic_type == 'cylinder':
        cylinder_radius = np.random.uniform(1.2, 1.8)
        cylinder_height = np.random.uniform(2.0, 3.0)
        strategic_radius = max(cylinder_radius, cylinder_height/2)  # For collision detection
        strategic_obstacle = Obstacle(strategic_position, strategic_radius, 'cylinder', [cylinder_radius, cylinder_height])
    else:  # box
        box_dims = np.random.uniform(1.5, 2.5, 3)  # width, height, depth
        box_radius = np.sqrt(np.sum(np.square(box_dims))) / 2  # For collision detection
        strategic_obstacle = Obstacle(strategic_position, box_radius, 'box', box_dims)
    
    obstacles.append(strategic_obstacle)
    
    print(f"Strategic obstacle placed at {strategic_position}, type: {strategic_type}, ensuring path must go around")
    
    # Generate remaining random obstacles with different shapes
    for _ in range(num_obstacles - 1):
        while True:
            center = np.random.uniform(min_coords, max_coords)
            
            # Determine obstacle type and size
            obstacle_type = np.random.choice(['sphere', 'cylinder', 'box'])
            
            if obstacle_type == 'sphere':
                radius = np.random.uniform(0.5, 2.0)
                dimensions = None
            elif obstacle_type == 'cylinder':
                cylinder_radius = np.random.uniform(0.5, 1.5)
                cylinder_height = np.random.uniform(1.0, 3.0)
                radius = max(cylinder_radius, cylinder_height/2)  # For collision detection
                dimensions = [cylinder_radius, cylinder_height]
            else:  # box
                size = np.random.uniform(0.5, 2.0, 3)  # width, height, depth
                radius = np.sqrt(np.sum(np.square(size))) / 2  # For collision detection
                dimensions = size
            
            # Check if obstacle is too close to start or goal
            if (np.linalg.norm(center - start_point) < radius + goal_safety_buffer or
                np.linalg.norm(center - goal_point) < radius + goal_safety_buffer):
                continue
            
            # If we're placing obstacles after the strategic one, check for overlap
            overlap = False
            for existing_obstacle in obstacles:
                dist = np.linalg.norm(center - existing_obstacle.center)
                if dist < radius + existing_obstacle.radius:
                    overlap = True
                    break
            
            if not overlap:
                # Check if this obstacle is on or near the path line
                # We want some obstacles near the path, but not too many
                v = goal_point - start_point
                v_len = np.linalg.norm(v)
                v_unit = v / v_len
                p = center - start_point
                # Projection of p onto v
                proj_len = np.dot(p, v_unit)
                # Calculate perpendicular distance to the line
                if 0 <= proj_len <= v_len:  # Check if projection is on line segment
                    proj_point = start_point + proj_len * v_unit
                    perp_dist = np.linalg.norm(center - proj_point)
                    # If too close to the strategic obstacle, try again
                    if perp_dist < 3.0 and np.linalg.norm(center - strategic_position) < strategic_obstacle.radius + radius + 1.0:
                        continue
                
                obstacle = Obstacle(center, radius, obstacle_type, dimensions)
                obstacles.append(obstacle)
                print(f"Obstacle placed at {center}, type: {obstacle_type}, radius/size: {dimensions if dimensions is not None else radius}")
                break
    
    return obstacles

##########################################
# 主程序入口 - 增强版
##########################################
if __name__ == "__main__":
    import copy
    import time
    import argparse
    
    # Process command-line arguments
    parser = argparse.ArgumentParser(description='FIRI Path Planning Test')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--obstacles', type=int, default=16, help='Number of obstacles to generate')
    parser.add_argument('--margin', type=float, default=1.5, help='Safety margin for obstacles')
    args = parser.parse_args()
    
    # Display run parameters
    print(f"Running with: obstacles={args.obstacles}, safety_margin={args.margin}, visualization={'disabled' if args.no_viz else 'enabled'}")
    
    # Clean temporary files
    print("\nCleaning temporary files...")
    clean_temp_files()
    
    # Generate random obstacles
    print("\nGenerating obstacles...")
    obstacles = generate_random_obstacles(args.obstacles)
    
    # Save obstacle data (only center and radius, avoid pickling Open3D objects)
    print("Saving obstacle data...")
    obstacle_data = [(obs.center, obs.radius) for obs in obstacles]
    with open('temp/obstacle_data.pkl', 'wb') as f:
        pickle.dump(obstacle_data, f)
    
    # Create obstacle set for collision detection
    obstacle_set = ObstacleSet()
    for obstacle in obstacles:
        obstacle_set.add_obstacle(obstacle)
    
    # Set start and goal points
    start_point = np.array([0.0, 0.0, 0.0])
    goal_point = np.array([10.0, 10.0, 0.0])
    
    # Save start and goal points
    with open('temp/waypoints.pkl', 'wb') as f:
        pickle.dump({'start': start_point, 'goal': goal_point}, f)
    
    # Initialize FIRI planner
    print("\nInitializing FIRI path planner...")
    planner = FIRIPlanner(obstacle_set)
    
    print(f"Number of obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"Obstacle {i+1}: center={obs.center}, radius={obs.radius}")
    
    # Plan path
    print("\nPlanning path...")
    start_time = time.time()
    path = planner.plan_path(start_point, goal_point)
    planning_time = time.time() - start_time
    
    # We're already smoothing in the planner, so we'll skip this step
    print(f"Path planning completed in {planning_time:.4f} seconds")
    
    # Check path safety
    safety_result = planner.check_path_safety(path)
    print(f"Path is safe: {safety_result['is_safe']}")
    if not safety_result['is_safe']:
        print(f"Collision detected in {safety_result['collision_count']} path segments")
    
    # Analyze path planning results
    analysis = analyze_planning_results(path, obstacle_set, start_point, goal_point, planning_time)
    
    # Use Open3D for 3D visualization
    if not args.no_viz:
        print("\nStarting Open3D visualization...")
        visualize_with_open3d(path, obstacle_set, start_point, goal_point)
    else:
        print("\nOpen3D visualization disabled")
    
    print("\nResults saved to temp directory:")
    print("- obstacle_data.pkl: Obstacle data")
    print("- final_path.pkl: Final planned path")
    print("- planning_analysis.txt: Path planning analysis report")
    print("- path_analysis.png: Path analysis visualization")
    print("- path_comparison.png: Path projection visualizations")
    if not args.no_viz:
        print("- open3d_path_planning.png: Open3D visualization screenshot")
    
    print("\nUsage:")
    print("- Run with visualization: python test/peng_test_v3.py")
    print("- Run without visualization: python test/peng_test_v3.py --no-viz")
    print("- Customize parameters: python test/peng_test_v3.py --obstacles 10 --margin 2.0")

