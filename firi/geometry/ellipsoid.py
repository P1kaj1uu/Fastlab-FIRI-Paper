import numpy as np
import scipy.linalg as spla

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
        
        # 确保Q是半正定矩阵，使用更稳健的方式检查
        try:
            # 使用Cholesky分解检查正定性，如果失败则调整
            _ = spla.cholesky(self.Q, lower=True)
        except np.linalg.LinAlgError:
            # 如果不是正定矩阵，强制使其正定
            eigvals, eigvecs = np.linalg.eigh(self.Q)
            min_eig = min(eigvals)
            if min_eig < 1e-8:
                # 将负或接近零的特征值调整为正值，保持条件数合理
                eigvals = np.maximum(eigvals, np.max(eigvals) * 1e-8)
                self.Q = eigvecs @ np.diag(eigvals) @ eigvecs.T
                print(f"警告: Q矩阵被调整为正定矩阵，最小特征值从 {min_eig} 调整为 {np.min(eigvals)}")

        # 计算缓存值，提高后续计算效率
        self._cached_values = {}
        self._svd_valid = False
        self._compute_svd()

    def _compute_svd(self):
        """
        预计算Q的SVD分解和各种衍生矩阵，提高数值稳定性
        缓存结果以避免重复计算
        """
        try:
            # 使用经济型SVD分解，减少计算量
            u, s, vh = spla.svd(self.Q, full_matrices=False, lapack_driver='gesvd')
            
            # 缓存原始SVD结果
            self._cached_values['svd_u'] = u
            self._cached_values['svd_s'] = s
            self._cached_values['svd_vh'] = vh
            
            # 计算和缓存条件数
            max_s = np.max(s)
            min_s = np.min(s)
            self.condition_number = max_s / min_s if min_s > 1e-10 else float('inf')
            
            # 计算阈值，适应不同尺度的数据
            threshold = max(max_s * 1e-12, 1e-15)
            
            # 过滤小奇异值以提高稳定性
            s_safe = np.maximum(s, threshold)
            s_inv = 1.0 / s_safe
            
            # 缓存平方根和逆平方根
            s_sqrt = np.sqrt(s_safe)
            s_sqrt_inv = 1.0 / s_sqrt
            
            # 保存所有需要的矩阵运算结果
            self._cached_values['Q_inv'] = (vh.T * s_inv) @ u.T
            self._cached_values['Q_sqrt'] = u @ np.diag(s_sqrt) @ vh
            self._cached_values['Q_sqrt_inv'] = vh.T @ np.diag(s_sqrt_inv) @ u.T
            
            # 标记SVD结果有效
            self._svd_valid = True
            
            # 保存体积的对数值，避免溢出
            self._cached_values['log_det_Q'] = np.sum(np.log(s_safe))
            
        except Exception as e:
            print(f"SVD分解出错，使用备用方法: {e}")
            try:
                # 备用方法：使用特征值分解
                eigvals, eigvecs = np.linalg.eigh(self.Q)
                
                # 类似于SVD，处理小特征值
                max_eig = np.max(eigvals)
                threshold = max(max_eig * 1e-12, 1e-15)
                eigvals_safe = np.maximum(eigvals, threshold)
                
                eigvals_inv = 1.0 / eigvals_safe
                eigvals_sqrt = np.sqrt(eigvals_safe)
                eigvals_sqrt_inv = 1.0 / eigvals_sqrt
                
                # 缓存计算结果
                self._cached_values['Q_inv'] = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T
                self._cached_values['Q_sqrt'] = eigvecs @ np.diag(eigvals_sqrt) @ eigvecs.T
                self._cached_values['Q_sqrt_inv'] = eigvecs @ np.diag(eigvals_sqrt_inv) @ eigvecs.T
                self._cached_values['log_det_Q'] = np.sum(np.log(eigvals_safe))
                
                self.condition_number = max_eig / np.min(eigvals_safe)
                self._svd_valid = True
                
            except Exception as err:
                print(f"备用方法也失败，使用默认值: {err}")
                # 使用安全的默认值
                self._cached_values['Q_inv'] = np.eye(self.dim)
                self._cached_values['Q_sqrt'] = np.eye(self.dim)
                self._cached_values['Q_sqrt_inv'] = np.eye(self.dim)
                self._cached_values['log_det_Q'] = 0.0
                self.condition_number = float('inf')
                self._svd_valid = False

    def _ensure_valid_svd(self):
        """确保SVD计算有效，如果无效则重新计算"""
        if not self._svd_valid or self.condition_number > 1e8:
            self._compute_svd()
        return self._svd_valid

    def volume(self):
        """
        计算椭球体体积，使用对数计算以避免数值溢出
        """
        self._ensure_valid_svd()
        try:
            # 使用预计算的对数行列式
            log_det_Q = self._cached_values.get('log_det_Q', 0.0)
            
            if log_det_Q <= -float('inf'):
                return 0.0
                
            # 计算对数体积
            if self.dim == 3:
                log_vol = np.log(4.0/3.0) + np.log(np.pi) + 0.5 * log_det_Q
            elif self.dim == 2:
                log_vol = np.log(np.pi) + 0.5 * log_det_Q
            else:
                # 通用公式
                log_vol = (self.dim/2) * np.log(np.pi) - np.log(np.math.gamma(self.dim/2 + 1)) + 0.5 * log_det_Q
            
            # 从对数转回实际体积
            return np.exp(log_vol)
        except Exception as e:
            print(f"体积计算出错: {e}")
            return 0.0
    
    def contains(self, point):
        """检查点是否在椭球体内，加强数值稳定性"""
        self._ensure_valid_svd()
        vec = np.array(point) - self.center
        try:
            # 使用预计算的Q_inv，应用二次型
            Q_inv = self._cached_values.get('Q_inv', np.eye(self.dim))
            dist_squared = vec @ Q_inv @ vec
            
            # 添加一点容差，避免数值误差导致的错误判断
            return dist_squared <= 1.001
        except Exception as e:
            print(f"包含检查出错: {e}")
            # 如果计算失败，使用欧氏距离作为备用
            radius_estimate = np.sqrt(np.max(np.diag(self.Q)))
            return np.linalg.norm(vec) <= radius_estimate
    
    def transform_point(self, point):
        """将点从世界坐标变换到椭球体标准坐标系（椭球体->单位球）"""
        self._ensure_valid_svd()
        try:
            # 应用变换 x' = Q^(-1/2)(x - c)
            Q_sqrt_inv = self._cached_values.get('Q_sqrt_inv', np.eye(self.dim))
            return Q_sqrt_inv @ (np.array(point) - self.center)
        except Exception as e:
            print(f"变换点时出错: {e}")
            # 简单返回相对于中心的坐标
            return np.array(point) - self.center
    
    def inverse_transform_point(self, point):
        """将点从椭球体标准坐标系变换回世界坐标（单位球->椭球体）"""
        self._ensure_valid_svd()
        try:
            # 应用逆变换: x = Q^(1/2)y + c
            Q_sqrt = self._cached_values.get('Q_sqrt', np.eye(self.dim))
            return self.center + Q_sqrt @ np.array(point)
        except Exception as e:
            print(f"逆变换点时出错: {e}")
            # 返回一个默认逆变换
            return self.center + np.array(point)
    
    def transform_halfspace(self, halfspace):
        """
        将半空间从原始坐标系变换到椭球体标准坐标系
        按照论文式(15)实现变换: {x | a^T x + b ≤ 0} 变换为 {y | a'^T y + b' ≤ 0}
        a' = Q^(1/2) a / ||Q^(1/2) a||
        b' = (b + a^T c) / ||Q^(1/2) a||
        
        优化数值稳定性处理
        """
        a = halfspace[:-1]  # 法向量
        b = halfspace[-1]   # 偏移量
        
        # 检查法向量是否为零向量
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-10:
            print("警告: 半空间变换中法向量接近零向量")
            # 返回一个安全的默认半空间
            result = np.zeros_like(halfspace)
            result[0] = 1.0  # 默认指向x轴正方向
            return result
            
        # 确保法向量是单位向量
        a = a / a_norm
        b = b / a_norm
        
        # 确保SVD有效
        self._ensure_valid_svd()
        
        try:
            # 使用缓存的Q^(1/2)
            Q_sqrt = self._cached_values.get('Q_sqrt', np.eye(self.dim))
            Q_sqrt_a = Q_sqrt @ a
            
            # 计算范数，带保护措施
            norm = np.linalg.norm(Q_sqrt_a)
            
            if norm < 1e-10:
                print("警告: 半空间变换中法向量映射接近零向量")
                # 返回一个默认的半空间
                result = np.zeros_like(halfspace)
                result[0] = 1.0
                return result
            
            # 按照公式计算变换后的半空间参数
            a_prime = Q_sqrt_a / norm
            b_prime = (b + np.dot(a, self.center)) / norm
            
            # 检查结果的有效性
            if not np.all(np.isfinite(a_prime)) or not np.isfinite(b_prime):
                raise ValueError("半空间变换产生非有限值")
                
            # 返回变换后的半空间
            transformed_halfspace = np.zeros_like(halfspace)
            transformed_halfspace[:-1] = a_prime
            transformed_halfspace[-1] = b_prime
            
            return transformed_halfspace
            
        except Exception as e:
            print(f"半空间变换出错: {e}")
            # 返回一个默认的半空间
            result = np.zeros_like(halfspace)
            result[0] = 1.0
            return result
    
    def inverse_transform_halfspace(self, halfspace):
        """
        将半空间从椭球体标准坐标系变换回原始坐标系
        变换公式: a = Q^(1/2) a', b = b' * ||Q^(1/2) a'|| - a'^T Q^(1/2) c
        
        优化数值稳定性处理
        """
        a_std = halfspace[:-1]  # 标准空间中的法向量
        b_std = halfspace[-1]   # 标准空间中的偏移量
        
        # 确保SVD有效
        self._ensure_valid_svd()
        
        try:
            # 使用缓存的Q^(1/2)
            Q_sqrt = self._cached_values.get('Q_sqrt', np.eye(self.dim))
            a_original = Q_sqrt @ a_std
            
            # 计算范数
            norm = np.linalg.norm(a_original)
            
            if norm < 1e-10:
                print("警告: 半空间逆变换中法向量接近零向量")
                # 返回一个默认的半空间
                result = np.zeros_like(halfspace)
                result[0] = 1.0
                return result
                
            # 可选：规范化法向量
            a_original = a_original / norm
            
            # 重新计算偏移量
            b_original = b_std * norm - np.dot(a_std, Q_sqrt @ self.center)
            
            # 检查结果的有效性
            if not np.all(np.isfinite(a_original)) or not np.isfinite(b_original):
                raise ValueError("半空间逆变换产生非有限值")
                
            # 返回变换后的半空间
            transformed_halfspace = np.zeros_like(halfspace)
            transformed_halfspace[:-1] = a_original
            transformed_halfspace[-1] = b_original
            
            return transformed_halfspace
            
        except Exception as e:
            print(f"半空间逆变换出错: {e}")
            # 返回一个默认的半空间
            result = np.zeros_like(halfspace)
            result[0] = 1.0
            return result
    
    def to_mesh(self):
        """创建椭球体的网格表示，用于可视化"""
        try:
            import open3d as o3d
            
            # 创建单位球
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            vertices = np.asarray(mesh.vertices)
            
            # 确保SVD有效
            self._ensure_valid_svd()
            
            # 变换顶点
            Q_sqrt = self._cached_values.get('Q_sqrt', np.eye(self.dim))
            transformed_vertices = vertices @ Q_sqrt.T + self.center
            
            # 更新网格顶点
            mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            mesh.compute_vertex_normals()
            
            return mesh
        except Exception as e:
            print(f"创建椭球体网格时出错: {e}")
            return None 