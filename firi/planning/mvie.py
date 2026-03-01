import numpy as np
import time
from ..geometry import Ellipsoid

class MVIE_SOCP:
    """
    使用二阶锥规划(SOCP)方法计算最大体积内接椭球 (MVIE)
    基于论文: "SOCP-based algorithm for the minimum volume enclosing ellipsoid"
    """
    def __init__(self, dimension=3):
        """初始化MVIE计算器"""
        self.dim = dimension
        self.max_iterations = 100
        self.eps = 1e-8
        self.condition_threshold = 1e12  # 用于检查矩阵条件数
        
    def compute(self, polytope):
        """
        使用SOCP模型计算MVIE
        
        参数:
            polytope: 凸多胞体对象，包含半空间表示
            
        返回:
            Ellipsoid对象，表示最大体积内接椭球
        """
        # 获取多胞体的半空间表示
        halfspaces = polytope.get_halfspaces()
        if halfspaces is None or halfspaces.shape[0] < self.dim + 1:
            print("警告: 多胞体没有有效的半空间表示")
            # 返回一个默认椭球体
            interior_point = polytope.get_interior_point()
            if interior_point is None:
                interior_point = np.zeros(self.dim)
            return Ellipsoid(interior_point, np.eye(self.dim) * 0.1)
        
        # 提取法向量A和偏移量b
        A = halfspaces[:, :-1]
        b = -halfspaces[:, -1]  # 注意符号变换
        
        # 过滤掉范数接近零的行
        norms = np.linalg.norm(A, axis=1)
        valid_indices = norms > 1e-10
        if np.sum(valid_indices) < self.dim + 1:
            print(f"警告: 有效半空间约束不足，只有 {np.sum(valid_indices)} 个")
            # 返回一个默认椭球体
            interior_point = polytope.get_interior_point()
            if interior_point is None:
                interior_point = np.zeros(self.dim)
            return Ellipsoid(interior_point, np.eye(self.dim) * 0.1)
            
        A = A[valid_indices]
        b = b[valid_indices]
        
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
            self._solve_affine_scaling,  # 首选: 改进的Affine Scaling方法
            self._solve_khachiyan,       # 备选: Khachiyan迭代算法
            self._solve_cvxpy            # 后备: CVXPY通用求解器(需要安装)
        ]
        
        start_time = time.time()
        best_ellipsoid = None
        best_volume = -1
        
        for method in methods:
            try:
                print(f"  尝试使用{method.__name__[7:]}方法求解MVIE...")
                E, center_opt = method(A, b, center)
                
                if E is not None and center_opt is not None:
                    # 构造椭球体
                    Q = E @ E.T
                    
                    # 验证矩阵是否有效
                    if self._is_valid_matrix(Q):
                        # 创建椭球体对象并计算体积
                        ellipsoid = Ellipsoid(center_opt, Q)
                        vol = ellipsoid.volume()
                        
                        # 检查体积是否合理
                        if vol > 0 and vol < 1e12 and not np.isnan(vol) and not np.isinf(vol):
                            print(f"  {method.__name__[7:]}方法成功，椭球体体积: {vol:.6f}")
                            
                            # 优先选择较大体积的椭球
                            if vol > best_volume:
                                best_volume = vol
                                best_ellipsoid = ellipsoid
                                
                            # 如果已经找到体积合理的解且计算时间超过2秒，不再尝试其他方法
                            if time.time() - start_time > 2 and best_ellipsoid is not None:
                                print("  已找到合理解，不再尝试其他方法")
                                break
                        else:
                            print(f"  {method.__name__[7:]}方法得到的体积异常: {vol:.6f}")
                    else:
                        print(f"  {method.__name__[7:]}方法求解结果矩阵无效")
            except Exception as e:
                print(f"  {method.__name__[7:]}方法出错: {e}")
        
        # 如果所有方法都失败，返回默认椭球
        if best_ellipsoid is None:
            print("  所有MVIE方法均失败，使用默认椭球")
            return Ellipsoid(center, np.eye(self.dim) * 0.1)
            
        return best_ellipsoid
    
    def _solve_affine_scaling(self, A, b, center_init, max_iter=100, tol=1e-6):
        """
        使用仿射缩放法求解MVIE问题
        
        参数:
            A: 半空间约束的法向量矩阵
            b: 半空间约束的偏移量向量
            center_init: 初始中心点
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            (E, center): E是形状矩阵，center是椭球体中心
        """
        m, n = A.shape  # m个约束，n维空间
        
        # 预处理：规范化所有半空间法向量
        norms = np.linalg.norm(A, axis=1)
        mask = norms > 1e-10  # 排除近似零向量
        
        if np.sum(mask) < n:
            print("  警告: 约束数量不足，问题可能无界")
            return np.eye(n), center_init.copy()
            
        # 归一化法向量和偏移量
        A_norm = np.zeros_like(A)
        b_norm = np.zeros_like(b)
        
        A_norm[mask] = A[mask] / norms[mask].reshape(-1, 1)
        b_norm[mask] = b[mask] / norms[mask]
        
        # 初始化
        E = np.eye(n)           # 初始矩阵为单位矩阵
        center = center_init.copy()
        
        # 初始化拉格朗日乘子 (均匀分布)
        lambda_vec = np.ones(m) / m
        lambda_vec[~mask] = 0.0  # 无效约束权重为0
        lambda_vec = lambda_vec / np.sum(lambda_vec)
        
        # 自适应步长控制
        initial_step = 0.2
        step_decay = 0.95
        min_step = 0.02
        
        # 记录上一次的目标函数值和最佳状态
        prev_det_E = np.linalg.det(E)
        best_E = E.copy()
        best_center = center.copy()
        best_vol = prev_det_E
        best_iter = 0
        
        # 体积变化阈值，超过此值视为不稳定更新
        vol_change_threshold = 3.0
        
        # 保存迭代历史，用于检测振荡
        vol_history = []
        center_history = []
        
        # 迭代求解
        for iter_idx in range(max_iter):
            try:
                # 计算每个约束的违反程度
                AE = np.zeros((m, n))
                for i in range(m):
                    if mask[i]:
                        AE[i] = A_norm[i] @ E
                
                norms_AE = np.linalg.norm(AE, axis=1)
                margins = b_norm - A_norm @ center
                
                # 过滤掉无效约束
                valid_indices = mask & (norms_AE > 1e-10)
                valid_norms = norms_AE[valid_indices] 
                valid_margins = margins[valid_indices]
                
                # 计算违反程度
                violations = np.zeros(m)
                violations[valid_indices] = valid_norms - valid_margins
                
                # 检查收敛
                max_violation = np.max(violations)
                if max_violation < tol:
                    print(f"  Affine Scaling方法收敛，迭代{iter_idx+1}次")
                    # 即使收敛，也要确保使用最佳状态
                    if best_vol > prev_det_E:
                        return best_E, best_center
                    break
                
                # 动态步长
                current_step = max(min_step, initial_step * (step_decay ** iter_idx))
                
                # 计算相对违反度
                rel_violations = np.zeros(m)
                rel_violations[valid_indices] = violations[valid_indices] / (valid_norms + 1e-10)
                
                # 更新lambda - 使用平滑的指数更新规则
                # 使用当前步长而不是固定值
                lambda_new = lambda_vec * np.exp(current_step * rel_violations)
                lambda_sum = np.sum(lambda_new)
                
                if lambda_sum > 1e-10:
                    lambda_vec = lambda_new / lambda_sum  # 归一化
                else:
                    # 如果所有lambda都接近零，重新初始化
                    lambda_vec = np.ones(m) / m
                    lambda_vec[~mask] = 0.0
                    lambda_vec = lambda_vec / np.sum(lambda_vec)
                
                # 构造M矩阵 (式18)，使用数值稳定的方式
                M = np.zeros((n, n))
                weighted_vectors = []
                weights = []
                
                for i in range(m):
                    if valid_indices[i] and lambda_vec[i] > 1e-10:
                        # 收集所有有效的向量和权重，避免逐个累加导致的误差
                        ai = A_norm[i]
                        weight = lambda_vec[i] / norms_AE[i]
                        weighted_vectors.append(ai)
                        weights.append(weight)
                
                # 批量计算M矩阵
                if weighted_vectors:
                    weighted_vectors = np.array(weighted_vectors)
                    weights = np.array(weights).reshape(-1, 1)
                    # 使用矩阵运算代替循环
                    weighted_ai = weighted_vectors * np.sqrt(weights)
                    M = weighted_ai.T @ weighted_ai
                
                # 添加正则化项，避免M接近奇异
                M += np.eye(n) * 1e-10 * np.trace(M) / n
                
                # 使用SVD分解求解E矩阵
                try:
                    # 使用Cholesky分解，更适合正定矩阵
                    try:
                        # 首先尝试Cholesky分解，更快
                        L = np.linalg.cholesky(M)
                        M_inv_sqrt = np.linalg.inv(L)
                        E_new = M_inv_sqrt
                    except np.linalg.LinAlgError:
                        # 如果Cholesky失败，使用SVD
                        U, s, Vh = np.linalg.svd(M)
                        
                        # 过滤太小的奇异值
                        max_s = np.max(s)
                        threshold = max(max_s * 1e-10, 1e-15)
                        s_inv_sqrt = np.where(s > threshold, 1.0/np.sqrt(s), 0.0)
                        
                        # 构造E = (M^(-1/2))
                        E_new = U @ np.diag(s_inv_sqrt) @ Vh
                    
                    # 渐进更新，避免过大变化
                    alpha = max(min_step, current_step)
                    E = (1 - alpha) * E + alpha * E_new
                    
                except Exception as e:
                    print(f"  E矩阵计算出错: {e}")
                    # 保守更新
                    alpha = min_step
                    # 使用扰动的特征值分解作为备用
                    eigvals, eigvecs = np.linalg.eigh(M + 1e-8 * np.eye(n))
                    eigvals = np.maximum(eigvals, 1e-8 * np.max(eigvals))
                    E_new = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    E = (1 - alpha) * E + alpha * E_new
                
                # 更新中心点 - 使用加权最小二乘
                try:
                    # 过滤有效约束
                    valid_lambda = lambda_vec[valid_indices]
                    valid_A = A_norm[valid_indices]
                    valid_b = b_norm[valid_indices]
                    
                    if len(valid_lambda) >= n and np.sum(valid_lambda) > 1e-10:
                        # 构建加权最小二乘问题
                        lambda_sqrt = np.sqrt(valid_lambda)
                        A_weighted = valid_A * lambda_sqrt.reshape(-1, 1)
                        b_weighted = valid_b * lambda_sqrt
                        
                        # 添加正则化
                        reg_param = 1e-6
                        A_augmented = np.vstack([A_weighted, np.sqrt(reg_param) * np.eye(n)])
                        b_augmented = np.concatenate([b_weighted, np.sqrt(reg_param) * center])
                        
                        # 使用QR分解求解，更稳定
                        Q, R = np.linalg.qr(A_augmented)
                        y = Q.T @ b_augmented
                        try:
                            center_new = np.linalg.solve(R, y)
                            # 检查解的合理性
                            if np.all(np.isfinite(center_new)) and np.max(np.abs(center_new)) < 1e5:
                                # 平滑更新
                                center = (1 - current_step) * center + current_step * center_new
                            else:
                                raise ValueError("中心点解不合理")
                        except:
                            # 回退到梯度下降
                            gradient = np.zeros(n)
                            for i, idx in enumerate(np.where(valid_indices)[0]):
                                gradient += lambda_vec[idx] * A_norm[idx] * (margins[idx] / norms_AE[idx])
                            center += min_step * gradient
                    
                except Exception as e:
                    print(f"  中心点更新出错: {e}")
                    # 使用简单的梯度更新
                    delta = np.zeros(n)
                    for i in range(m):
                        if valid_indices[i] and lambda_vec[i] > 1e-10:
                            ai = A_norm[i]
                            ai_norm = norms_AE[i]
                            margin = margins[i]
                            delta += lambda_vec[i] * ai * (margin / ai_norm)
                    center += min_step * delta
                
                # 检查中心点是否在约束内
                violations_center = A_norm @ center - b_norm
                if np.any(violations_center > 0.01):
                    # 如果中心点违反约束，回退到上一个状态
                    center = best_center.copy()
                
                # 计算当前体积并检查变化
                try:
                    current_det_E = np.linalg.det(E)
                    vol_history.append(current_det_E)
                    center_history.append(center.copy())
                    
                    # 如果历史记录太长，删除旧的
                    if len(vol_history) > 10:
                        vol_history.pop(0)
                        center_history.pop(0)
                        
                    # 检测体积变化
                    vol_change = abs(current_det_E / prev_det_E - 1) if prev_det_E > 0 else float('inf')
                    
                    # 检测振荡 - 如果体积来回变化但没有实质改善
                    oscillating = False
                    if len(vol_history) >= 4:
                        recent_changes = np.diff(vol_history[-4:])
                        oscillating = (np.any(recent_changes > 0) and np.any(recent_changes < 0) and 
                                      abs(vol_history[-1] / vol_history[-4] - 1) < 0.01)
                    
                    # 如果体积变化异常或者振荡，采取措施
                    if (vol_change > vol_change_threshold or np.isnan(current_det_E) or 
                        np.isinf(current_det_E) or oscillating):
                        
                        if vol_change > vol_change_threshold:
                            print(f"  体积变化异常: {vol_change:.2f}倍，回退到更稳定状态")
                        elif oscillating:
                            print(f"  检测到迭代振荡，使用历史最佳状态")
                            
                        # 回退到最佳状态
                        E = best_E.copy()
                        center = best_center.copy()
                        current_det_E = best_vol
                        lambda_vec = np.ones(m) / m  # 重置lambda
                        lambda_vec[~mask] = 0.0
                        lambda_vec = lambda_vec / np.sum(lambda_vec)
                    else:
                        # 更新历史最佳
                        if current_det_E > best_vol:
                            best_E = E.copy()
                            best_center = center.copy()
                            best_vol = current_det_E
                            best_iter = iter_idx
                    
                    prev_det_E = current_det_E
                    
                except Exception as e:
                    print(f"  体积计算出错: {e}")
                    # 出错时使用最近的有效状态
                    E = best_E.copy()
                    center = best_center.copy()
                
                # 检查是否已经很久没有改进
                if iter_idx - best_iter > 20:
                    print(f"  迭代停滞，返回最佳解 (迭代{best_iter})")
                    return best_E, best_center
                
            except Exception as e:
                print(f"  迭代{iter_idx}出错: {e}")
                # 使用最佳状态继续
                E = best_E.copy()
                center = best_center.copy()
                prev_det_E = best_vol
        
        # 如果未收敛
        if iter_idx == max_iter - 1:
            print(f"  Affine Scaling方法未收敛，迭代次数: {max_iter}")
        
        # 使用最佳结果
        if best_vol > prev_det_E:
            print(f"  返回历史最佳解 (迭代{best_iter})")
            return best_E, best_center
        
        return E, center
    
    def _solve_khachiyan(self, A, b, center_init, tol=1e-6, max_iter=100):
        """
        使用Khachiyan迭代算法求解MVIE
        这是一个经典的椭球体拟合算法，稳定性好
        
        参数:
            A: 半空间约束的法向量矩阵
            b: 半空间约束的偏移量向量
            center_init: 初始中心点
            tol: 收敛容差
            max_iter: 最大迭代次数
            
        返回:
            (E, center): E是形状矩阵，center是椭球体中心
        """
        m, n = A.shape
        
        # 预处理: 平移约束使内点在原点
        # 这样可以先求解中心在原点的椭球，然后平移回去
        center = center_init.copy()
        b_shifted = b - A @ center
        
        # 初始化协方差矩阵
        P = np.eye(n)
        
        # 迭代求解
        for it in range(max_iter):
            # 计算每个约束的位置
            distances = np.zeros(m)
            for i in range(m):
                a_i = A[i].reshape(-1, 1)
                distances[i] = b_shifted[i] / np.sqrt(a_i.T @ P @ a_i)
            
            # 找到最近的约束
            min_dist = np.min(distances)
            
            # 如果最近约束足够远，结束迭代
            if min_dist > 1.0 - tol:
                break
                
            # 找到最近的约束索引
            j = np.argmin(distances)
            aj = A[j].reshape(-1, 1)
            
            # 计算更新因子
            sigma = (1.0 - min_dist) / (n + 1)
            
            # 更新P矩阵
            P_aj = P @ aj
            P = P - sigma * (P_aj @ P_aj.T) / (aj.T @ P_aj)
        
        # 从P矩阵恢复E (P = E^(-2))
        try:
            # 使用特征值分解确保数值稳定性
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)  # 确保非负特征值
            
            # 计算P^(-1/2) = E
            E = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # 检查E的条件数
            cond_E = np.max(eigvals) / np.min(eigvals)
            if cond_E > 1e10:
                print(f"  警告: E矩阵条件数较大 ({cond_E:.1e})，进行正则化")
                # 正则化特征值
                min_eig = np.min(eigvals)
                reg_eigvals = eigvals + 0.01 * min_eig
                E = eigvecs @ np.diag(1.0 / np.sqrt(reg_eigvals)) @ eigvecs.T
        except Exception as e:
            print(f"  计算E矩阵出错: {e}")
            return None, None
        
        if it == max_iter - 1:
            print(f"  Khachiyan算法未收敛，迭代{max_iter}次")
            
        return E, center
    
    def _solve_cvxpy(self, A, b, center_init):
        """
        使用CVXPY求解MVIE标准问题
        这是最准确但计算成本最高的方法
        
        参数:
            A: 半空间约束的法向量矩阵
            b: 半空间约束的偏移量向量
            center_init: 初始中心点
            
        返回:
            (E, center): E是形状矩阵，center是椭球体中心
        """
        try:
            import cvxpy as cp
        except ImportError:
            print("  未安装CVXPY库，无法使用此方法")
            return None, None
            
        try:
            # 定义SOCP问题变量
            E_var = cp.Variable((self.dim, self.dim), symmetric=True)
            center_var = cp.Variable(self.dim)
            
            # 初始化中心点
            center_var.value = center_init
            
            # 目标函数: 最大化log(det(E))
            objective = cp.Maximize(cp.log_det(E_var))
            
            # 约束条件: ||E^T a_i||_2 + a_i^T d <= b_i
            constraints = []
            for i in range(A.shape[0]):
                a_i = A[i]  # 半空间法向量
                b_i = b[i]  # 半空间偏移量
                
                # 添加SOCP约束
                constraints.append(cp.norm(E_var @ a_i) + a_i @ center_var <= b_i)
            
            # 添加E必须是正定矩阵的约束
            constraints.append(E_var >> 0)
            
            # 求解问题
            prob = cp.Problem(objective, constraints)
            
            # 尝试不同的求解器
            solvers = [cp.SCS, cp.ECOS]
            
            for solver in solvers:
                try:
                    prob.solve(solver=solver, verbose=False, eps=1e-5)
                    
                    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        # 获取最优解
                        E_opt = E_var.value
                        center_opt = center_var.value
                        return E_opt, center_opt
                    else:
                        print(f"  CVXPY求解状态: {prob.status}")
                except Exception as e:
                    print(f"  CVXPY求解器{solver}出错: {e}")
            
            # 所有求解器都失败
            return None, None
            
        except Exception as e:
            print(f"  CVXPY求解过程出错: {e}")
            return None, None
    
    def _is_valid_matrix(self, Q):
        """
        检查矩阵是否为有效的椭球体形状矩阵
        
        参数:
            Q: 椭球体形状矩阵
            
        返回:
            布尔值，表示矩阵是否有效
        """
        if Q is None:
            return False
            
        try:
            # 检查矩阵是否包含NaN或Inf
            if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
                return False
                
            # 检查矩阵是否对称
            if not np.allclose(Q, Q.T, rtol=1e-5, atol=1e-8):
                return False
                
            # 检查矩阵是否正定
            eigvals = np.linalg.eigvalsh(Q)
            if np.any(eigvals <= 0):
                return False
                
            # 检查条件数
            cond = np.max(eigvals) / np.min(eigvals)
            if cond > self.condition_threshold or np.isinf(cond):
                return False
                
            return True
        except Exception:
            return False
    
    def _min_vol_ellipsoid(self, points, tol=0.001):
        """
        计算包含给定点集的最小体积椭球体
        当多胞体的半空间表示不可用时，可以使用此方法
        
        参数:
            points: 点集，每行一个点
            tol: 收敛容差
            
        返回:
            (center, E): 椭球体中心和形状矩阵
        """
        points = np.asarray(points)
        N, d = points.shape
        
        # 初始化
        Q = np.eye(d)
        center = np.mean(points, axis=0)
        
        # 将点集中心化
        centered_points = points - center
        
        # 迭代优化
        prev_det = np.linalg.det(Q)
        for _ in range(100):
            # 计算每个点的马氏距离
            mahal_dist = np.zeros(N)
            for i in range(N):
                p = centered_points[i].reshape(-1, 1)
                mahal_dist[i] = p.T @ np.linalg.inv(Q) @ p
                
            # 找到最远的点
            furthest_idx = np.argmax(mahal_dist)
            furthest_dist = mahal_dist[furthest_idx]
            
            # 如果最远点在椭球体内，结束迭代
            if furthest_dist <= d + tol:
                break
                
            # 获取最远点
            p = centered_points[furthest_idx].reshape(-1, 1)
            
            # 计算更新步长
            beta = (furthest_dist - d) / ((d + 1) * furthest_dist)
            
            # 更新Q矩阵
            Q_inv = np.linalg.inv(Q)
            Q = (1 - beta) * Q + beta * (d + 1) * (p @ p.T)
            
            # 检查收敛性
            current_det = np.linalg.det(Q)
            if abs(current_det - prev_det) / prev_det < tol:
                break
            prev_det = current_det
            
        return center, Q 