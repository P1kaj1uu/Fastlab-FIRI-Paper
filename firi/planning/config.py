import numpy as np

class FIRIConfig:
    """FIRI算法的配置类，提供参数管理和自适应调整"""
    
    def __init__(self, space_size=(10, 10, 10)):
        """
        初始化配置
        
        参数:
            space_size: 空间尺寸，默认为(10, 10, 10)
        """
        # 基本参数
        self.space_size = space_size
        self.dimension = len(space_size)
        
        # 安全区域计算参数
        self.safety_iterations = 2        # FIRI迭代次数
        self.volume_threshold = 0.01      # 收敛阈值
        self.safety_margin = 0.5          # 安全边距
        self.use_adaptive_margin = True   # 是否使用自适应边距
        
        # 路径规划参数
        self.collision_threshold = 0.02   # 碰撞判定阈值
        self.path_samples = 10            # 路径碰撞检测采样点数
        self.max_replanning_attempts = 3  # 最大重规划次数
        self.smoothing_iterations = 2     # 路径平滑迭代次数
        self.smoothing_window = 3         # 路径平滑窗口大小
        
        # MVIE计算参数
        self.mvie_max_iterations = 100    # MVIE最大迭代次数
        self.mvie_tolerance = 1e-6        # MVIE收敛容差
        
        # 计时统计
        self.timing_stats = {}
    
    def update_adaptive_params(self, obstacle_count=None, path_length=None, 
                               complexity_estimate=None):
        """
        根据环境复杂度自适应更新参数
        
        参数:
            obstacle_count: 障碍物数量
            path_length: 路径长度
            complexity_estimate: 环境复杂度估计值(0~1)
        """
        # 如果提供了障碍物数量，调整安全参数
        if obstacle_count is not None:
            # 障碍物越多，安全边距越大
            if obstacle_count > 20:
                self.safety_margin = min(0.8, self.safety_margin * 1.2)
                self.safety_iterations = min(4, self.safety_iterations + 1)
            elif obstacle_count < 5:
                self.safety_margin = max(0.3, self.safety_margin * 0.8)
                self.safety_iterations = max(1, self.safety_iterations - 1)
        
        # 如果提供了路径长度，调整路径参数
        if path_length is not None:
            # 路径越长，采样点越多
            base_samples = 10
            self.path_samples = max(base_samples, int(path_length / 2))
            
            # 路径越长，平滑窗口越大
            self.smoothing_window = min(5, max(3, int(path_length / 3)))
        
        # 如果提供了复杂度估计，综合调整
        if complexity_estimate is not None:
            # 复杂度越高，安全边距越大，迭代次数越多
            self.safety_margin = 0.3 + complexity_estimate * 0.7
            self.safety_iterations = max(1, min(4, int(2 + complexity_estimate * 2)))
            self.volume_threshold = max(0.005, 0.02 - complexity_estimate * 0.015)
    
    def get_param(self, name):
        """获取指定参数值"""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"未知参数: {name}")
    
    def record_timing(self, operation, time_ms):
        """记录操作耗时"""
        if operation not in self.timing_stats:
            self.timing_stats[operation] = []
        self.timing_stats[operation].append(time_ms)
    
    def get_timing_summary(self):
        """获取计时统计摘要"""
        summary = {}
        for op, times in self.timing_stats.items():
            summary[op] = {
                'min': min(times),
                'max': max(times),
                'avg': sum(times) / len(times),
                'count': len(times)
            }
        return summary 