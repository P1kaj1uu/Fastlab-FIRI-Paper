import numpy as np
import open3d as o3d

class ObstacleGenerator:
    """障碍物生成器，用于生成随机障碍物和策略性障碍物"""
    
    def __init__(self, space_size=(10, 10, 10)):
        """
        初始化障碍物生成器
        
        参数:
            space_size: 空间尺寸
        """
        self.space_size = space_size
        self.dimension = len(space_size)
    
    def generate_random_obstacle(self, inflation=1.0):
        """
        生成随机障碍物
        
        参数:
            inflation: 障碍物膨胀系数
            
        返回:
            (obstacle, inflated_obstacle): 原始障碍物和膨胀后的障碍物
        """
        # 随机生成障碍物中心
        center = [np.random.rand() * self.space_size[i] for i in range(self.dimension)]
        
        # 随机生成障碍物半径
        radius = 0.5 + np.random.rand() * 0.5
        
        # 创建原始障碍物
        obstacle = self._create_sphere(center, radius)
        
        # 创建膨胀后的障碍物
        inflated_obstacle = self._create_sphere(center, radius * inflation)
        
        return obstacle, inflated_obstacle
    
    def generate_strategic_obstacles(self, num_obstacles=30, start=None, goal=None):
        """
        生成策略性障碍物，确保路径必须绕行
        
        参数:
            num_obstacles: 障碍物数量
            start: 起点
            goal: 终点
            
        返回:
            (obstacles, inflated_obstacles): 原始障碍物列表和膨胀后的障碍物列表
        """
        obstacles = []
        inflated_obstacles = []
        obstacle_centers = []
        
        # 起点和终点
        if start is None:
            start = np.array([1.0, 1.0, 1.0])
        if goal is None:
            goal = np.array([9.0, 9.0, 9.0])
        
        # 在起点和终点之间放置策略性障碍物
        mid_point = (start + goal) / 2
        direction = goal - start
        
        # 在中点附近略微偏移放置障碍物
        offset = np.random.rand(self.dimension) * 0.4 - 0.2
        strategic_pos = mid_point + offset
        
        # 确保障碍物足够大，迫使路径绕行
        strategic_radius = np.linalg.norm(direction) * 0.1 + 0.5
        
        # 创建策略性障碍物
        strategic_obs = self._create_sphere(strategic_pos, strategic_radius)
        strategic_inf_obs = self._create_sphere(strategic_pos, strategic_radius * 1.2)
        
        obstacles.append(strategic_obs)
        inflated_obstacles.append(strategic_inf_obs)
        obstacle_centers.append(strategic_pos)
        
        print(f"策略性障碍物放置在 {strategic_pos}，确保路径必须绕行")
        
        # 添加随机障碍物
        for _ in range(num_obstacles - 1):
            # 随机生成位置，避免与起点终点太近
            valid_position = False
            max_attempts = 50
            position = None
            
            for attempt in range(max_attempts):
                position = [np.random.rand() * self.space_size[i] for i in range(self.dimension)]
                
                # 确保障碍物不太靠近起点和终点
                dist_to_start = np.linalg.norm(np.array(position) - start)
                dist_to_goal = np.linalg.norm(np.array(position) - goal)
                
                if dist_to_start > 2.0 and dist_to_goal > 2.0:
                    # 检查与现有障碍物的距离
                    too_close = False
                    for existing_center in obstacle_centers:
                        if np.linalg.norm(np.array(position) - existing_center) < 1.5:
                            too_close = True
                            break
                    
                    if not too_close:
                        valid_position = True
                        break
            
            if not valid_position:
                continue
                
            # 随机生成半径
            radius = 0.5 + np.random.rand() * 0.5
            
            # 创建障碍物
            obs = self._create_sphere(position, radius)
            inf_obs = self._create_sphere(position, radius * 1.2)
            
            obstacles.append(obs)
            inflated_obstacles.append(inf_obs)
            obstacle_centers.append(position)
            
            print(f"障碍物中心: {position}")
        
        return obstacles, inflated_obstacles
    
    def _create_sphere(self, center, radius):
        """
        创建球形障碍物
        
        参数:
            center: 中心坐标
            radius: 半径
            
        返回:
            障碍物对象
        """
        # 简单地返回一个包含中心和半径的字典
        return {
            'center': np.array(center),
            'radius': radius,
            'type': 'sphere'
        } 