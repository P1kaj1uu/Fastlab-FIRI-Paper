import numpy as np
import pickle
import os

class Obstacle:
    def __init__(self, shape, center, radius=None, size=None, height=None):
        self.shape = shape  # 'sphere', 'cylinder', 'cuboid'
        self.center = center
        self.radius = radius if radius is not None else 1.0  # 如果未指定半径，则默认为1.0
        self.size = size  # For cuboid: (length, width, height)
        self.height = height  # For cylinder: height

class ObstacleSet:
    def __init__(self):
        self.obstacle_list = []

    def add_obstacle(self, shape, center, radius=None, size=None, height=None):
        self.obstacle_list.append(Obstacle(shape, center, radius, size, height))

    def __len__(self):
        return len(self.obstacle_list)

    def __iter__(self):
        return iter(self.obstacle_list)

    def check_collision(self, new_obs):
        """检查新障碍物是否与已有障碍物碰撞"""
        for obs in self.obstacle_list:
            if obs.shape == 'sphere' and new_obs.shape == 'sphere':
                dist = np.linalg.norm(np.array(obs.center) - np.array(new_obs.center))
                if dist < obs.radius + new_obs.radius:
                    return True
            elif obs.shape == 'cylinder' and new_obs.shape == 'cylinder':
                dist = np.linalg.norm(np.array(obs.center) - np.array(new_obs.center))
                if dist < obs.radius + new_obs.radius:
                    return True
            elif obs.shape == 'cuboid' and new_obs.shape == 'cuboid':
                dist = np.linalg.norm(np.array(obs.center) - np.array(new_obs.center))
                if dist < np.linalg.norm(obs.size / 2 + new_obs.size / 2):
                    return True
        return False

def place_obstacles(space_boundary, n=5):
    obstacles = ObstacleSet()

    # 设置起点和终点
    start = np.array([1.0, 1.0, 1.0])
    goal = np.array([9.0, 9.0, 9.0])

    # 计算起点到终点的方向向量
    direction = goal - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm

    # 在起点和终点的连线上放置一个障碍物
    t = np.random.uniform(0.3, 0.7)  # 在30%到70%的位置之间
    obstacle_pos = start + t * (goal - start)
    obstacle_radius = np.random.uniform(0.8, 1.2)  # 适当调整半径
    obstacles.add_obstacle('sphere', obstacle_pos, radius=obstacle_radius)
    print(f"Placed obstacle on path at {obstacle_pos} with radius {obstacle_radius}")

    # 放置剩余的随机障碍物
    for _ in range(n - 1):
        shape = np.random.choice(['sphere', 'cylinder', 'cuboid'])
        if shape == 'sphere':
            while True:
                x = np.random.uniform(space_boundary[0][0], space_boundary[0][1])
                y = np.random.uniform(space_boundary[1][0], space_boundary[1][1])
                z = np.random.uniform(space_boundary[2][0], space_boundary[2][1])
                radius = np.random.uniform(0.5, 1.5)
                new_obs = Obstacle('sphere', np.array([x, y, z]), radius=radius)
                if not obstacles.check_collision(new_obs):
                    break
            obstacles.add_obstacle('sphere', np.array([x, y, z]), radius=radius)
            print(f"Placed sphere at {obstacles.obstacle_list[-1].center} with radius {radius}")

        elif shape == 'cylinder':
            while True:
                x = np.random.uniform(space_boundary[0][0], space_boundary[0][1])
                y = np.random.uniform(space_boundary[1][0], space_boundary[1][1])
                z = np.random.uniform(space_boundary[2][0], space_boundary[2][1])
                radius = np.random.uniform(0.5, 1.5)
                height = np.random.uniform(1.0, 2.0)
                new_obs = Obstacle('cylinder', np.array([x, y, z]), radius=radius, height=height)
                if not obstacles.check_collision(new_obs):
                    break
            obstacles.add_obstacle('cylinder', np.array([x, y, z]), radius=radius, height=height)
            print(f"Placed cylinder at {obstacles.obstacle_list[-1].center} with radius {radius} and height {height}")

        elif shape == 'cuboid':
            while True:
                x = np.random.uniform(space_boundary[0][0], space_boundary[0][1])
                y = np.random.uniform(space_boundary[1][0], space_boundary[1][1])
                z = np.random.uniform(space_boundary[2][0], space_boundary[2][1])
                size = np.random.uniform(0.5, 1.5, size=3)  # (length, width, height)
                new_obs = Obstacle('cuboid', np.array([x, y, z]), size=size)
                if not obstacles.check_collision(new_obs):
                    break
            obstacles.add_obstacle('cuboid', np.array([x, y, z]), size=size)
            print(f"Placed cuboid at {obstacles.obstacle_list[-1].center} with size {size}")

    return obstacles

def save_obstacles_to_file(obstacles):
    """保存障碍物信息到文件"""
    with open('temp/obstacles.pkl', 'wb') as f:
        pickle.dump(obstacles, f)
    print(f"Saved {len(obstacles.obstacle_list)} obstacles to file")
    
