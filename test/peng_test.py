import open3d as o3d
import numpy as np
from scipy.spatial import KDTree


class ObstacleGenerator:
    def __init__(self, space_size=(10, 10, 10)):
        self.space_size = space_size
        self.obstacles = []
        self.inflated_obstacles = []

    def generate_random_obstacle(self, inflation=0.5):
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


class InflationRRTStar:
    class Node:
        def __init__(self, position):
            self.position = np.array(position)
            self.parent = None
            self.cost = 0.0

    def __init__(self, obstacles, space_size):
        self.obstacles = obstacles
        self.space_size = np.array(space_size)
        self.nodes = []

    def sample_free(self):
        return np.random.rand(3) * self.space_size

    def find_nearest(self, target_point):
        if len(self.nodes) == 0:
            return None
        if len(self.nodes) > 500:
            kdtree = KDTree([node.position for node in self.nodes])
            _, idx = kdtree.query(target_point)
            return self.nodes[idx]
        else:
            min_dist = float('inf')
            nearest_node = None
            for node in self.nodes:
                dist = np.linalg.norm(node.position - target_point)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            return nearest_node

    def steer(self, from_point, to_point, step_size=0.5):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance * step_size
        return from_point + direction

    def collision_check(self, point):
        for obstacle in self.obstacles:
            vertices = np.asarray(obstacle.vertices)
            if self.point_in_obstacle(point, vertices, obstacle):
                return True
        return False

    def point_in_obstacle(self, point, vertices, obstacle):
        if "Sphere" in str(type(obstacle)):
            center = np.mean(vertices, axis=0)
            radius = np.linalg.norm(vertices[0] - center)
            return np.linalg.norm(point - center) < radius

        elif "Cylinder" in str(type(obstacle)):
            base = vertices[0]
            top = vertices[-1]
            axis = top - base
            height = np.linalg.norm(axis)
            axis = axis / height
            vec = point - base
            proj = np.dot(vec, axis)
            radial_dist = np.linalg.norm(vec - proj * axis)
            radius = np.linalg.norm(vertices[1] - base)
            return 0 <= proj <= height and radial_dist < radius

        elif "Box" in str(type(obstacle)):
            min_coord = np.min(vertices, axis=0)
            max_coord = np.max(vertices, axis=0)
            return np.all(point > min_coord) and np.all(point < max_coord)

        return False

    def plan(self, start, goal, max_iter=2000, step_size=0.8):
        self.nodes = [self.Node(start)]
        goal_node = self.Node(goal)

        for _ in range(max_iter):
            rand_point = self.sample_free()
            nearest_node = self.find_nearest(rand_point)

            if nearest_node is None:
                continue

            new_point = self.steer(nearest_node.position, rand_point, step_size)

            if not self.collision_check(new_point):
                new_node = self.Node(new_point)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_point - nearest_node.position)
                self.nodes.append(new_node)

                # 目标连接检查
                if np.linalg.norm(new_point - goal) < step_size:
                    if not self.collision_check(goal):
                        goal_node.parent = new_node
                        self.nodes.append(goal_node)
                        return self.generate_path(goal_node)

        return None

    def generate_path(self, node):
        path = []
        while node is not None:
            path.append(node.position)
            node = node.parent
        return np.array(path[::-1])


def visualize_environment(obstacles, start, goal, path=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    # 原始障碍物（红色）
    for obs in obstacles:
        obs.paint_uniform_color([0.7, 0.1, 0.1])
        vis.add_geometry(obs)

    # 起点（绿色）
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0.1, 0.7, 0.1])
    vis.add_geometry(start_sphere)

    # 终点（蓝色）
    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    goal_sphere.translate(goal)
    goal_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    vis.add_geometry(goal_sphere)

    # 路径（橙色）
    if path is not None and len(path) > 1:
        points = o3d.utility.Vector3dVector(path)
        lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path) - 1)])
        line_set = o3d.geometry.LineSet(points=points, lines=lines)
        line_set.paint_uniform_color([0.9, 0.5, 0])
        vis.add_geometry(line_set)

    vis.get_render_option().mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # 生成障碍物
    generator = ObstacleGenerator(space_size=(10, 10, 10))
    obstacles, inflated_obs = [], []
    for _ in range(15):
        obs, inf_obs = generator.generate_random_obstacle(inflation=0.3)
        obstacles.append(obs)
        inflated_obs.append(inf_obs)

    # 设置起点终点
    start = np.array([1, 1, 1])
    goal = np.array([9, 9, 9])

    # 路径规划
    planner = InflationRRTStar(inflated_obs, space_size=(10, 10, 10))
    path = planner.plan(start, goal, max_iter=3000)

    # 可视化
    visualize_environment(obstacles, start, goal, path)
