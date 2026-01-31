import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev

def visualize_results(path, obstacles, space_bounds):
    """
    使用matplotlib可视化路径规划结果

    参数:
        path: 规划路径点
        obstacles: 障碍物集合
        space_bounds: 空间边界 [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    """
    try:
        # 设置matplotlib使用英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 使用B样条函数平滑路径
        tck, u = splprep(path.T, u=None, s=0.0, k=3)  # s=0 表示不平滑，k=3 表示三次样条
        u_new = np.linspace(u.min(), u.max(), 1000)  # 生成新的参数值
        path_smooth = np.array(splev(u_new, tck)).T  # 计算平滑后的路径点

        # 绘制平滑后的路径
        ax.plot(path_smooth[:, 0], path_smooth[:, 1], path_smooth[:, 2], 'b-', linewidth=2, label='Smoothed Path')

        # 标记起点和终点
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='green', s=100, marker='o', label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='red', s=100, marker='o', label='End')

        # 绘制障碍物
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)

        for obs in obstacles:
            center = np.array(obs.center)
            print(f"Rendering obstacle with shape: {obs.shape}")  # 调试输出障碍物的形状

            if obs.shape == 'sphere':
                radius = obs.radius
                # 使用 parametric 方程绘制球体
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='r', alpha=0.3)

            elif obs.shape == 'cylinder':
                radius = obs.radius
                height = obs.height
                # 绘制圆柱体，使用圆柱体的顶端和底端圆形网格
                z = np.linspace(center[2] - height / 2, center[2] + height / 2, 20)
                theta = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = radius * np.cos(theta_grid) + center[0]
                y_grid = radius * np.sin(theta_grid) + center[1]
                ax.plot_surface(x_grid, y_grid, z_grid, color='b', alpha=0.3)

            elif obs.shape == 'cuboid':
                # 使用 Poly3DCollection 绘制长方体
                length, width, height = obs.size
                # 定义长方体的8个顶点
                vertices = np.array([
                    [center[0] - length / 2, center[1] - width / 2, center[2] - height / 2],
                    [center[0] + length / 2, center[1] - width / 2, center[2] - height / 2],
                    [center[0] + length / 2, center[1] + width / 2, center[2] - height / 2],
                    [center[0] - length / 2, center[1] + width / 2, center[2] - height / 2],
                    [center[0] - length / 2, center[1] - width / 2, center[2] + height / 2],
                    [center[0] + length / 2, center[1] - width / 2, center[2] + height / 2],
                    [center[0] + length / 2, center[1] + width / 2, center[2] + height / 2],
                    [center[0] - length / 2, center[1] + width / 2, center[2] + height / 2]
                ])

                # 定义长方体的面（每个面由四个顶点组成）
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                    [vertices[3], vertices[0], vertices[4], vertices[7]]   # 左面
                ]
                
                # 使用 Poly3DCollection 绘制这些面
                ax.add_collection3d(Poly3DCollection(faces, facecolors='g', linewidths=1, edgecolors='r', alpha=0.3))

        # 设置图形属性 (使用英文标签)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Path Planning Visualization')

        # 设置轴范围
        ax.set_xlim(space_bounds[0][0], space_bounds[1][0])
        ax.set_ylim(space_bounds[0][1], space_bounds[1][1])
        ax.set_zlim(space_bounds[0][2], space_bounds[1][2])

        # 添加图例
        ax.legend()

        # 保存图像
        plt.savefig('temp/path_visualization.png', dpi=300, bbox_inches='tight')

        print("Visualization saved to temp/path_visualization.png")
    except Exception as e:
        print(f"Visualization error: {str(e)}")

def visualize_with_open3d(path, obstacles, start_point, goal_point, inflated_obstacles=None, safety_margin=1.2):
    """
    使用Open3D可视化路径规划结果，包括障碍物膨胀和轨迹

    参数:
        path: 规划的路径点列表
        obstacles: 原始障碍物集合
        start_point: 起点坐标
        goal_point: 终点坐标
        inflated_obstacles: 膨胀后的障碍物集合(可选)
        safety_margin: 障碍物膨胀系数(如果没有提供膨胀障碍物)
    """
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800, window_name="Path Planning with Open3D")

    # 获取渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 8.0

    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)

    # 添加原始障碍物（球体、圆柱体、长方体）
    for obs in obstacles.obstacle_list:
        center = np.array(obs.center)

        if obs.shape == 'sphere':
            radius = obs.radius
            # 创建球体
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(center)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            vis.add_geometry(sphere)

        elif obs.shape == 'cylinder':
            radius = obs.radius
            height = obs.height
            # 创建圆柱体
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height)
            cylinder.translate(center)
            cylinder.compute_vertex_normals()
            cylinder.paint_uniform_color([0, 1, 0])  # 绿色
            vis.add_geometry(cylinder)

        elif obs.shape == 'cuboid':
            length, width, height = obs.size
            # 创建长方体
            box = o3d.geometry.TriangleMesh.create_box(width, length, height)
            box.translate(center - np.array([width/2, length/2, height/2]))  # 确保中心点对齐
            box.compute_vertex_normals()
            box.paint_uniform_color([0, 0, 1])  # 蓝色
            vis.add_geometry(box)

    # 添加膨胀障碍物（半透明颜色）
    if inflated_obstacles is None and safety_margin > 1.0:
        for obs in obstacles.obstacle_list:
            center = np.array(obs.center)
            if obs.shape == 'sphere':
                inflated_radius = obs.radius * safety_margin
                inflated_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=inflated_radius)
                inflated_sphere.translate(center)
                inflated_sphere.compute_vertex_normals()
                inflated_sphere.paint_uniform_color([1, 1, 0])  # 黄色
                vis.add_geometry(inflated_sphere)
            elif obs.shape == 'cylinder':
                inflated_radius = obs.radius * safety_margin
                inflated_height = obs.height * safety_margin
                inflated_cylinder = o3d.geometry.TriangleMesh.create_cylinder(inflated_radius, inflated_height)
                inflated_cylinder.translate(center)
                inflated_cylinder.compute_vertex_normals()
                inflated_cylinder.paint_uniform_color([0, 1, 1])  # 青色
                vis.add_geometry(inflated_cylinder)
            elif obs.shape == 'cuboid':
                inflated_size = obs.size * safety_margin
                inflated_box = o3d.geometry.TriangleMesh.create_box(width=inflated_size[0], height=inflated_size[1], depth=inflated_size[2])
                inflated_box.translate(center - inflated_size / 2)
                inflated_box.compute_vertex_normals()
                inflated_box.paint_uniform_color([1, 0, 1])  # 紫色
                vis.add_geometry(inflated_box)

    # 使用B样条函数平滑路径
    if path is not None and len(path) > 1:
        # 转换路径点为numpy数组
        path_array = np.array(path)
        print("路径点形状:", path_array.shape)
        print("路径点内容:", path_array)

        if len(path_array) < 4:
            print("路径点数量不足，无法生成B样条曲线。")
            return False

        try:
            # 使用B样条函数平滑路径
            tck, u = splprep(path_array.T, u=None, s=0.0, k=3)  # s=0 表示不平滑
            u_new = np.linspace(u.min(), u.max(), 1000)  # 生成新的参数值
            path_smooth = np.array(splev(u_new, tck)).T  # 计算平滑后的路径点

            # 创建平滑路径的LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(path_smooth)
            line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path_smooth) - 1)])
            line_set.colors = o3d.utility.Vector3dVector([[1, 0.5, 0.5] for _ in range(len(path_smooth) - 1)])
            vis.add_geometry(line_set)

            # 创建路径点云（黄色点）
            path_points = o3d.geometry.PointCloud()
            path_points.points = o3d.utility.Vector3dVector(path_smooth)
            path_points.paint_uniform_color([1, 0.5, 0.5])  # 黄色
            vis.add_geometry(path_points)

        except Exception as e:
            print(f"B样条曲线生成失败: {e}")
            return False

    # 添加起点（绿色球体）
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.translate(start_point)
    start_sphere.paint_uniform_color([0, 1, 0])  # 绿色
    vis.add_geometry(start_sphere)

    # 添加终点（蓝色球体）
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    end_sphere.translate(goal_point)
    end_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    vis.add_geometry(end_sphere)

    # 设置视图
    vis.get_view_control().set_zoom(0.7)

    # 运行可视化器
    vis.run()
    vis.destroy_window()

    return True
