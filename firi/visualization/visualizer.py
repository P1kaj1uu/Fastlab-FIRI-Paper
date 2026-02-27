import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

def visualize_firi_results(obstacles, safe_regions, path=None):
    """
    可视化FIRI算法的结果，包括障碍物、安全区域和路径
    
    参数:
        obstacles: 障碍物列表
        safe_regions: 安全区域列表，每个元素为(polytope, ellipsoid)
        path: 路径点序列
    """
    try:
        # 首先尝试使用Open3D进行可视化
        return _visualize_with_open3d(obstacles, safe_regions, path)
    except Exception as e:
        print(f"Open3D可视化失败: {e}，尝试使用Matplotlib备用方案")
        # 如果Open3D失败，使用Matplotlib作为备用
        return _visualize_with_matplotlib(obstacles, safe_regions, path)

def _visualize_with_open3d(obstacles, safe_regions, path=None):
    """使用Open3D进行可视化的内部实现"""
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800, window_name="FIRI 可视化")
    
    # 获取渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 5.0
    opt.light_on = True
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)
    
    # 添加障碍物
    obstacle_meshes = []
    for obs in obstacles:
        try:
            # 尝试不同的障碍物表示
            if hasattr(obs, 'vertices') and obs.vertices is not None:
                # 基于顶点的表示
                vertices = np.asarray(obs.vertices)
                if len(vertices) >= 4:
                    # 创建点云
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(vertices)
                    
                    # 尝试创建凸包
                    try:
                        hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
                        if hull.is_empty() or len(hull.vertices) < 3:
                            # 如果alpha shape失败，尝试凸包
                            hull = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                        
                        if not hull.is_empty() and len(hull.vertices) >= 3:
                            hull.compute_vertex_normals()
                            hull.paint_uniform_color([1, 0, 0])  # 红色
                            obstacle_meshes.append(hull)
                            vis.add_geometry(hull)
                        else:
                            # 备用：直接显示点云
                            pcd.paint_uniform_color([1, 0, 0])
                            vis.add_geometry(pcd)
                    except Exception as e:
                        print(f"凸包计算错误: {e}")
                        # 备用：直接显示点云
                        pcd.paint_uniform_color([1, 0, 0])
                        vis.add_geometry(pcd)
                else:
                    print("警告: 障碍物顶点数量不足")
            elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
                # 基于中心和半径的表示
                center = np.array(obs.center)
                radius = obs.radius
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center)
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([1, 0, 0])  # 红色
                obstacle_meshes.append(sphere)
                vis.add_geometry(sphere)
            elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                # 字典表示
                center = np.array(obs['center'])
                radius = obs['radius']
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center)
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([1, 0, 0])  # 红色
                obstacle_meshes.append(sphere)
                vis.add_geometry(sphere)
            else:
                print("警告: 未知的障碍物类型，跳过")
        except Exception as e:
            print(f"处理障碍物时出错: {e}")
    
    # 添加安全区域
    region_meshes = []
    for i, (polytope, ellipsoid) in enumerate(safe_regions):
        try:
            # 添加多胞体
            try:
                polytope_mesh = polytope.to_mesh()
                if polytope_mesh and not polytope_mesh.is_empty() and len(polytope_mesh.vertices) >= 3:
                    # 设置多胞体颜色，使用RGB不透明颜色
                    polytope_mesh.paint_uniform_color([0, 1, 0])  # 绿色
                    # 设置透明度
                    polytope_mesh.compute_vertex_normals()
                    region_meshes.append(polytope_mesh)
                    vis.add_geometry(polytope_mesh)
            except Exception as e:
                print(f"多胞体可视化错误: {e}")
                
            # 添加椭球体
            try:
                ellipsoid_mesh = ellipsoid.to_mesh()
                if ellipsoid_mesh and not ellipsoid_mesh.is_empty() and len(ellipsoid_mesh.vertices) >= 3:
                    # 设置椭球体颜色，使用RGB不透明颜色
                    ellipsoid_mesh.paint_uniform_color([0, 0, 1])  # 蓝色
                    ellipsoid_mesh.compute_vertex_normals()
                    region_meshes.append(ellipsoid_mesh)
                    vis.add_geometry(ellipsoid_mesh)
            except Exception as e:
                print(f"椭球体可视化错误: {e}")
        except Exception as e:
            print(f"处理安全区域 {i} 时出错: {e}")
    
    # 添加路径
    if path is not None and len(path) > 1:
        try:
            # 创建路径线段
            line_points = []
            line_indices = []
            for i in range(len(path) - 1):
                line_points.extend([path[i], path[i+1]])
                line_indices.append([2*i, 2*i+1])
                
            # 创建线集
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            
            # 设置颜色
            colors = [[1, 1, 0] for _ in range(len(line_indices))]  # 黄色
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            vis.add_geometry(line_set)
            
            # 添加路径点
            path_pcd = o3d.geometry.PointCloud()
            path_pcd.points = o3d.utility.Vector3dVector(path)
            path_pcd.paint_uniform_color([1, 1, 0])  # 黄色
            vis.add_geometry(path_pcd)
            
            # 特别标注起点和终点
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            start_sphere.translate(path[0])
            start_sphere.paint_uniform_color([0, 1, 0])  # 绿色起点
            vis.add_geometry(start_sphere)
            
            end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            end_sphere.translate(path[-1])
            end_sphere.paint_uniform_color([1, 0, 0])  # 红色终点
            vis.add_geometry(end_sphere)
        except Exception as e:
            print(f"处理路径时出错: {e}")
    
    # 尝试设置透明度
    try:
        # 检测Open3D版本并适配相应的API
        o3d_version = o3d.__version__.split('.')
        if int(o3d_version[0]) >= 0 and int(o3d_version[1]) >= 12:
            # 新版本API
            opt.mesh_show_wireframe = True
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.FLAT
            opt.mesh_show_back_face = True
        else:
            # 旧版本API尝试
            for attr_name in ['WIREFRAME', 'Wireframe', 'wireframe']:
                if hasattr(opt, attr_name):
                    setattr(opt, attr_name, True)
                    break
            
            if hasattr(opt, 'mesh_shade_option'):
                for attr_name in ['FLAT', 'Flat']:
                    if hasattr(o3d.visualization.MeshShadeOption, attr_name):
                        opt.mesh_shade_option = getattr(o3d.visualization.MeshShadeOption, attr_name)
                        break
        
        # 通用设置            
        if hasattr(opt, 'transparency'):
            # 设置全局透明度 (如果API支持)
            opt.transparency = 0.5
    except Exception as e:
        print(f"设置渲染选项时出错: {e}")
    
    # 创建一个截图并保存，以防窗口交互失败
    try:
        # 更新几何体和渲染一帧
        vis.update_geometry(coord_frame)
        vis.poll_events()
        vis.update_renderer()
        
        # 捕获截图
        timestamp = int(time.time())
        screenshot_path = f'temp/visualization_{timestamp}.png'
        vis.capture_screen_image(screenshot_path)
        print(f"可视化截图已保存至: {screenshot_path}")
    except Exception as e:
        print(f"截图保存失败: {e}")
    
    # 运行可视化窗口
    try:
        vis.run()
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"可视化窗口运行失败: {e}")
        return False

def _visualize_with_matplotlib(obstacles, safe_regions, path=None):
    """使用Matplotlib作为Open3D的备用可视化方法"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 添加障碍物
    for obs in obstacles:
        try:
            if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                center = np.array(obs.center)
                radius = obs.radius
            elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                center = np.array(obs['center'])
                radius = obs['radius']
            else:
                continue
                
            # 绘制球体
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color='red', alpha=0.2)
        except Exception as e:
            print(f"绘制障碍物时出错: {e}")
    
    # 添加安全区域 (简化处理为中心点)
    for i, (polytope, ellipsoid) in enumerate(safe_regions):
        try:
            # 多胞体的简化处理
            interior_point = polytope.get_interior_point()
            if interior_point is not None:
                ax.scatter(interior_point[0], interior_point[1], interior_point[2], c='green', marker='o', s=30)
                
            # 椭球体的中心点标记
            if hasattr(ellipsoid, 'center'):
                center = ellipsoid.center
                ax.scatter(center[0], center[1], center[2], c='blue', marker='^', s=50)
        except Exception as e:
            print(f"绘制安全区域时出错: {e}")
            
    # 添加路径
    if path is not None and len(path) > 1:
        try:
            # 路径线
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            z = [p[2] for p in path]
            ax.plot(x, y, z, 'yellow', linewidth=2)
            
            # 路径点
            ax.scatter(x, y, z, c='yellow', marker='.', s=20)
            
            # 起点和终点
            ax.scatter(path[0][0], path[0][1], path[0][2], c='green', marker='o', s=100, label='起点')
            ax.scatter(path[-1][0], path[-1][1], path[-1][2], c='red', marker='o', s=100, label='终点')
        except Exception as e:
            print(f"绘制路径时出错: {e}")

    # 坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FIRI路径规划可视化 (Matplotlib备用视图)')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 保存为图像文件
    timestamp = int(time.time())
    filename = f'temp/mpl_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # 显示图像
    plt.tight_layout()
    plt.show()
    
    print(f"Matplotlib可视化图像已保存至: {filename}")
    return True

def visualize_path_with_angles(path, obstacles=None, safety_margin=1.2):
    """
    可视化路径并标注角度变化较大的点
    
    参数:
        path: 路径点序列
        obstacles: 障碍物列表 (可选)
        safety_margin: 安全边界系数 (可选)
    """
    if path is None or len(path) < 3:
        print("路径点不足，无法分析角度")
        return
        
    # 计算角度
    angles = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        
        # 计算角度
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            angles.append(0)
            continue
            
        dot_product = np.dot(v1, v2)
        cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)
        
    # 打印角度信息
    print("\n角度分析:")
    for i, angle in enumerate(angles):
        print(f"  点 {i+1}: {angle:.2f}°")
        
    # 统计信息
    if angles:
        avg_angle = np.mean(angles)
        max_angle = np.max(angles)
        min_angle = np.min(angles)
        count_90plus = sum(1 for a in angles if a > 90)
        count_60plus = sum(1 for a in angles if a > 60)
        
        print("\n统计信息:")
        print(f"  平均角度: {avg_angle:.2f}°")
        print(f"  最大角度: {max_angle:.2f}°")
        print(f"  最小角度: {min_angle:.2f}°")
        print(f"  角度>90°的数量: {count_90plus}")
        print(f"  角度>60°的数量: {count_60plus}")
    
    # 可视化
    try:
        # 创建两个视图: 3D路径和角度变化曲线
        fig = plt.figure(figsize=(15, 10))
        
        # 3D路径视图
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 绘制路径
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = [p[2] for p in path]
        ax1.plot(x, y, z, 'b-', linewidth=2, label='路径')
        
        # 标记角度大于60度的点
        large_angle_indices = [i+1 for i, angle in enumerate(angles) if angle > 60]
        if large_angle_indices:
            large_x = [path[i][0] for i in large_angle_indices]
            large_y = [path[i][1] for i in large_angle_indices]
            large_z = [path[i][2] for i in large_angle_indices]
            ax1.scatter(large_x, large_y, large_z, c='orange', marker='*', s=150, label='大角度变化 (>60°)')
            
            # 添加标注
            for i, idx in enumerate(large_angle_indices):
                ax1.text(path[idx][0], path[idx][1], path[idx][2], f"{angles[idx-1]:.1f}°", 
                         color='black', fontsize=10, horizontalalignment='center')
        
        # 标记起点和终点
        ax1.scatter(x[0], y[0], z[0], c='green', marker='o', s=100, label='起点')
        ax1.scatter(x[-1], y[-1], z[-1], c='red', marker='o', s=100, label='终点')
        
        # 添加所有路径点标记
        ax1.scatter(x, y, z, c='blue', marker='.', s=30)
        
        # 如果有障碍物，绘制障碍物
        if obstacles is not None:
            for obs in obstacles:
                try:
                    # 获取障碍物中心和半径
                    if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                        center = np.array(obs.center)
                        radius = obs.radius * safety_margin
                    elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                        center = np.array(obs['center'])
                        radius = obs['radius'] * safety_margin
                    else:
                        continue
                        
                    # 绘制球体
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = center[0] + radius * np.cos(u) * np.sin(v)
                    y = center[1] + radius * np.sin(u) * np.sin(v)
                    z = center[2] + radius * np.cos(v)
                    ax1.plot_surface(x, y, z, color='red', alpha=0.2)
                except Exception as e:
                    print(f"绘制障碍物时出错: {e}")
                    continue
                    
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('路径与角度变化')
        ax1.legend()
        
        # 调整视角
        ax1.view_init(elev=30, azim=45)
        
        # 角度变化曲线
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, len(angles)+1), angles, 'b-o', linewidth=2)
        
        # 添加参考线
        ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60° 阈值')
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90° 阈值')
        
        # 标记大角度点
        for i, angle in enumerate(angles):
            if angle > 60:
                ax2.annotate(f'{angle:.1f}°', xy=(i+1, angle), xytext=(0, 5), 
                            textcoords='offset points', ha='center', fontsize=10)
        
        ax2.set_xlabel('路径点索引')
        ax2.set_ylabel('角度 (度)')
        ax2.set_title('路径角度变化曲线')
        ax2.grid(True)
        ax2.legend()
        
        # 保存图像
        timestamp = int(time.time())
        filename = f'temp/path_angle_analysis_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"角度分析可视化已保存至: {filename}")
        return True
        
    except Exception as e:
        print(f"角度可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_path_only(obstacles, path, start, goal):
    """
    简化版可视化，只显示路径和障碍物
    
    参数:
        obstacles: 障碍物列表
        path: 路径点序列
        start: 起点
        goal: 终点
    """
    try:
        # 首先尝试使用Open3D
        return _visualize_path_only_open3d(obstacles, path, start, goal)
    except Exception as e:
        print(f"Open3D路径可视化失败: {e}，尝试使用Matplotlib备用方案")
        # 如果Open3D失败，使用Matplotlib作为备用
        return _visualize_path_only_matplotlib(obstacles, path, start, goal)

def _visualize_path_only_open3d(obstacles, path, start, goal):
    """使用Open3D进行简化路径可视化的内部实现"""
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800, window_name="路径规划结果")
    
    # 获取渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 5.0
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)
    
    # 添加障碍物
    for obs in obstacles:
        try:
            # 尝试不同的障碍物表示
            if hasattr(obs, 'vertices') and obs.vertices is not None:
                # 基于顶点的表示
                vertices = np.asarray(obs.vertices)
                if len(vertices) >= 4:
                    # 创建点云
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(vertices)
                    
                    # 尝试创建凸包
                    try:
                        hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
                        if hull.is_empty() or len(hull.vertices) < 3:
                            # 如果alpha shape失败，尝试凸包
                            hull = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                        
                        if not hull.is_empty() and len(hull.vertices) >= 3:
                            hull.compute_vertex_normals()
                            hull.paint_uniform_color([1, 0, 0])  # 红色
                            vis.add_geometry(hull)
                        else:
                            # 备用：直接显示点云
                            pcd.paint_uniform_color([1, 0, 0])
                            vis.add_geometry(pcd)
                    except:
                        # 备用：直接显示点云
                        pcd.paint_uniform_color([1, 0, 0])
                        vis.add_geometry(pcd)
            elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
                # 基于中心和半径的表示
                center = np.array(obs.center)
                radius = obs.radius
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center)
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([1, 0, 0])  # 红色
                vis.add_geometry(sphere)
            elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                # 字典表示
                center = np.array(obs['center'])
                radius = obs['radius']
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center)
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([1, 0, 0])  # 红色
                vis.add_geometry(sphere)
        except:
            continue
    
    # 添加路径
    if path is not None and len(path) > 1:
        # 创建路径线段
        line_points = []
        line_indices = []
        for i in range(len(path) - 1):
            line_points.extend([path[i], path[i+1]])
            line_indices.append([2*i, 2*i+1])
            
        # 创建线集
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        
        # 设置颜色
        colors = [[1, 1, 0] for _ in range(len(line_indices))]  # 黄色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        vis.add_geometry(line_set)
        
        # 添加路径点
        path_pcd = o3d.geometry.PointCloud()
        path_pcd.points = o3d.utility.Vector3dVector(path)
        path_pcd.paint_uniform_color([1, 1, 0])  # 黄色
        vis.add_geometry(path_pcd)
    
    # 添加起点和终点
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0, 1, 0])  # 绿色起点
    vis.add_geometry(start_sphere)
    
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    end_sphere.translate(goal)
    end_sphere.paint_uniform_color([1, 0, 0])  # 红色终点
    vis.add_geometry(end_sphere)
    
    # 创建一个截图并保存，以防窗口交互失败
    try:
        # 更新几何体和渲染一帧
        vis.poll_events()
        vis.update_renderer()
        
        # 捕获截图
        timestamp = int(time.time())
        screenshot_path = f'temp/path_only_visualization_{timestamp}.png'
        vis.capture_screen_image(screenshot_path)
        print(f"路径可视化截图已保存至: {screenshot_path}")
    except Exception as e:
        print(f"截图保存失败: {e}")
    
    # 运行可视化窗口
    vis.run()
    vis.destroy_window()
    return True

def _visualize_path_only_matplotlib(obstacles, path, start, goal):
    """使用Matplotlib作为Open3D的备用路径可视化方法"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 添加障碍物
    for obs in obstacles:
        try:
            if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                center = np.array(obs.center)
                radius = obs.radius
            elif isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                center = np.array(obs['center'])
                radius = obs['radius']
            else:
                continue
                
            # 绘制球体
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color='red', alpha=0.2)
        except Exception as e:
            print(f"绘制障碍物时出错: {e}")
    
    # 添加路径
    if path is not None and len(path) > 1:
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = [p[2] for p in path]
        ax.plot(x, y, z, 'yellow', linewidth=2)
        ax.scatter(x, y, z, c='yellow', marker='.', s=20)
    
    # 添加起点和终点
    ax.scatter(start[0], start[1], start[2], c='green', marker='o', s=100, label='起点')
    ax.scatter(goal[0], goal[1], goal[2], c='red', marker='o', s=100, label='终点')
    
    # 坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('路径规划可视化 (Matplotlib备用视图)')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 保存为图像文件
    timestamp = int(time.time())
    filename = f'temp/mpl_path_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # 显示图像
    plt.tight_layout()
    plt.show()
    
    print(f"Matplotlib路径可视化图像已保存至: {filename}")
    return True 