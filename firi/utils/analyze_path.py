import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from firi.geometry.convex_polytope import ConvexPolytope
from firi.geometry.ellipsoid import Ellipsoid

def load_path():
    """从文件加载路径数据"""
    try:
        with open('temp/final_path.pkl', 'rb') as file:
            path = pickle.load(file)
            print(f"轨迹点数量: {len(path)}")
            print(f"起点: {path[0]}")
            print(f"终点: {path[-1]}")
            return path
    except Exception as e:
        print(f"加载路径数据时出错: {e}")
        return None

def load_obstacles():
    """从文件加载障碍物数据"""
    try:
        # 尝试从文件加载障碍物数据
        try:
            with open('temp/obstacles.pkl', 'rb') as file:
                obstacles = pickle.load(file)
                print(f"已加载 {len(obstacles)} 个障碍物数据")
                return obstacles
        except FileNotFoundError:
            print("未找到障碍物信息文件，使用默认障碍物列表")
            # 构造默认障碍物（如果没有找到文件）
            obstacles = [
                {'center': np.array([5.0, 5.0, 5.0]), 'radius': 2.0},
                {'center': np.array([3.0, 3.0, 3.0]), 'radius': 1.5},
                {'center': np.array([7.0, 7.0, 7.0]), 'radius': 1.5}
            ]
            return obstacles
    except Exception as e:
        print(f"加载障碍物数据时出错: {e}")
        return []

def load_safe_regions():
    """从文件加载安全区域数据"""
    safe_regions = []
    try:
        i = 0
        while True:
            filename = f'temp/safe_region_{i}.pkl'
            if not os.path.exists(filename):
                break
            with open(filename, 'rb') as file:
                safe_region = pickle.load(file)
                safe_regions.append(safe_region)
            i += 1
        print(f"已加载 {len(safe_regions)} 个安全区域")
        return safe_regions
    except Exception as e:
        print(f"加载安全区域数据时出错: {e}")
        return []

def analyze_path_smoothness(path):
    """分析路径平滑度"""
    if len(path) < 3:
        return 0.0
    
    angles = []
    for i in range(1, len(path) - 1):
        prev_point = np.array(path[i-1])
        current_point = np.array(path[i])
        next_point = np.array(path[i+1])
        
        # 计算两个向量
        v1 = current_point - prev_point
        v2 = next_point - current_point
        
        # 归一化向量
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # 计算夹角（弧度）
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            
            # 转换为角度
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
    
    if not angles:
        return 0.0
    
    # 返回平均角度（越小越平滑）
    return sum(angles) / len(angles)

def calculate_path_length(path):
    """计算路径总长度"""
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        total_length += np.linalg.norm(p2 - p1)
    
    return total_length

def check_collisions(path, obstacles):
    """检查路径与障碍物的碰撞"""
    if not obstacles or len(path) < 2:
        return []
    
    collision_points = []
    
    # 对路径进行稠密采样以提高检测精度
    dense_path = []
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        segment_length = np.linalg.norm(p2 - p1)
        num_samples = max(10, int(segment_length * 5))  # 每单位长度采样5个点
        
        for j in range(num_samples + 1):
            t = j / num_samples
            point = p1 + t * (p2 - p1)
            dense_path.append(point)
    
    # 检查碰撞
    for point in dense_path:
        for obs in obstacles:
            if check_point_in_obstacle(point, obs):
                collision_points.append(point)
                break  # 一个点只记录一次碰撞
    
    return collision_points

def check_path_collision(path, obstacles):
    """检查路径是否与障碍物发生碰撞"""
    if not obstacles:
        return {'collisions': False, 'collision_count': 0, 'collision_points': []}
    
    # 对路径进行稠密采样以提高检测精度
    dense_path = []
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        segment_length = np.linalg.norm(p2 - p1)
        num_samples = max(10, int(segment_length * 5))  # 每单位长度采样5个点
        
        for j in range(num_samples + 1):
            t = j / num_samples
            point = p1 + t * (p2 - p1)
            dense_path.append(point)
    
    # 检查碰撞
    collision_points = []
    
    for point in dense_path:
        for obs in obstacles:
            if check_point_in_obstacle(point, obs):
                collision_info = {
                    'point': point,
                    'obstacle_center': obs['center'],
                    'distance': np.linalg.norm(point - obs['center']),
                    'obstacle_radius': obs.get('radius', 1.0)
                }
                collision_points.append(collision_info)
                break  # 一个点只记录一次碰撞
    
    collision_count = len(collision_points)
    
    if collision_count > 0:
        print(f"检测到 {collision_count} 个碰撞点!")
        # 输出前5个碰撞的详细信息
        for i, collision in enumerate(collision_points[:5]):
            print(f"碰撞 {i+1}:")
            print(f"  位置: {collision['point']}")
            print(f"  障碍物位置: {collision['obstacle_center']}")
            print(f"  距离: {collision['distance']} (障碍物半径: {collision['obstacle_radius']})")
    
    return {
        'collisions': collision_count > 0,
        'collision_count': collision_count,
        'collision_points': collision_points
    }

def check_point_in_obstacle(point, obstacle):
    """检查点是否在障碍物内"""
    try:
        # 如果障碍物有contains方法（比如凸多面体或椭球体类）
        if hasattr(obstacle, 'contains'):
            return obstacle.contains(point)
            
        # 如果是简单的球体表示
        elif 'center' in obstacle and 'radius' in obstacle:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            distance = np.linalg.norm(point - center)
            return distance <= radius
        else:
            print(f"未知障碍物类型: {type(obstacle)}")
            return False
    except Exception as e:
        print(f"检查点碰撞时出错: {e}")
        return False

def visualize_path_with_obstacles(path, obstacles, collision_points=None):
    """将路径与障碍物可视化，并保存为图像"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制路径
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'b-', linewidth=2, label='Path')
        
        # 标记起点和终点
        ax.scatter(path_array[0, 0], path_array[0, 1], path_array[0, 2], c='g', s=100, label='Start')
        ax.scatter(path_array[-1, 0], path_array[-1, 1], path_array[-1, 2], c='r', s=100, label='End')
        
        # 绘制障碍物
        for i, obs in enumerate(obstacles):
            if 'center' in obs and 'radius' in obs:
                # 绘制球体
                center = obs['center']
                radius = obs['radius']
                
                # 创建球体的网格表示
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = center[0] + radius * np.cos(u) * np.sin(v)
                y = center[1] + radius * np.sin(u) * np.sin(v)
                z = center[2] + radius * np.cos(v)
                
                # 绘制半透明的球体
                ax.plot_surface(x, y, z, color='gray', alpha=0.3)
            else:
                # 对于其他类型的障碍物，绘制中心点
                if hasattr(obs, 'center'):
                    center = obs.center
                elif 'center' in obs:
                    center = obs['center']
                else:
                    continue
                    
                ax.scatter(center[0], center[1], center[2], c='orange', s=50, alpha=0.7)
        
        # 绘制碰撞点
        if collision_points:
            collision_positions = np.array([c['point'] for c in collision_points])
            if len(collision_positions) > 0:
                ax.scatter(
                    collision_positions[:, 0], 
                    collision_positions[:, 1], 
                    collision_positions[:, 2], 
                    c='red', s=30, alpha=0.7, label='Collisions'
                )
        
        # 设置轴标签和标题
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Path Planning Visualization')
        
        # 添加图例
        ax.legend()
        
        # 设置轴范围
        min_bound = np.min(path_array, axis=0) - 1
        max_bound = np.max(path_array, axis=0) + 1
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])
        
        # 保存图像
        timestamp = int(time.time())
        filename = f'temp/path_visualization_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图像已保存至: {filename}")
    except Exception as e:
        print(f"路径可视化失败: {e}")
        import traceback
        traceback.print_exc()

def save_analysis_results(results, filename='temp/path_analysis.txt'):
    """保存分析结果到文件"""
    with open(filename, 'w') as f:
        f.write("路径分析结果\n")
        f.write("=============\n\n")
        
        # 写入路径基本信息
        f.write(f"路径点数量: {results['path_points']}\n")
        f.write(f"路径长度: {results['path_length']:.2f}\n\n")
        
        # 写入平滑度信息
        f.write("平滑度分析:\n")
        f.write(f"  平均角度: {results['smoothness']['avg_angle']:.2f}°\n")
        f.write(f"  最大角度: {results['smoothness']['max_angle']:.2f}°\n")
        f.write(f"  急转弯数量 (>90°): {results['smoothness']['angles_over_90']}\n\n")
        
        # 写入碰撞信息
        f.write("碰撞分析:\n")
        f.write(f"  是否有碰撞: {'是' if results['collision']['collisions'] else '否'}\n")
        if results['collision']['collisions']:
            f.write(f"  碰撞点数量: {results['collision']['collision_count']}\n\n")
        
        # 写入安全区域信息
        if 'safety' in results:
            f.write("安全区域分析:\n")
            f.write(f"  路径是否在安全区域内: {'是' if results['safety']['safe'] else '否'}\n")
            f.write(f"  安全分数: {results['safety']['safety_score']:.2f}\n")
        
        f.write("\n完成时间: " + time.ctime())
    
    print(f"分析结果已保存至: {filename}")

def check_path_safety(path, safe_regions):
    """检查路径是否在安全区域内"""
    try:
        # 对路径进行稠密采样
        dense_path = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            segment_length = np.linalg.norm(p2 - p1)
            num_samples = max(10, int(segment_length * 5))  # 每单位长度采样5个点
            
            for j in range(num_samples):
                t = j / num_samples
                point = p1 + t * (p2 - p1)
                dense_path.append(point)
        
        dense_path.append(np.array(path[-1]))  # 添加终点
        
        # 检查每个点是否在安全区域内
        points_in_safe_region = 0
        
        for point in dense_path:
            in_any_region = False
            for region in safe_regions:
                # 检查是否为ConvexPolytope类型或Ellipsoid类型
                if isinstance(region, (ConvexPolytope, Ellipsoid)):
                    if region.contains(point):
                        in_any_region = True
                        break
                elif isinstance(region, dict):
                    if 'center' in region and 'Q' in region:
                        # 椭球体字典表示
                        e = Ellipsoid(region['center'], region['Q'])
                        if e.contains(point):
                            in_any_region = True
                            break
                    elif 'center' in region and 'radius' in region:
                        # 球体表示
                        center = np.array(region['center'])
                        radius = region['radius'] 
                        if np.linalg.norm(point - center) <= radius:
                            in_any_region = True
                            break
                    
            if in_any_region:
                points_in_safe_region += 1
        
        # 计算安全分数
        safety_score = points_in_safe_region / len(dense_path) if dense_path else 0
        safe = safety_score > 0.95  # 如果95%以上的点在安全区域内，则认为是安全的
        
        return {
            'safe': safe,
            'safety_score': safety_score,
            'inside_points': points_in_safe_region,
            'total_points': len(dense_path)
        }
    except Exception as e:
        print(f"安全区域检查失败: {e}")
        import traceback
        traceback.print_exc()
        return {'safe': False, 'safety_score': 0, 'error': str(e)}

def main():
    # 加载路径数据
    path = load_path()
    if path is None:
        print("无法加载路径数据，分析终止")
        return
        
    # 加载障碍物数据
    obstacles = load_obstacles()
    
    # 加载安全区域
    safe_regions = load_safe_regions()
    
    # 计算路径长度
    path_length = calculate_path_length(path)
    print(f"路径长度: {path_length:.2f}")
    
    # 计算平滑度
    smoothness = analyze_path_smoothness(path)
    print(f"平均角度: {smoothness:.2f}°")
    
    # 检查碰撞
    collision_result = check_collisions(path, obstacles)
    if len(collision_result) > 0:
        print(f"警告: 路径与障碍物碰撞，共有 {len(collision_result)} 个碰撞点")
    else:
        print("路径与障碍物无碰撞")
    
    # 检查安全区域
    safety_result = check_path_safety(path, safe_regions)
    if safety_result['safe']:
        print(f"路径在安全区域内，安全分数: {safety_result['safety_score']:.2f}")
    else:
        print(f"警告: 路径不完全在安全区域内，安全分数: {safety_result['safety_score']:.2f}")
    
    # 汇总分析结果
    results = {
        'path_points': len(path),
        'path_length': path_length,
        'smoothness': {'avg_angle': smoothness},
        'collision': {'collisions': len(collision_result) > 0, 'collision_count': len(collision_result), 'collision_points': collision_result},
        'safety': safety_result
    }
    
    # 保存分析结果
    save_analysis_results(results)
    
    # 可视化路径和障碍物
    try:
        visualize_path_with_obstacles(path, obstacles, collision_result if len(collision_result) > 0 else None)
    except Exception as e:
        print(f"可视化时出错: {e}")
        
    print("分析完成!")

if __name__ == "__main__":
    main() 