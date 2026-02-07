import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time

# 从main.py中导入Obstacle和ObstacleSet类
from main import Obstacle, ObstacleSet

def load_path(path_file='temp/smoothed_path.pkl'):
    """
    加载路径数据，优先从smoothed_path.pkl加载，如果不存在则尝试其他文件
    """
    try:
        # 先尝试指定的文件
        if os.path.exists(path_file):
            with open(path_file, 'rb') as f:
                path = pickle.load(f)
                return path
                
        # 按优先级尝试其他可能的路径文件
        potential_files = ['temp/smoothed_path.pkl', 'temp/final_path.pkl', 'temp/adjusted_path.pkl', 'temp/path_points.pkl']
        for file in potential_files:
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    path = pickle.load(f)
                    print(f"从 {file} 加载轨迹数据")
                    return path
                
        print("未找到任何路径数据文件")
        return None
    except Exception as e:
        print(f"加载路径数据时出错: {e}")
        return None

def load_obstacles(obstacle_file='temp/obstacles.pkl'):
    """
    加载障碍物数据
    """
    try:
        with open(obstacle_file, 'rb') as f:
            obstacles = pickle.load(f)
        return obstacles
    except Exception as e:
        print(f"加载障碍物数据时出错: {e}")
        return None

def load_safe_regions():
    """
    加载安全区域数据
    """
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
        if i > 0:
            print(f"已加载 {len(safe_regions)} 个安全区域")
        return safe_regions
    except Exception as e:
        print(f"加载安全区域数据时出错: {e}")
        return []

def analyze_angles(path_points):
    """
    计算路径中连续段之间的角度
    """
    angles = []
    
    for i in range(1, len(path_points) - 1):
        v1 = path_points[i] - path_points[i-1]
        v2 = path_points[i+1] - path_points[i]
        
        # 处理零向量情况
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            # 如果任一向量接近零向量，则设置角度为0
            angle = 0.0
        else:
            # 计算夹角（弧度）
            cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
        
        angles.append(angle)
    
    return angles

def smooth_path(path_points, window_size=3, iterations=5, angle_limit=45.0):
    """
    平滑路径，减小急转弯
    
    参数:
        path_points: 原始路径点
        window_size: 平滑窗口大小
        iterations: 平滑迭代次数
        angle_limit: 角度限制，超过此角度的点会重点处理
        
    返回:
        平滑后的路径点
    """
    if len(path_points) <= 2:
        return path_points.copy()
    
    # 转换为numpy数组方便处理
    path = np.array(path_points)
    smoothed_path = path.copy()
    
    # 计算原始角度
    original_angles = analyze_angles(path)
    
    # 记录急转弯点
    sharp_turns = [i+1 for i, angle in enumerate(original_angles) if angle > angle_limit]
    
    # 如果没有急转弯点，使用标准平滑
    if not sharp_turns:
        for _ in range(iterations):
            # 保存起点和终点
            start_point = smoothed_path[0].copy()
            end_point = smoothed_path[-1].copy()
            
            # 平滑中间点
            for i in range(1, len(smoothed_path) - 1):
                # 获取窗口内的点
                window_start = max(0, i - window_size // 2)
                window_end = min(len(smoothed_path), i + window_size // 2 + 1)
                window_points = smoothed_path[window_start:window_end]
                
                # 计算窗口内点的加权平均
                weights = np.ones(len(window_points))
                weights = weights / np.sum(weights)
                smoothed_path[i] = np.sum(window_points * weights.reshape(-1, 1), axis=0)
            
            # 恢复起点和终点
            smoothed_path[0] = start_point
            smoothed_path[-1] = end_point
    else:
        # 针对急转弯点进行优化平滑
        for _ in range(iterations):
            # 保存起点和终点
            start_point = smoothed_path[0].copy()
            end_point = smoothed_path[-1].copy()
            
            # 平滑中间点，特别关注急转弯点
            for i in range(1, len(smoothed_path) - 1):
                # 对急转弯点使用较大窗口
                if i in sharp_turns:
                    local_window = min(window_size * 2, len(smoothed_path) // 2)
                    window_start = max(0, i - local_window // 2)
                    window_end = min(len(smoothed_path), i + local_window // 2 + 1)
                    
                    # 使用加权平均，中心点权重低，远点权重高
                    window_points = smoothed_path[window_start:window_end]
                    weights = np.ones(len(window_points))
                    
                    # 中心点权重降低
                    center_idx = i - window_start
                    weights[center_idx] = 0.5
                    
                    # 归一化权重
                    weights = weights / np.sum(weights)
                    smoothed_path[i] = np.sum(window_points * weights.reshape(-1, 1), axis=0)
                else:
                    # 普通点使用标准平滑
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(smoothed_path), i + window_size // 2 + 1)
                    window_points = smoothed_path[window_start:window_end]
                    
                    weights = np.ones(len(window_points))
                    weights = weights / np.sum(weights)
                    smoothed_path[i] = np.sum(window_points * weights.reshape(-1, 1), axis=0)
            
            # 恢复起点和终点
            smoothed_path[0] = start_point
            smoothed_path[-1] = end_point
    
    return smoothed_path

def insert_midpoints(path_points, angle_threshold=90.0):
    """
    在急转弯处插入中间点，减小角度变化
    
    参数:
        path_points: 原始路径点
        angle_threshold: 角度阈值，超过此值将插入中间点
        
    返回:
        插入中间点后的路径
    """
    if len(path_points) <= 2:
        return path_points.copy()
    
    new_path = [path_points[0]]
    angles = analyze_angles(path_points)
    
    for i in range(len(angles)):
        point_idx = i + 1
        
        # 添加当前点
        new_path.append(path_points[point_idx])
        
        # 如果角度超过阈值，在当前点和下一点之间插入中间点
        if angles[i] > angle_threshold and point_idx < len(path_points) - 1:
            # 计算插入点数量（角度越大，插入越多）
            num_points = int(min(5, math.ceil(angles[i] / 45.0)))
            
            # 当前点和下一点
            current = path_points[point_idx]
            next_point = path_points[point_idx + 1]
            
            # 插入等距离的中间点
            for j in range(1, num_points):
                t = j / (num_points + 1)
                mid_point = current + t * (next_point - current)
                new_path.append(mid_point)
    
    # 确保添加最后一个点（如果不是最后一个转角）
    if len(path_points) > len(new_path):
        new_path.append(path_points[-1])
        
    return np.array(new_path)

def visualize_path_with_angles(path_points, angles, obstacles=None, threshold=60, save_path='temp'):
    """
    可视化路径，标记显著角度，并显示障碍物
    """
    # 确保保存目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制路径
    x = [p[0] for p in path_points]
    y = [p[1] for p in path_points]
    z = [p[2] for p in path_points]
    ax.plot(x, y, z, 'b-', linewidth=2, label='路径')
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=100, label='起点')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='o', s=100, label='终点')
    
    # 标记所有路径点
    ax.scatter(x, y, z, c='blue', marker='.', s=30)
    
    # 标记显著角度的点
    significant_indices = [i+1 for i, angle in enumerate(angles) if angle > threshold and not np.isnan(angle)]
    if significant_indices:
        sig_x = [path_points[i][0] for i in significant_indices]
        sig_y = [path_points[i][1] for i in significant_indices]
        sig_z = [path_points[i][2] for i in significant_indices]
        ax.scatter(sig_x, sig_y, sig_z, c='orange', marker='*', s=200, label=f'角度 > {threshold}°')
        
        # 添加角度标注
        for i, idx in enumerate(significant_indices):
            angle_val = angles[idx-1]
            ax.text(path_points[idx][0], path_points[idx][1], path_points[idx][2], 
                   f"{angle_val:.1f}°", color='black', fontsize=8)
    
    # 绘制障碍物
    if obstacles is not None:
        for obstacle in obstacles:
            try:
                # 获取障碍物中心和半径
                if hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
                    center = obstacle.center
                    radius = obstacle.radius
                elif isinstance(obstacle, dict) and 'center' in obstacle and 'radius' in obstacle:
                    center = obstacle['center']
                    radius = obstacle['radius']
                else:
                    continue
                
                # 创建障碍物的包围球可视化
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x, y, z, color='red', alpha=0.1)
            except Exception as e:
                print(f"绘制障碍物时出错: {e}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('路径分析：角度变化与障碍物')
    plt.legend()
    
    timestamp = int(time.time())
    path_filename = f'{save_path}/path_analysis_{timestamp}.png'
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f"已保存路径分析图到 {path_filename}")
    
    # 绘制角度变化的折线图
    plt.figure(figsize=(12, 6))
    valid_angles = [a if not np.isnan(a) else 0 for a in angles]
    plt.plot(range(1, len(valid_angles)+1), valid_angles, 'b-o', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold}°)')
    
    # 标记显著角度
    for i, angle in enumerate(valid_angles):
        if angle > threshold:
            plt.annotate(f'{angle:.2f}°', 
                        (i+1, angle), 
                        textcoords="offset points",
                        xytext=(0, 10), 
                        ha='center')
    
    plt.xlabel('路径点索引')
    plt.ylabel('角度变化 (度)')
    plt.title('角度变化分析')
    plt.grid(True)
    plt.legend()
    
    angle_filename = f'{save_path}/angle_analysis_{timestamp}.png'
    plt.savefig(angle_filename, dpi=300, bbox_inches='tight')
    print(f"已保存角度分析图到 {angle_filename}")

def check_path_safety(path, obstacles, safety_margin=1.0):
    """
    检查路径是否与障碍物碰撞
    
    参数:
        path: 路径点列表
        obstacles: 障碍物列表
        safety_margin: 安全边界系数
        
    返回:
        collision_points: 碰撞点列表
    """
    if obstacles is None or not obstacles.obstacle_list:
        return []
    
    collision_points = []
    for i, point in enumerate(path):
        for obs in obstacles.obstacle_list:
            center = np.array(obs.center)
            dist = np.linalg.norm(point - center)
            if dist < obs.radius * safety_margin:
                collision_points.append(i)
                break
    
    return collision_points

def check_path_curvature(path_points):
    """
    分析路径的曲率，找出变化剧烈的区域
    
    参数:
        path_points: 路径点序列
        
    返回:
        曲率数据
    """
    if len(path_points) < 3:
        return np.zeros(len(path_points))
    
    # 计算路径长度
    segment_lengths = []
    for i in range(len(path_points) - 1):
        segment_lengths.append(np.linalg.norm(path_points[i+1] - path_points[i]))
    
    total_length = sum(segment_lengths)
    avg_segment = total_length / len(segment_lengths)
    
    # 计算曲率
    curvatures = []
    for i in range(1, len(path_points) - 1):
        # 相邻三点确定圆的曲率
        p1 = path_points[i-1]
        p2 = path_points[i]
        p3 = path_points[i+1]
        
        # 计算三点确定的圆的曲率
        # 如果三点共线或几乎共线，曲率接近0
        # 如果三点几乎重合，定义曲率为0
        
        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 检查向量是否接近零向量
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            curvatures.append(0.0)
            continue
            
        # 归一化
        v1 = v1 / norm_v1
        v2 = v2 / norm_v2
        
        # 计算夹角的余弦值
        cos_angle = np.dot(v1, v2)
        
        # 避免数值问题
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 角度越小，曲率越大；角度为180度时，曲率为0
        curvature = 1.0 - cos_angle
        curvatures.append(curvature)
    
    # 首尾点的曲率定义为相邻点的曲率
    curvatures = [curvatures[0]] + curvatures + [curvatures[-1]]
    
    return curvatures

def save_path(path_points, filename='temp/smoothed_path.pkl'):
    """
    保存路径到文件
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(path_points, f)
        print(f"已保存路径到: {filename}")
        return True
    except Exception as e:
        print(f"保存路径时出错: {e}")
        return False

def main():
    # 加载路径数据（优先使用smoothed_path.pkl）
    path_points = load_path('temp/final_path.pkl')
    if path_points is None:
        print("无法加载路径数据")
        return
    
    # 加载障碍物数据
    obstacles = load_obstacles()
    
    # 打印路径信息
    print(f"轨迹点数量: {len(path_points)}")
    print(f"起点: {path_points[0]}")
    print(f"终点: {path_points[-1]}\n")
    
    # 分析角度
    angles = analyze_angles(path_points)
    
    # 打印角度结果
    print("路径角度分析:")
    for i, angle in enumerate(angles):
        print(f"第{i+1}个点的角度: {angle:.2f}°")
    
    print()
    
    # 计算统计信息
    valid_angles = [a for a in angles if not np.isnan(a)]
    if valid_angles:
        avg_angle = np.mean(valid_angles)
        max_angle = np.max(valid_angles)
        min_angle = np.min(valid_angles)
        count_90plus = sum(1 for a in valid_angles if a > 90)
        count_60plus = sum(1 for a in valid_angles if a > 60)
        
        print(f"平均角度: {avg_angle:.2f}°")
        print(f"最大角度: {max_angle:.2f}°")
        print(f"最小角度: {min_angle:.2f}°")
        print(f"角度>90°的数量: {count_90plus}")
        print(f"角度>60°的数量: {count_60plus}")
    else:
        print("没有有效的角度数据")
    
    print()
    
    # 检查路径是否安全（无碰撞）
    if obstacles is not None:
        collision_points = check_path_safety(path_points, obstacles)
        if collision_points:
            print(f"❌ 检测到碰撞，共 {len(collision_points)} 个碰撞点")
        else:
            print("✅ 路径安全，无碰撞")
    
    # 可视化原始路径
    visualize_path_with_angles(path_points, angles, obstacles, threshold=60)
    
    # 如果有大角度变化，进行路径平滑
    if count_60plus > 0:
        print("\n正在进行路径平滑以减小角度变化...")
        
        # 首先在大角度处插入中间点
        enhanced_path = insert_midpoints(path_points, angle_threshold=60.0)
        print(f"在大角度处插入中间点后，路径点数量: {len(enhanced_path)}")
        
        # 然后进行平滑
        smoothed_path = smooth_path(enhanced_path, window_size=3, iterations=5, angle_limit=60.0)
        print("平滑完成")
        
        # 分析平滑后的角度
        smoothed_angles = analyze_angles(smoothed_path)
        
        # 计算平滑后的统计信息
        valid_smoothed_angles = [a for a in smoothed_angles if not np.isnan(a)]
        if valid_smoothed_angles:
            avg_smoothed = np.mean(valid_smoothed_angles)
            max_smoothed = np.max(valid_smoothed_angles)
            count_90plus_smoothed = sum(1 for a in valid_smoothed_angles if a > 90)
            count_60plus_smoothed = sum(1 for a in valid_smoothed_angles if a > 60)
            
            print("\n平滑后的角度统计:")
            print(f"  平均角度: {avg_smoothed:.2f}° (原始: {avg_angle:.2f}°)")
            print(f"  最大角度: {max_smoothed:.2f}° (原始: {max_angle:.2f}°)")
            print(f"  角度>90°的数量: {count_90plus_smoothed} (原始: {count_90plus})")
            print(f"  角度>60°的数量: {count_60plus_smoothed} (原始: {count_60plus})")
            
            # 保存平滑后的路径
            save_path(smoothed_path, 'temp/smoothed_path.pkl')
            
            # 可视化平滑后的路径
            print("\n正在可视化平滑后的路径...")
            visualize_path_with_angles(smoothed_path, smoothed_angles, obstacles, threshold=60)
            
            # 打印改进情况
            angle_reduction = avg_angle - avg_smoothed
            if angle_reduction > 0:
                print(f"\n路径平滑成功！平均角度减小了 {angle_reduction:.2f}°")
            else:
                print("\n路径平滑效果不明显，可能需要调整平滑参数或使用其他方法")
    else:
        print("\n路径角度变化已经较小，无需平滑")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main() 