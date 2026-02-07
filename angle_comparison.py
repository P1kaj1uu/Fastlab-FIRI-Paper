import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def load_path(path_file):
    """加载轨迹数据"""
    try:
        if os.path.exists(path_file):
            with open(path_file, 'rb') as f:
                path = pickle.load(f)
            return path
        return None
    except Exception as e:
        print(f"加载路径数据失败: {e}")
        return None

def load_obstacles(obstacle_file='temp/obstacles.pkl'):
    """加载障碍物数据"""
    try:
        if os.path.exists(obstacle_file):
            with open(obstacle_file, 'rb') as f:
                obstacles = pickle.load(f)
            return obstacles
        return None
    except Exception as e:
        print(f"加载障碍物数据失败: {e}")
        return None

def calculate_angles(path):
    """计算轨迹点之间的角度"""
    angles = []
    
    for i in range(1, len(path) - 1):
        # 相邻两段的向量
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        
        # 计算夹角
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            angles.append(0)
            continue
            
        cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)
    
    return angles

def compare_paths(original_path, smoothed_path, obstacles=None):
    """比较优化前后的路径角度变化"""
    # 计算角度
    orig_angles = calculate_angles(original_path)
    smooth_angles = calculate_angles(smoothed_path)
    
    # 输出统计信息
    print("原始路径统计:")
    print(f"  点数: {len(original_path)}")
    print(f"  平均角度: {np.mean(orig_angles):.2f}°")
    print(f"  最大角度: {np.max(orig_angles):.2f}°")
    print(f"  角度>90°的数量: {sum(1 for a in orig_angles if a > 90)}")
    print(f"  角度>60°的数量: {sum(1 for a in orig_angles if a > 60)}")
    
    print("\n优化后路径统计:")
    print(f"  点数: {len(smoothed_path)}")
    print(f"  平均角度: {np.mean(smooth_angles):.2f}°")
    print(f"  最大角度: {np.max(smooth_angles):.2f}°")
    print(f"  角度>90°的数量: {sum(1 for a in smooth_angles if a > 90)}")
    print(f"  角度>60°的数量: {sum(1 for a in smooth_angles if a > 60)}")
    
    # 角度改进比例
    angle_reduction = 100 * (1 - np.mean(smooth_angles) / np.mean(orig_angles))
    print(f"\n平均角度减小: {angle_reduction:.2f}%")
    
    # 可视化比较图
    fig = plt.figure(figsize=(16, 10))
    
    # 3D路径比较
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot([p[0] for p in original_path], 
             [p[1] for p in original_path], 
             [p[2] for p in original_path], 'r-', linewidth=2, label='原始路径')
    ax1.scatter([p[0] for p in original_path], 
                [p[1] for p in original_path], 
                [p[2] for p in original_path], c='red', marker='.', s=30)
    ax1.set_title('原始路径')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot([p[0] for p in smoothed_path], 
             [p[1] for p in smoothed_path], 
             [p[2] for p in smoothed_path], 'g-', linewidth=2, label='优化路径')
    ax2.scatter([p[0] for p in smoothed_path], 
                [p[1] for p in smoothed_path], 
                [p[2] for p in smoothed_path], c='green', marker='.', s=30)
    ax2.set_title('优化后路径')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 绘制障碍物（如果有）
    if obstacles is not None:
        for obs in obstacles.obstacle_list:
            try:
                center = np.array(obs.center)
                radius = obs.radius
                
                # 在两个子图上添加
                for ax in [ax1, ax2]:
                    u = np.linspace(0, 2*np.pi, 20)
                    v = np.linspace(0, np.pi, 10)
                    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax.plot_surface(x, y, z, color='blue', alpha=0.1)
            except:
                continue
    
    # 同时显示两条路径
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot([p[0] for p in original_path], 
             [p[1] for p in original_path], 
             [p[2] for p in original_path], 'r-', linewidth=2, label='原始路径')
    ax3.plot([p[0] for p in smoothed_path], 
             [p[1] for p in smoothed_path], 
             [p[2] for p in smoothed_path], 'g-', linewidth=2, label='优化路径')
    ax3.set_title('路径对比')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # 角度变化曲线
    ax4 = fig.add_subplot(212)
    x_orig = range(1, len(orig_angles)+1)
    x_smooth = range(1, len(smooth_angles)+1)
    
    # 调整X轴以匹配位置（如果点数不同）
    if len(original_path) != len(smoothed_path):
        x_orig = np.linspace(1, 100, len(orig_angles))
        x_smooth = np.linspace(1, 100, len(smooth_angles))
    
    ax4.plot(x_orig, orig_angles, 'r-o', linewidth=2, markersize=5, label='原始角度')
    ax4.plot(x_smooth, smooth_angles, 'g-o', linewidth=2, markersize=5, label='优化后角度')
    ax4.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60° 阈值')
    ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90° 阈值')
    ax4.set_xlabel('路径点位置 (%)')
    ax4.set_ylabel('角度 (度)')
    ax4.set_title('角度变化对比')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = int(time.time())
    filename = f'temp/angle_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"比较图已保存至: {filename}")

def main():
    # 加载原始路径和优化后的路径
    original_path = load_path('temp/final_path.pkl')
    smoothed_path = load_path('temp/smoothed_path.pkl')
    
    if original_path is None:
        print("无法加载原始路径数据")
        return
        
    if smoothed_path is None:
        print("无法加载优化后的路径数据")
        return
    
    # 加载障碍物
    obstacles = load_obstacles()
    
    # 比较两条路径
    compare_paths(original_path, smoothed_path, obstacles)

if __name__ == "__main__":
    main() 