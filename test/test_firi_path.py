import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from peng_test_v2 import ConvexPolytope, Ellipsoid  # 添加导入

# 加载保存的障碍物信息
def load_obstacles():
    # 查找最新的debug文件
    temp_files = [f for f in os.listdir('temp') if f.startswith('path_planning_debug_')]
    if not temp_files:
        print("未找到障碍物信息文件")
        return []
    
    # 选择最新的文件
    latest_file = sorted(temp_files)[-1]
    file_path = os.path.join('temp', latest_file)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    obstacles = []
    if 'obstacles' in data:
        for obs in data['obstacles']:
            vertices = np.array(obs['vertices'])
            center = np.mean(vertices, axis=0)
            radius = np.max(np.linalg.norm(vertices - center, axis=1))
            obstacles.append({'center': center, 'radius': radius, 'vertices': vertices})
    
    return obstacles

def visualize_path():
    # 尝试加载路径数据
    try:
        with open('temp/final_path.pkl', 'rb') as f:
            path = pickle.load(f)
        
        with open('temp/adjusted_path.pkl', 'rb') as f:
            adjusted_path = pickle.load(f)
        
        # 加载安全区域信息
        safe_regions = []
        i = 0
        while os.path.exists(f'temp/safe_region_{i}.pkl'):
            with open(f'temp/safe_region_{i}.pkl', 'rb') as f:
                region_data = pickle.load(f)
                # 获取椭球体中心
                if isinstance(region_data, tuple) and len(region_data) == 2:
                    _, ellipsoid = region_data  # 解包元组
                    if ellipsoid is not None:
                        center = ellipsoid.center
                        safe_regions.append({"ellipsoid_center": center})
            i += 1
        
        # 加载障碍物
        obstacles = load_obstacles()
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制障碍物
        for obs in obstacles:
            # 绘制简化的球体表示
            center = obs['center']
            radius = obs['radius']
            
            # 创建球体
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color='red', alpha=0.3)
        
        # 绘制调整后的路径点
        adj_path_array = np.array(adjusted_path)
        ax.scatter(adj_path_array[:, 0], adj_path_array[:, 1], adj_path_array[:, 2], 
                  color='green', s=50, marker='o', label='调整路径点')
        
        # 绘制调整后的路径连线
        for i in range(len(adjusted_path)-1):
            start = adjusted_path[i]
            end = adjusted_path[i+1]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'g--', linewidth=1)
        
        # 绘制安全区域中心
        for region in safe_regions:
            center = region['ellipsoid_center']
            ax.scatter(center[0], center[1], center[2], color='cyan', s=80, 
                      marker='*', label='安全区域中心')
        
        # 绘制最终平滑路径
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
               'b-', linewidth=2, label='平滑路径')
        
        # 标记起点和终点
        ax.scatter(path_array[0, 0], path_array[0, 1], path_array[0, 2], 
                  color='lime', s=100, marker='o', label='起点')
        ax.scatter(path_array[-1, 0], path_array[-1, 1], path_array[-1, 2], 
                  color='purple', s=100, marker='o', label='终点')
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('FIRI路径规划结果')
        
        # 读取安全状态信息
        safety_info = ""
        if os.path.exists('temp/path_safety.txt'):
            with open('temp/path_safety.txt', 'r') as f:
                safety_info = f.read()
        
        ax.text2D(0.05, 0.95, f"路径信息:\n{safety_info}", transform=ax.transAxes)
        
        # 添加图例，但去除重复项
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
                 if l not in labels[:i]]
        ax.legend(*zip(*unique))
        
        # 保存图片
        timestamp = int(time.time())
        plt.savefig(f'temp/path_visualization_{timestamp}.png')
        print(f"可视化结果已保存至 temp/path_visualization_{timestamp}.png")
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")

if __name__ == "__main__":
    visualize_path()
    print("分析完成!") 