# FIRI: 快速增量区域膨胀路径规划算法

## 论文复现

复现论文《Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions of Obstacle-Free Space》

## 项目介绍

FIRI（Fast Incremental Region Inflation）是一种先进的路径规划算法，专为在三维障碍环境中运行的自主机器人设计。该算法通过迭代计算约束椭球体和凸多面体的安全区域，生成平滑轨迹，可避开障碍物同时保持方向变化的平滑性。

## 功能特点

- **3D路径规划**: 在三维空间中生成无碰撞路径
- **基于区域的规划**: 使用增量膨胀的安全区域引导路径规划
- **自适应平滑**: 具有角度约束的路径平滑处理，减少急转弯
- **性能监控**: 内置性能评估工具用于基准测试
- **可视化工具**: 多种可视化选项（Open3D和Matplotlib）

## 项目结构

```
FIRI/
├── firi/                       # FIRI算法核心实现
│   ├── geometry/               # 几何工具（凸多面体，椭球体）
│   ├── planning/               # 路径规划算法
│   ├── utils/                  # 工具函数
│   └── visualization/          # 可视化工具
├── temp/                       # 临时数据输出目录
├── main.py                     # 主执行脚本
├── analyze_trajectory.py       # 轨迹分析工具
├── angle_comparison.py         # 路径角度比较脚本
└── README.md                   # 本文件
```

## 安装方法

### 环境要求

- Python 3.7+
- NumPy
- Matplotlib
- Open3D
- SciPy

### 安装步骤

```bash
# 安装依赖
pip install numpy matplotlib open3d scipy psutil
```

## 使用方法

### 基本执行

```bash
python main.py
```

执行后将：

1. 在3D空间中生成随机障碍物
2. 从起点到终点规划无碰撞路径
3. 应用路径平滑以减少急转弯
4. 使用Matplotlib和Open3D可视化结果
5. 在`temp`目录中生成性能指标

### 路径分析

```bash
python analyze_trajectory.py
```

分析生成的路径并提供角度、曲率和安全性统计数据。

### 路径角度比较

```bash
python angle_comparison.py
```

比较原始路径与平滑后的路径，重点关注角度变化。

## 算法详情

FIRI算法分为以下几个步骤：

1. **障碍物处理**: 将障碍物转换为适合碰撞检测的形式
2. **安全区域计算**: 生成代表安全区域的凸多面体和椭球体
3. **路径规划**: 通过安全区域创建无碰撞路径
4. **路径平滑**: 应用自适应平滑以减少急转弯
5. **碰撞验证**: 确保最终路径无碰撞

## 性能指标

算法的性能在各个阶段进行测量：

- 路径规划: ~0.08秒（核心算法）
- 路径平滑: ~0.03秒
- 总规划时间: ~0.15秒（不包括可视化）

性能指标自动保存到`temp`目录以供分析。

## 可视化功能

FIRI提供两种可视化方法：

1. **Matplotlib**: 路径和障碍物的静态3D可视化
2. **Open3D**: 交互式3D可视化，带有膨胀障碍物和路径段
