import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy import interpolate
import scipy.spatial

class Obstacle:
    """
    Simple sphere obstacle representation
    """
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        
    def __str__(self):
        return f"Obstacle(center={self.center}, radius={self.radius})"
        
    def is_point_in_collision(self, point, safety_margin=0):
        """Check if a point collides with this obstacle"""
        distance = np.linalg.norm(np.array(point) - self.center)
        return distance <= self.radius + safety_margin
        
    def is_segment_in_collision(self, p1, p2, safety_margin=0, samples=10):
        """Check if a line segment collides with this obstacle"""
        # Check endpoints
        if self.is_point_in_collision(p1, safety_margin) or self.is_point_in_collision(p2, safety_margin):
            return True
            
        # Sample points along the segment
        for t in np.linspace(0, 1, samples):
            point = p1 * (1-t) + p2 * t
            if self.is_point_in_collision(point, safety_margin):
                return True
                
        return False

class ObstacleSet:
    """
    A collection of obstacles that can be used for collision checking
    """
    def __init__(self):
        self.obstacle_list = []
        
    def add_obstacle(self, obstacle):
        self.obstacle_list.append(obstacle)
        
    def is_point_in_collision(self, point, safety_margin=0):
        """Check if a point collides with any obstacle"""
        for obstacle in self.obstacle_list:
            if obstacle.is_point_in_collision(point, safety_margin):
                return True
        return False
        
    def is_segment_in_collision(self, p1, p2, safety_margin=0, samples=10):
        """Check if a line segment collides with any obstacle"""
        for obstacle in self.obstacle_list:
            if obstacle.is_segment_in_collision(p1, p2, safety_margin, samples):
                return True
        return False
        
    def __iter__(self):
        """Make the obstacle set iterable"""
        return iter(self.obstacle_list)
        
    def __len__(self):
        """Return the number of obstacles"""
        return len(self.obstacle_list)

class Ellipsoid:
    """
    Simple ellipsoid representation for safe regions
    """
    def __init__(self, center, axes_lengths):
        self.center = np.array(center)
        self.axes_lengths = np.array(axes_lengths)
        
    def contains(self, point):
        """Check if point is inside the ellipsoid"""
        p = np.array(point) - self.center
        return sum((p / self.axes_lengths)**2) <= 1.0
        
    def __str__(self):
        return f"Ellipsoid(center={self.center}, axes_lengths={self.axes_lengths})"

class FIRIPlanner:
    """
    Simplified Fast Iterative Regional Inflation (FIRI) path planner
    """
    def __init__(self, obstacles):
        """
        Initialize the planner with obstacles
        """
        self.obstacles = obstacles
        
    def plan_path(self, start, goal, safety_margin=1.0):
        """
        Plan a path from start to goal while avoiding obstacles
        
        Parameters:
            start: Start point coordinates
            goal: Goal point coordinates
            safety_margin: Safety margin for obstacle avoidance
            
        Returns:
            path: List of waypoints
            safe_areas: List of safe ellipsoids for each waypoint
        """
        print(f"Planning path from {start} to {goal} with safety margin {safety_margin}...")
        
        # Check if start or goal is in collision
        if self.obstacles.is_point_in_collision(start, safety_margin):
            print("Error: Start point is in collision with obstacles")
            return [], []
            
        if self.obstacles.is_point_in_collision(goal, safety_margin):
            print("Error: Goal point is in collision with obstacles")
            return [], []
            
        # In a simplified version, we'll create a straight-line path and check for collisions
        direction = goal - start
        distance = np.linalg.norm(direction)
        
        # If the straight line is collision-free, use it
        if not self.obstacles.is_segment_in_collision(start, goal, safety_margin):
            print("Direct path is collision-free")
            return [start, goal], self._generate_safe_ellipsoids([start, goal])
            
        # Otherwise, create a simple path with intermediate waypoints
        print("Direct path has collisions, creating waypoints...")
        num_waypoints = 5  # Including start and goal
        
        # Create waypoints along a slightly curved path
        t = np.linspace(0, 1, num_waypoints)
        
        # Create a curved path by adding an offset in the middle
        midpoint = (start + goal) / 2
        # Find a direction perpendicular to the start-goal line
        if np.abs(direction[0]) > 0.1 or np.abs(direction[1]) > 0.1:
            perp_dir = np.array([-direction[1], direction[0], 0])
        else:
            perp_dir = np.array([0, -direction[2], direction[1]])
        perp_dir = perp_dir / np.linalg.norm(perp_dir)
        
        # The curve will bend out by up to half the direct distance
        max_offset = distance * 0.5
        
        # Create the path with a curved middle section
        path = []
        for i in range(num_waypoints):
            offset = max_offset * np.sin(t[i] * np.pi)
            point = start + t[i] * direction + offset * perp_dir
            
            # Adjust the point to avoid collisions
            attempts = 10
            while self.obstacles.is_point_in_collision(point, safety_margin) and attempts > 0:
                # Move the point away from nearby obstacles
                gradient = np.zeros(3)
                for obs in self.obstacles:
                    vec = point - obs.center
                    dist = np.linalg.norm(vec)
                    if dist < obs.radius + safety_margin + 1.0:
                        # Add repulsive force inversely proportional to distance
                        force = max(0, obs.radius + safety_margin + 1.0 - dist)
                        gradient += force * vec / (dist + 1e-6)
                
                if np.linalg.norm(gradient) > 0:
                    # Move in the direction of the gradient
                    point += 0.2 * gradient / np.linalg.norm(gradient)
                else:
                    # If no gradient, make a random move
                    point += 0.2 * np.random.randn(3)
                    
                attempts -= 1
            
            path.append(point)
        
        # Generate safe ellipsoids for each waypoint
        safe_areas = self._generate_safe_ellipsoids(path)
        
        return np.array(path), safe_areas
    
    def _generate_safe_ellipsoids(self, path):
        """Generate simple safe ellipsoids around waypoints"""
        safe_areas = []
        for point in path:
            # Find distance to closest obstacle
            min_dist = float('inf')
            for obs in self.obstacles:
                dist = np.linalg.norm(point - obs.center) - obs.radius
                min_dist = min(min_dist, dist)
            
            # Ensure min_dist is positive (point is outside obstacles)
            min_dist = max(min_dist, 0.1)
            
            # Create an ellipsoid with all axes equal to the minimum distance
            axes_lengths = np.ones(3) * min_dist
            
            safe_areas.append(Ellipsoid(point, axes_lengths))
        
        return safe_areas
    
    def check_path_safety(self, path):
        """
        Check if a path is collision-free
        
        Parameters:
            path: List of waypoints
            
        Returns:
            Dictionary with safety information
        """
        collision_count = 0
        
        # Check each segment
        for i in range(len(path) - 1):
            if self.obstacles.is_segment_in_collision(path[i], path[i+1]):
                collision_count += 1
                
        return {
            'is_safe': collision_count == 0,
            'collision_count': collision_count
        }

def generate_random_obstacles(num_obstacles=6):
    """Generate random 3D sphere obstacles"""
    obstacles = []
    min_coords = np.array([-5, -5, -5])
    max_coords = np.array([15, 15, 5])
    
    # Ensure no obstacles near start and goal points
    start_point = np.array([0.0, 0.0, 0.0])
    goal_point = np.array([10.0, 10.0, 0.0])
    
    # Increase the safety buffer around start and goal
    start_safety_buffer = 3.0  # Increased from 1.0
    goal_safety_buffer = 3.0   # Increased from 1.0
    
    # Add a strategic obstacle in the middle of the path
    midpoint = (start_point + goal_point) / 2
    midpoint_obstacle = Obstacle(midpoint, 2.0)  # Radius 2.0 ensures it blocks the direct path
    obstacles.append(midpoint_obstacle)
    print(f"Added strategic obstacle at {midpoint} with radius 2.0")
    
    # Generate the remaining random obstacles
    for _ in range(num_obstacles - 1):  # -1 because we already added one
        while True:
            center = np.random.uniform(min_coords, max_coords)
            radius = np.random.uniform(0.5, 2.0)
            
            # Check if obstacle is too close to start or goal
            if (np.linalg.norm(center - start_point) < radius + start_safety_buffer or
                np.linalg.norm(center - goal_point) < radius + goal_safety_buffer):
                continue
            
            # Check if obstacle overlaps with existing obstacles
            overlap = False
            for existing_obstacle in obstacles:
                dist = np.linalg.norm(center - existing_obstacle.center)
                if dist < radius + existing_obstacle.radius:
                    overlap = True
                    break
            
            if not overlap:
                obstacle = Obstacle(center, radius)
                obstacles.append(obstacle)
                break
    
    return obstacles

def clean_temp_files():
    """Clean temporary files from previous runs"""
    if os.path.exists('temp'):
        for file in os.listdir('temp'):
            file_path = os.path.join('temp', file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs('temp')

def visualize_path(path, obstacles, start_point, goal_point, safe_areas=None):
    """Create a 3D visualization of the path, obstacles, and safe areas"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles as spheres
    for i, obs in enumerate(obstacles):
        # Create a meshgrid for the sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obs.radius * np.outer(np.cos(u), np.sin(v)) + obs.center[0]
        y = obs.radius * np.outer(np.sin(u), np.sin(v)) + obs.center[1]
        z = obs.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs.center[2]
        
        # Plot the surface
        ax.plot_surface(x, y, z, color='r', alpha=0.5)
    
    # Plot path
    if len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'bo-', linewidth=2, markersize=5, label='Path')
    
    # Plot safe ellipsoids (if provided)
    if safe_areas:
        for ellipsoid in safe_areas:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            
            # Parametric equation of an ellipsoid
            x = ellipsoid.axes_lengths[0] * np.outer(np.cos(u), np.sin(v)) + ellipsoid.center[0]
            y = ellipsoid.axes_lengths[1] * np.outer(np.sin(u), np.sin(v)) + ellipsoid.center[1]
            z = ellipsoid.axes_lengths[2] * np.outer(np.ones(np.size(u)), np.cos(v)) + ellipsoid.center[2]
            
            ax.plot_surface(x, y, z, color='g', alpha=0.2)
    
    # Plot start and goal
    ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], color='g', s=100, label='Start')
    ax.scatter([goal_point[0]], [goal_point[1]], [goal_point[2]], color='b', s=100, label='Goal')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Path Planning Results')
    ax.legend()
    
    # Save figure
    plt.savefig('temp/path_planning.png')
    
    return fig

def smooth_path(path, safe_areas):
    """
    Smooth the path using B-spline interpolation while staying within safe areas
    
    Parameters:
        path: Original path points
        safe_areas: List of safe areas corresponding to each path point
    """
    if len(path) < 3:
        return path
    
    # Convert path to numpy array
    path_array = np.array(path)
    
    # Create a B-spline representation
    t = np.linspace(0, 1, len(path))
    t_smooth = np.linspace(0, 1, 3 * len(path))
    
    # Fit the B-spline for each dimension
    k = min(3, len(path)-1)
    x_smooth = interpolate.make_interp_spline(t, path_array[:, 0], k=k)(t_smooth)
    y_smooth = interpolate.make_interp_spline(t, path_array[:, 1], k=k)(t_smooth)
    z_smooth = interpolate.make_interp_spline(t, path_array[:, 2], k=k)(t_smooth)
    
    # Combine the smoothed coordinates
    smoothed_path = np.column_stack((x_smooth, y_smooth, z_smooth))
    
    return smoothed_path

def analyze_planning_results(path, obstacles, start, goal, planning_time):
    """
    Analyze path planning results and generate detailed report
    
    Parameters:
        path: Planned path points
        obstacles: List of obstacles
        start: Start coordinates
        goal: Goal coordinates
        planning_time: Planning time (seconds)
    """
    print("\n======= Path Planning Result Analysis =======")
    
    # Create results directory
    if not os.path.exists('temp'):
        os.makedirs('temp')
        
    # Calculate path length
    path_length = 0.0
    for i in range(len(path)-1):
        segment_length = np.linalg.norm(path[i+1] - path[i])
        path_length += segment_length
    
    # Calculate direct distance
    direct_distance = np.linalg.norm(goal - start)
    
    # Calculate path curvature (ratio of path length to direct distance)
    path_curvature_ratio = path_length / direct_distance if direct_distance > 0 else 1.0
    
    # Calculate path smoothness (through angle changes)
    angles = []
    sharp_turns = 0
    if len(path) > 2:
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                angles.append(angle_deg)
                if angle_deg > 90:
                    sharp_turns += 1
    
    avg_angle = np.mean(angles) if angles else 0
    max_angle = np.max(angles) if angles else 0
    smoothness_score = 10 - min(10, max_angle / 18)  # Map max angle to 0-10 score
    
    # Check if each path segment collides with obstacles
    planner = FIRIPlanner(obstacles)
    path_safety = planner.check_path_safety(path)
    
    # Generate report
    report = {
        'path_points': len(path),
        'path_length': path_length,
        'direct_distance': direct_distance,
        'path_curvature_ratio': path_curvature_ratio,
        'avg_angle': avg_angle,
        'max_angle': max_angle,
        'sharp_turns': sharp_turns,
        'smoothness_score': smoothness_score,
        'is_collision_free': path_safety['is_safe'],
        'collision_segments': path_safety['collision_count'],
        'planning_time': planning_time
    }
    
    # Print report
    print(f"Path points count: {report['path_points']}")
    print(f"Total path length: {report['path_length']:.2f} units")
    print(f"Direct distance between start and goal: {report['direct_distance']:.2f} units")
    print(f"Path curvature ratio: {report['path_curvature_ratio']:.2f} (closer to 1 is straighter)")
    print(f"Average turning angle: {report['avg_angle']:.2f}°")
    print(f"Maximum turning angle: {report['max_angle']:.2f}°")
    print(f"Sharp turns count (>90°): {report['sharp_turns']}")
    print(f"Smoothness score (0-10): {report['smoothness_score']:.2f}")
    print(f"Collision free: {'Yes' if report['is_collision_free'] else 'No'}")
    if not report['is_collision_free']:
        print(f"Collision segments count: {report['collision_segments']}")
    print(f"Planning time: {report['planning_time']:.4f} seconds")
    
    # Save to file
    with open('temp/planning_analysis.txt', 'w') as f:
        f.write("====== FIRI Path Planning Analysis Report ======\n\n")
        for key, value in report.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return report

if __name__ == "__main__":
    # Process command-line arguments
    parser = argparse.ArgumentParser(description='Simplified FIRI Path Planning Test')
    parser.add_argument('--obstacles', type=int, default=6, help='Number of obstacles to generate')
    parser.add_argument('--margin', type=float, default=1.5, help='Safety margin for obstacles')
    args = parser.parse_args()
    
    # Display run parameters
    print(f"Running with: obstacles={args.obstacles}, safety_margin={args.margin}")
    
    # Clean temporary files
    print("\nCleaning temporary files...")
    clean_temp_files()
    
    # Generate random obstacles
    print("\nGenerating obstacles...")
    obstacles = generate_random_obstacles(args.obstacles)
    
    # Create obstacle set for collision detection
    obstacle_set = ObstacleSet()
    for obstacle in obstacles:
        obstacle_set.add_obstacle(obstacle)
    
    # Set start and goal points
    start_point = np.array([0.0, 0.0, 0.0])
    goal_point = np.array([10.0, 10.0, 0.0])
    
    # Print obstacle information
    print(f"Number of obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"Obstacle {i+1}: center={obs.center}, radius={obs.radius}")
    
    # Initialize FIRI planner
    print("\nInitializing FIRI path planner...")
    planner = FIRIPlanner(obstacle_set)
    
    # Plan path
    print("\nPlanning path...")
    start_time = time.time()
    path, safe_areas = planner.plan_path(start_point, goal_point, safety_margin=args.margin)
    planning_time = time.time() - start_time
    
    # Smooth path
    print("\nSmoothing path...")
    smoothed_path = smooth_path(path, safe_areas)
    
    # Check path safety
    safety_result = planner.check_path_safety(smoothed_path)
    print(f"Path planning completed in {planning_time:.4f} seconds")
    print(f"Path is safe: {safety_result['is_safe']}")
    if not safety_result['is_safe']:
        print(f"Collision detected in {safety_result['collision_count']} path segments")
    
    # Save path data
    print("\nSaving path data...")
    with open('temp/path_data.pkl', 'wb') as f:
        pickle.dump({
            'path': path,
            'smoothed_path': smoothed_path,
            'start': start_point,
            'goal': goal_point,
            'obstacles': [(obs.center, obs.radius) for obs in obstacles]
        }, f)
    
    # Analyze path
    analysis = analyze_planning_results(smoothed_path, obstacle_set, start_point, goal_point, planning_time)
    
    # Visualize path and obstacles
    print("\nVisualizing path planning results...")
    visualize_path(path, obstacle_set, start_point, goal_point, safe_areas)
    visualize_path(smoothed_path, obstacle_set, start_point, goal_point)
    
    print("\nResults saved to temp directory:")
    print("- path_data.pkl: Path and obstacle data")
    print("- planning_analysis.txt: Path planning analysis report")
    print("- path_planning.png: Visualization of path and obstacles")
    
    print("\nSimplified FIRI planner test completed successfully!") 