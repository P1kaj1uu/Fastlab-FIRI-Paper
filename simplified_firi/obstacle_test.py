import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pickle
import argparse

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

def generate_random_obstacles(num_obstacles=6):
    """Generate random 3D sphere obstacles"""
    obstacles = []
    min_coords = np.array([-5, -5, -5])
    max_coords = np.array([15, 15, 5])
    
    # Ensure no obstacles near start and goal points
    start_point = np.array([0.0, 0.0, 0.0])
    goal_point = np.array([10.0, 10.0, 0.0])
    
    for _ in range(num_obstacles):
        while True:
            center = np.random.uniform(min_coords, max_coords)
            radius = np.random.uniform(0.5, 2.0)
            
            # Check if obstacle is too close to start or goal
            if (np.linalg.norm(center - start_point) < radius + 1.0 or
                np.linalg.norm(center - goal_point) < radius + 1.0):
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

def visualize_obstacles(obstacles, start_point, goal_point):
    """Create a simple 3D visualization of obstacles"""
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
    
    # Plot start and goal
    ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], color='g', s=100, label='Start')
    ax.scatter([goal_point[0]], [goal_point[1]], [goal_point[2]], color='b', s=100, label='Goal')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Obstacles and Waypoints')
    ax.legend()
    
    # Save figure
    plt.savefig('temp/obstacles.png')
    
    return fig

if __name__ == "__main__":
    # Process command-line arguments
    parser = argparse.ArgumentParser(description='Obstacle Generation Test')
    parser.add_argument('--obstacles', type=int, default=6, help='Number of obstacles to generate')
    args = parser.parse_args()
    
    # Display run parameters
    print(f"Running with: obstacles={args.obstacles}")
    
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
    
    # Save obstacle data
    print("\nSaving obstacle data...")
    obstacle_data = [(obs.center, obs.radius) for obs in obstacles]
    with open('temp/obstacle_data.pkl', 'wb') as f:
        pickle.dump(obstacle_data, f)
    
    # Visualize obstacles
    print("\nVisualizing obstacles...")
    visualize_obstacles(obstacles, start_point, goal_point)
    
    print("\nResults saved to temp directory:")
    print("- obstacle_data.pkl: Obstacle data")
    print("- obstacles.png: Visualization of obstacles")
    
    print("\nTest completed successfully!") 