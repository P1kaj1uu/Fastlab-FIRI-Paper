import os
import time
import json
from datetime import datetime
import platform
import psutil
import matplotlib.pyplot as plt
import traceback

class PerformanceEvaluator:
    """
    Performance evaluation tool for recording algorithm execution time and resource usage
    """
    def __init__(self, output_file="temp/performance_data.json"):
        self.timestamps = {}
        self.durations = {}
        self.start_times = {}
        self.output_file = output_file
        self.system_info = self._get_system_info()

        # Record program start time
        self.program_start_time = time.time()
        self.timestamps["program_start"] = self._get_current_time_str()

    def _get_system_info(self):
        """Get system information"""
        try:
            # Get basic system info
            system_info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
            }

            # Try to get detailed CPU and memory info
            try:
                import psutil
                memory = psutil.virtual_memory()
                system_info.update({
                    "total_memory_gb": round(memory.total / (1024 ** 3), 2),
                    "available_memory_gb": round(memory.available / (1024 ** 3), 2),
                })
            except ImportError:
                pass

            # Try to detect if running on Jetson platform
            try:
                if os.path.exists("/etc/nv_tegra_release"):
                    system_info["platform_type"] = "NVIDIA Jetson"
                    # Try to get Jetson model
                    try:
                        with open("/proc/device-tree/model", "r") as f:
                            jetson_model = f.read().strip()
                            system_info["jetson_model"] = jetson_model
                    except:
                        pass
            except:
                pass

            return system_info
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {"error": str(e)}

    def _get_current_time_str(self):
        """Get formatted string of current time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def start_timer(self, step_name):
        """Start timing for a specific step"""
        self.start_times[step_name] = time.time()
        self.timestamps[f"{step_name}_start"] = self._get_current_time_str()
        return self.start_times[step_name]

    def stop_timer(self, step_name):
        """Stop timing for a specific step and record duration"""
        if step_name in self.start_times:
            end_time = time.time()
            duration = end_time - self.start_times[step_name]
            self.durations[step_name] = duration
            self.timestamps[f"{step_name}_end"] = self._get_current_time_str()
            print(f"Step '{step_name}' took: {duration:.4f} seconds")
            return duration
        else:
            print(f"Error: Step '{step_name}' was not started")
            return None

    def record_value(self, key, value):
        """Record a specific value (e.g., path length, point count)"""
        self.durations[key] = value

    def save_results(self):
        """Save performance evaluation results to file"""
        # Calculate total execution time
        total_time = time.time() - self.program_start_time
        self.durations["total_execution_time"] = total_time
        self.timestamps["program_end"] = self._get_current_time_str()

        # Prepare output data
        output_data = {
            "system_info": self.system_info,
            "timestamps": self.timestamps,
            "durations": self.durations,
            "total_execution_time": total_time
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Save to JSON file
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Also generate a readable text report
        report_file = self.output_file.replace('.json', '.txt')
        with open(report_file, 'w') as f:
            f.write("=== Path Planning Algorithm Performance Report ===\n\n")

            f.write("System Information:\n")
            for key, value in self.system_info.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nTimestamps:\n")
            for key, value in self.timestamps.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nExecution Times:\n")
            for key, value in self.durations.items():
                if isinstance(value, (int, float)):
                    if key.endswith("_count") or key.endswith("_size") or key.endswith("_length"):
                        f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {key}: {value:.4f} seconds\n")
                else:
                    f.write(f"  {key}: {value}\n")

            f.write(f"\nTotal Execution Time: {total_time:.4f} seconds\n")
            f.write(f"\nReport Generated: {self._get_current_time_str()}\n")

        print(f"Performance results saved to {self.output_file} and {report_file}")

        # Generate performance chart
        self._generate_performance_chart()

        return output_data

    def _generate_performance_chart(self):
        """Generate performance chart"""
        try:
            # Set font to ensure compatibility
            plt.rcParams['font.family'] = 'DejaVu Sans'

            # Filter timing-related data
            timing_data = {k: v for k, v in self.durations.items()
                          if isinstance(v, (int, float)) and not (
                              k.endswith("_count") or
                              k.endswith("_size") or
                              k.endswith("_length") or
                              k == "total_execution_time"
                          )}

            if not timing_data:
                return

            # Sort by value
            sorted_items = sorted(timing_data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels, values, color='skyblue')

            # Add value labels
            for i, v in enumerate(values):
                plt.text(v + 0.01, i, f"{v:.4f}s", va='center')

            plt.title('Algorithm Stage Execution Times (seconds)')
            plt.xlabel('Execution Time (seconds)')
            plt.tight_layout()

            # Save chart
            chart_path = self.output_file.replace('.json', '_chart.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Performance chart saved to {chart_path}")
        except Exception as e:
            print(f"Error generating performance chart: {e}")
            traceback.print_exc()
