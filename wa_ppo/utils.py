"""
Utility Functions
Các hàm tiện ích chung
"""
import numpy as np
import json
import pickle
from typing import List, Dict, Tuple
import os

def save_checkpoint(model, optimizer_state, epoch, metrics, filepath):
    """
    Lưu checkpoint cho training
    """
    checkpoint = {
        'model_state': model.policy.state_dict(),
        'optimizer_state': optimizer_state,
        'epoch': epoch,
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath):
    """Load checkpoint"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

def calculate_pareto_front(solutions: List[Tuple[float, float]]) -> List[int]:
    """
    Tính Pareto front từ danh sách solutions
    Args:
        solutions: List of (objective1, objective2) tuples
    Returns:
        List of indices that are on the Pareto front
    """
    n = len(solutions)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if j dominates i
                if (solutions[j][0] <= solutions[i][0] and 
                    solutions[j][1] <= solutions[i][1] and
                    (solutions[j][0] < solutions[i][0] or 
                     solutions[j][1] < solutions[i][1])):
                    is_pareto[i] = False
                    break
    
    return [i for i in range(n) if is_pareto[i]]

def compute_hypervolume(pareto_front: List[Tuple[float, float]], 
                       reference_point: Tuple[float, float]) -> float:
    """
    Tính hypervolume cho Pareto front
    Simple 2D implementation
    """
    if not pareto_front:
        return 0.0
    
    # Sort by first objective
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    
    hypervolume = 0.0
    prev_x = reference_point[0]
    
    for point in sorted_front:
        width = prev_x - point[0]
        height = reference_point[1] - point[1]
        hypervolume += width * height
        prev_x = point[0]
    
    return hypervolume

def normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    """
    Normalize objectives về [0, 1]
    """
    min_vals = objectives.min(axis=0)
    max_vals = objectives.max(axis=0)
    
    # Tránh chia cho 0
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    normalized = (objectives - min_vals) / range_vals
    return normalized

def weighted_sum_aggregation(objectives: np.ndarray, 
                             weights: np.ndarray) -> np.ndarray:
    """
    Weight aggregation cho multi-objective
    """
    if weights.sum() != 1.0:
        weights = weights / weights.sum()
    
    return np.dot(objectives, weights)

def create_weight_vectors(n_weights: int, n_objectives: int = 2) -> np.ndarray:
    """
    Tạo weight vectors đều đặn cho multi-objective
    """
    if n_objectives == 2:
        weights = []
        for i in range(n_weights):
            w1 = i / (n_weights - 1)
            w2 = 1 - w1
            weights.append([w1, w2])
        return np.array(weights)
    else:
        raise NotImplementedError("Only 2 objectives supported")

def log_metrics_to_file(metrics: Dict, filepath: str, mode='a'):
    """
    Log metrics vào file
    """
    with open(filepath, mode) as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

def format_time(seconds: float) -> str:
    """
    Format seconds thành readable string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def validate_route(route: List[int], 
                  n_customers: int,
                  visited: np.ndarray) -> bool:
    """
    Validate route có hợp lệ không
    """
    if not route:
        return True
    
    # Check range
    if any(c < 1 or c > n_customers for c in route):
        return False
    
    # Check duplicates
    if len(route) != len(set(route)):
        return False
    
    # Check visited
    for c in route:
        if visited[c - 1]:
            return False
    
    return True

def calculate_route_statistics(routes: List[List[int]], 
                               distance_matrix: np.ndarray) -> Dict:
    """
    Tính thống kê cho routes
    """
    stats = {
        'n_routes': len(routes),
        'total_distance': 0.0,
        'avg_route_length': 0.0,
        'max_route_length': 0,
        'min_route_length': float('inf')
    }
    
    route_lengths = []
    
    for route in routes:
        if not route:
            continue
        
        full_route = [0] + route + [0]
        distance = 0.0
        
        for i in range(len(full_route) - 1):
            distance += distance_matrix[full_route[i]][full_route[i + 1]]
        
        stats['total_distance'] += distance
        route_lengths.append(len(route))
    
    if route_lengths:
        stats['avg_route_length'] = np.mean(route_lengths)
        stats['max_route_length'] = max(route_lengths)
        stats['min_route_length'] = min(route_lengths)
    
    return stats

def seed_everything(seed: int = 42):
    """
    Set random seed cho reproducibility
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def compute_wait_and_makespan(env_info: Dict, env: ParallelVRPDEnv):
    """
    Giả sử sau khi episode kết thúc:
    - env.service_time đã được fill
    - env.current_time là makespan
    """
    service_time = getattr(env, "service_time", None)
    if service_time is None:
        return None, None

    total_wait = float(np.nansum(service_time))
    makespan = float(env.current_time)
    return total_wait, makespan


class EarlyStopping:
    """
    Early stopping cho training
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Returns True nếu nên stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def print_model_summary(model):
    """
    In summary của model
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    print("\nPolicy Network:")
    print(model.policy)
    print("="*60 + "\n")

def export_results_to_json(results: Dict, filepath: str):
    """
    Export kết quả ra JSON
    """
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    converted_results = convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=4)
    
    print(f"Results exported to {filepath}")

def compare_solutions(solution1: Dict, solution2: Dict) -> Dict:
    """
    So sánh 2 solutions
    """
    comparison = {
        'completion_time_diff': solution1['completion_time'] - solution2['completion_time'],
        'waiting_time_diff': solution1['waiting_time'] - solution2['waiting_time'],
        'total_diff': (solution1['completion_time'] + solution1['waiting_time']) - 
                     (solution2['completion_time'] + solution2['waiting_time'])
    }
    
    # Determine winner
    if comparison['total_diff'] < 0:
        comparison['winner'] = 'solution1'
    elif comparison['total_diff'] > 0:
        comparison['winner'] = 'solution2'
    else:
        comparison['winner'] = 'tie'
    
    return comparison

def generate_random_instance(n_customers: int, 
                            area_size: float = 5000.0,
                            demand_range: Tuple[float, float] = (0.01, 0.2),
                            truck_only_prob: float = 0.3) -> Dict:
    """
    Generate random instance cho testing
    """
    np.random.seed()
    
    customers = []
    for i in range(n_customers):
        customer = {
            'id': i + 1,
            'x': np.random.uniform(-area_size, area_size),
            'y': np.random.uniform(-area_size, area_size),
            'demand': np.random.uniform(*demand_range),
            'only_truck': np.random.random() < truck_only_prob,
            'service_time_truck': 60,
            'service_time_drone': 30
        }
        customers.append(customer)
    
    return {
        'n_customers': n_customers,
        'customers': customers,
        'num_trucks': 1,
        'num_drones': 1
    }

def save_instance_to_file(instance: Dict, filepath: str):
    """
    Lưu instance ra file txt format
    """
    with open(filepath, 'w') as f:
        f.write(f"number_staff {instance['num_trucks']}\n")
        f.write(f"number_drone {instance['num_drones']}\n")
        f.write(f"droneLimitationFightTime(s) 3600\n")
        f.write(f"Customers {instance['n_customers']}\n")
        f.write("Coordinate X\t\tCoordinate Y\t\tDemand\t\tOnlyServicedByStaff\t\tServiceTimeByTruck(s)\t\tServiceTimeByDrone(s)\n")
        
        for customer in instance['customers']:
            f.write(f"{customer['x']}\t{customer['y']}\t{customer['demand']}\t")
            f.write(f"{int(customer['only_truck'])}\t{customer['service_time_truck']}\t{customer['service_time_drone']}\n")
    
    print(f"Instance saved to {filepath}")