"""
Evaluation Script for Trained Models
"""
import argparse
import os
import numpy as np
from stable_baselines3 import PPO

from config import ConfigManager
from data_loader import DataLoader
from environment import ParallelVRPDEnv
from visualizer import VRPDVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VRPD models')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--data', type=str, required=True, help='Path to data')
    parser.add_argument('--drone_config', type=str, default='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json')
    parser.add_argument('--truck_config', type=str, default='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json')
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./eval_results/')
    return parser.parse_args()

def evaluate_model(model, env, n_episodes=10):
    all_metrics = {
        'completion_time': [],
        'waiting_time': [],
        'total_objective': [],
        'n_truck_routes': [],
        'n_drone_routes': [],
        'feasible': []
    }
    
    all_routes = []
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        truck_routes = info['truck_routes']
        drone_routes = info['drone_routes']
        
        calc = env.calculator
        times, wait, feasible = [], 0, True
        
        for route in truck_routes:
            if route:
                t, w = calc.calculate_truck_time(route)
                times.append(t)
                wait += w
        
        for route in drone_routes:
            if route:
                demands = [env.demands[c-1] for c in route]
                if not calc.check_drone_energy(route, demands):
                    feasible = False
                if not calc.check_drone_capacity(route, demands):
                    feasible = False
                
                t, w, _ = calc.calculate_drone_time(route)
                times.append(t)
                wait += w
        
        completion = max(times) if times else 0
        
        all_metrics['completion_time'].append(completion)
        all_metrics['waiting_time'].append(wait)
        all_metrics['total_objective'].append(completion + wait)
        all_metrics['n_truck_routes'].append(len([r for r in truck_routes if r]))
        all_metrics['n_drone_routes'].append(len([r for r in drone_routes if r]))
        all_metrics['feasible'].append(feasible)
        
        all_routes.append({
            'truck_routes': truck_routes,
            'drone_routes': drone_routes
        })
        
        print(f"  Completion: {completion:.2f}s")
        print(f"  Waiting: {wait:.2f}s")
        print(f"  Feasible: {feasible}")
    
    # Statistics
    stats = {}
    for key in all_metrics:
        if key != 'feasible':
            stats[f'{key}_mean'] = np.mean(all_metrics[key])
            stats[f'{key}_std'] = np.std(all_metrics[key])
            stats[f'{key}_min'] = np.min(all_metrics[key])
            stats[f'{key}_max'] = np.max(all_metrics[key])
        else:
            stats['feasibility_rate'] = np.mean(all_metrics[key])
    
    return stats, all_routes, all_metrics

def print_statistics(stats):
    print("\n" + "="*60)
    print("EVALUATION STATISTICS")
    print("="*60)
    
    print("\nCompletion Time:")
    print(f"  Mean: {stats['completion_time_mean']:.2f}s")
    print(f"  Std:  {stats['completion_time_std']:.2f}s")
    print(f"  Min:  {stats['completion_time_min']:.2f}s")
    print(f"  Max:  {stats['completion_time_max']:.2f}s")
    
    print("\nWaiting Time:")
    print(f"  Mean: {stats['waiting_time_mean']:.2f}s")
    print(f"  Std:  {stats['waiting_time_std']:.2f}s")
    
    print("\nTotal Objective:")
    print(f"  Mean: {stats['total_objective_mean']:.2f}s")
    
    print(f"\nFeasibility: {stats['feasibility_rate']*100:.1f}%")
    print("="*60)

def save_results(stats, all_routes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'statistics.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION STATISTICS\n")
        f.write("="*60 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    with open(os.path.join(save_dir, 'routes.txt'), 'w') as f:
        for i, routes in enumerate(all_routes):
            f.write(f"\n=== Episode {i+1} ===\n")
            f.write(f"Truck: {routes['truck_routes']}\n")
            f.write(f"Drone: {routes['drone_routes']}\n")

def main():
    args = parse_args()
    
    print("="*60)
    print("Model Evaluation for VRPD")
    print("="*60)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load
    config_manager = ConfigManager(
        drone_config_path=args.drone_config,
        truck_config_path=args.truck_config
    )
    
    # data_loader = DataLoader(args.data)
    # print(data_loader.summary())
    
    # env = ParallelVRPDEnv(data_loader, config_manager)

    # Load data
    data_loader = DataLoader(args.data)
    print(data_loader.summary())
    x_coords, y_coords = data_loader.get_coordinates()
    coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    demands = data_loader.get_customer_demands().astype(np.float32)

    truck_only_ids = data_loader.get_truck_only_customers()
    truck_only = np.zeros_like(demands, dtype=bool)
    for idx in truck_only_ids:
        truck_only[idx - 1] = True

    capacity = getattr(config_manager, "truck_capacity", 50.0)

    # Create environment (matching the new constructor)
    env = ParallelVRPDEnv(
        n_customers=len(demands),
        coords=coords,
        demands=demands,
        truck_only=truck_only,
        capacity=capacity,
        calculator=None,
        use_drone=True,
        debug=False,
    )

    
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, env=env)
    
    # Evaluate
    print(f"\nEvaluating {args.n_episodes} episodes...")
    stats, all_routes, all_metrics = evaluate_model(
        model, env, n_episodes=args.n_episodes
    )
    
    print_statistics(stats)
    
    print(f"\nSaving results to {args.save_dir}...")
    save_results(stats, all_routes, args.save_dir)
    
    # Visualize
    if args.visualize:
        print("\nCreating visualizations...")
        visualizer = VRPDVisualizer(data_loader, save_dir=args.save_dir)
        
        best_idx = np.argmin(all_metrics['total_objective'])
        best_routes = all_routes[best_idx]
        
        visualizer.plot_routes(
            best_routes['truck_routes'],
            best_routes['drone_routes'],
            title=f"Best Route (Episode {best_idx+1})",
            save_path=os.path.join(args.save_dir, 'best_route.png')
        )
        
        visualizer.plot_metrics_history(
            all_metrics,
            save_path=os.path.join(args.save_dir, 'metrics.png')
        )
        
        print(f"Saved to {args.save_dir}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)

if __name__ == "__main__":
    main()