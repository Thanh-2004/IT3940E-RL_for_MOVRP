"""
Train WA-PPO (MaskablePPO + Attention) on VRPD (Parallel Vehicles)
with Multi-Objective Optimization: Service Time + Waiting Time
"""

import os
import argparse
import numpy as np
from datetime import datetime
from sb3_contrib.common.wrappers import ActionMasker

# Project modules
from config import ConfigManager
from data_loader import DataLoader
from environment import ParallelVRPDEnv
from route_calculator import TimeDistanceCalculator 
from wa_ppo import WAPPOTrainer, mask_fn
from visualizer import VRPDVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train WA-PPO for Parallel VRPD")
    parser.add_argument(
        "--data",
        type=str,
        default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/200.40.4.txt",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--drone_config",
        type=str,
        default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json",
        help="Path to drone config JSON",
    )
    parser.add_argument(
        "--truck_config",
        type=str,
        default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json",
        help="Path to truck config JSON",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Directory for training logs")
    parser.add_argument("--save_dir", type=str, default="./models/", help="Directory for saving models")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Max steps configuration
    parser.add_argument("--train_max_steps_multiplier", type=int, default=10,
                       help="Training max_steps = n_customers × this value")
    parser.add_argument("--eval_max_steps_multiplier", type=int, default=50,
                       help="Evaluation max_steps = n_customers × this value")
    
    # Objective weights (MUST sum to 1.0)
    parser.add_argument("--w_service", type=float, default=0.5, 
                       help="Weight for total service time (makespan)")
    parser.add_argument("--w_wait", type=float, default=0.5, 
                       help="Weight for total waiting time")
    
    # Reward shaping
    parser.add_argument("--enable_shaping", action="store_true", 
                       help="Enable reward shaping for training")
    parser.add_argument("--step_penalty", type=float, default=0.1)
    parser.add_argument("--progress_reward", type=float, default=1.0)
    parser.add_argument("--idle_penalty", type=float, default=0.5)
    
    return parser.parse_args()


def setup_directories(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("./results/", exist_ok=True)


def evaluate_model(model, eval_env, n_customers, debug=True):
    """Evaluate model and log objectives"""
    print("\n" + "=" * 70)
    print("📊 EVALUATING MODEL")
    print("=" * 70)

    # Get max_steps from environment
    env_inner = eval_env.env if hasattr(eval_env, 'env') else eval_env
    max_steps = env_inner.max_steps
    
    print(f"Max steps allowed: {max_steps} (n_customers={n_customers})")

    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    invalid_actions = 0

    while not done and step_count < max_steps:
        mask = info.get("action_mask", None)
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        
        if mask is not None and not mask[action]:
            valid = np.where(mask)[0]
            if len(valid) == 0:
                print("❌ No valid actions left.")
                break
            action = int(valid[0])
            invalid_actions += 1

        obs, reward, terminated, truncated, info = eval_env.step(int(action))
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if debug and step_count % 100 == 0:
            left = info.get('customers_left', '?')
            completion = info.get('completion_rate', 0.0) * 100
            print(f"[{step_count}/{max_steps}] Left={left}, Completion={completion:.1f}%, "
                  f"Time={info.get('time', 0):.1f}s")

    # Extract objectives
    env_inner = eval_env.env if hasattr(eval_env, 'env') else eval_env
    
    # PRIMARY OBJECTIVES
    total_service_time = info.get("total_service_time", float(env_inner.current_time))
    total_waiting_time = info.get("total_waiting_time", float(np.nansum(env_inner.service_time)))
    
    # Secondary metrics
    total_travel_distance = info.get("total_travel_distance", 0.0)
    completion_rate = info.get("completion_rate", 0.0)

    print("\n" + "=" * 70)
    print("🎯 MULTI-OBJECTIVE OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\n📏 PRIMARY OBJECTIVES (to minimize):")
    print(f"  • Total Service Time (Makespan): {total_service_time:.2f}s")
    print(f"  • Total Waiting Time: {total_waiting_time:.2f}s")
    
    weights = info.get("weights", {})
    w_service = weights.get("w_service", 0.5)
    w_wait = weights.get("w_wait", 0.5)
    
    weighted_objective = w_service * total_service_time + w_wait * total_waiting_time
    print(f"\n🎲 Weighted Objective (w_service={w_service:.2f}, w_wait={w_wait:.2f}):")
    print(f"  • {weighted_objective:.2f}")
    
    print(f"\n📊 SECONDARY METRICS:")
    print(f"  • Completion rate: {completion_rate*100:.1f}%")
    print(f"  • Total travel distance: {total_travel_distance:.2f}m")
    print(f"  • Steps taken: {step_count}/{max_steps}")
    print(f"  • Invalid actions: {invalid_actions}")
    print(f"  • Total reward (with shaping): {total_reward:.2f}")

    print(f"\n🚚 ROUTES:")
    truck_routes = info.get("truck_routes", [])
    for i, route in enumerate(truck_routes):
        print(f"  • Truck-{i} route: {route}")
    
    drone_routes = info.get("drone_routes", [])
    for i, r in enumerate(drone_routes):
        print(f"  • Drone-{i} route: {r}")

    print("=" * 70)

    return {
        # Primary objectives
        "total_service_time": total_service_time,
        "total_waiting_time": total_waiting_time,
        "weighted_objective": weighted_objective,
        # Secondary metrics
        "total_reward": total_reward,
        "completion_rate": completion_rate,
        "total_travel_distance": total_travel_distance,
        "step_count": step_count,
        "invalid_actions": invalid_actions,
        # Routes - 
        "truck_routes": truck_routes,
        "drone_routes": drone_routes,
        # Weights
        "w_service": w_service,
        "w_wait": w_wait,
    }


def train_single_weight(args, config_manager, data_loader, weight_id):
    """Train model with specific weight configuration"""
    
    # Normalize weights to sum to 1.0
    weight_sum = args.w_service + args.w_wait
    w_service = args.w_service / weight_sum
    w_wait = args.w_wait / weight_sum
    
    print(f"\n{'=' * 70}")
    print(f"🎯 MULTI-OBJECTIVE VRPD TRAINING")
    print(f"{'=' * 70}")
    print(f"\nObjective Weights (sum=1.0):")
    print(f"  • w_service (makespan): {w_service:.3f}")
    print(f"  • w_wait (waiting):     {w_wait:.3f}")
    
    if args.enable_shaping:
        print(f"\nReward Shaping Enabled:")
        print(f"  • Step penalty: {args.step_penalty}")
        print(f"  • Progress reward: {args.progress_reward}")
        print(f"  • Idle penalty: {args.idle_penalty}")
    print(f"{'=' * 70}\n")

    # Load dataset
    x_coords, y_coords = data_loader.get_coordinates()
    coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    demands = data_loader.get_customer_demands().astype(np.float32)

    truck_only_ids = data_loader.get_truck_only_customers()
    truck_only = np.zeros_like(demands, dtype=bool)
    for idx in truck_only_ids:
        if 1 <= idx <= len(demands):
            truck_only[idx - 1] = True

    n_customers = len(demands)
    
    # Use correct capacity from config
    capacity = config_manager.truck.capacity
    
    # Get number of vehicles from config
    n_trucks = config_manager.problem.num_trucks
    n_drones = config_manager.problem.num_drones
    
    print(f"Dataset information:")
    print(f"  • Customers: {n_customers}")
    print(f"  • Trucks: {n_trucks}")  
    print(f"  • Drones: {n_drones}")
    print(f"  • Truck capacity: {capacity}kg")
    print(f"  • Drone battery: {config_manager.drone.battery_power}J")
    print(f"  • Truck-only customers: {np.sum(truck_only)}")
    
    # Create TimeDistanceCalculator
    print(f"\n🧮 Creating TimeDistanceCalculator...")
    
    drone_cfg = {
        "takeoff_speed": config_manager.drone.takeoff_speed,
        "cruise_speed": config_manager.drone.cruise_speed,
        "landing_speed": config_manager.drone.landing_speed,
        "cruise_alt": config_manager.drone.cruise_alt,
        "capacity": config_manager.drone.capacity,
        "battery_power": config_manager.drone.battery_power,
        "beta": config_manager.drone.beta,
        "gamma": config_manager.drone.gamma,
    }
    
    # Load truck config with time-based speed
    import json
    with open(args.truck_config, 'r') as f:
        truck_cfg = json.load(f)
    
    # Add capacity if not in config
    if "capacity" not in truck_cfg:
        truck_cfg["capacity"] = capacity
    
    calculator = TimeDistanceCalculator(
        coords=coords,
        drone_config=drone_cfg,
        truck_config=truck_cfg,
        service_time_drone=config_manager.problem.service_time_drone,
        service_time_truck=config_manager.problem.service_time_truck,
    )
    
    # Print calculator summary
    calculator.print_summary()
    
    # Max steps configuration
    train_max_steps = n_customers * args.train_max_steps_multiplier
    eval_max_steps = n_customers * args.eval_max_steps_multiplier
    
    print(f"\n⚙️  Environment Configuration:")
    print(f"  • Training max_steps: {train_max_steps} ({args.train_max_steps_multiplier}x)")
    print(f"  • Evaluation max_steps: {eval_max_steps} ({args.eval_max_steps_multiplier}x)")

    # Common environment parameters - 
    common_params = {
        "n_customers": n_customers,
        "coords": coords,
        "demands": demands,
        "truck_only": truck_only,
        "capacity": capacity,
        "calculator": calculator, 
        "use_drone": n_drones > 0,
        "n_drones": n_drones,
        "n_trucks": n_trucks,  
        "drone_battery": config_manager.drone.battery_power,
        "w_service": w_service,
        "w_wait": w_wait,
        "enable_shaping": args.enable_shaping,
        "step_penalty": args.step_penalty,
        "progress_reward": args.progress_reward,
        "idle_penalty": args.idle_penalty,
        "include_weight_in_obs": True,
    }

    # Training environment (shorter max_steps for faster training)
    train_params = common_params.copy()
    train_params["max_steps"] = train_max_steps
    train_params["debug"] = False
    
    print(f"\n🏭 Creating training environment...")
    env = ParallelVRPDEnv(**train_params)
    
    # Evaluation environment (longer max_steps for complete solutions)
    eval_params = common_params.copy()
    eval_params["max_steps"] = eval_max_steps
    eval_params["enable_shaping"] = False  # Pure objectives for evaluation
    eval_params["debug"] = args.debug
    
    print(f"🔬 Creating evaluation environment...")
    eval_env = ParallelVRPDEnv(**eval_params)

    # Wrap with ActionMasker
    env = ActionMasker(env, mask_fn)
    eval_env = ActionMasker(eval_env, mask_fn)

    # Trainer
    print("\n🚀 Starting training...")
    print(f"  • Total timesteps: {args.total_timesteps:,}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Batch size: {args.batch_size}")
    
    trainer = WAPPOTrainer(
        env, 
        eval_env=eval_env, 
        n_customers=n_customers,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )
    
    trainer.learn(total_timesteps=args.total_timesteps)

    # Save model
    model_path = os.path.join(args.save_dir, f"maskppo_{weight_id}.zip")
    trainer.save(model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Evaluate
    print("\n" + "=" * 70)
    print("🔍 FINAL EVALUATION")
    print("=" * 70)
    eval_results = evaluate_model(trainer.model, eval_env, n_customers, debug=True)

    # Visualize if good completion
    if eval_results["completion_rate"] > 0.5:
        print(f"\n📊 Generating visualization...")
        vis = VRPDVisualizer(data_loader, save_dir="./results/")
        title = (f"Routes (completion={eval_results['completion_rate']*100:.1f}%)\n"
                f"Service Time={eval_results['total_service_time']:.1f}s, "
                f"Wait Time={eval_results['total_waiting_time']:.1f}s")
        
        # Pass all truck routes
        vis.plot_routes(
            eval_results["truck_routes"], 
            eval_results["drone_routes"],
            title=title,
            save_path=f"./results/routes_{weight_id}.png",
        )
        print(f"✅ Visualization saved to ./results/routes_{weight_id}.png")
    else:
        print(f"\n⚠️  Skipping visualization (low completion rate: {eval_results['completion_rate']*100:.1f}%)")
        print(f"   Consider:")
        print(f"   • Increasing eval_max_steps_multiplier (current: {args.eval_max_steps_multiplier}x)")
        print(f"   • Training longer (current: {args.total_timesteps:,} timesteps)")
        print(f"   • Adjusting reward shaping parameters")

    return trainer.model, eval_results


def main():
    args = parse_args()
    setup_directories(args)

    print("=" * 70)
    print("🤖 WA-PPO for Multi-Objective Parallel VRPD")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n📋 Loading configuration...")

    config_manager = ConfigManager(
        data_file=args.data,
        drone_config_path=args.drone_config,
        truck_config_path=args.truck_config
    )
    print(config_manager)

    print("\n📦 Loading dataset...")
    data_loader = DataLoader(args.data)
    print(data_loader.summary())
    
    # Create weight ID for file naming
    weight_id = f"wS{args.w_service:.2f}_wW{args.w_wait:.2f}".replace(".", "p")
    
    # Train
    model, eval_results = train_single_weight(args, config_manager, data_loader, weight_id)

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🎯 FINAL OBJECTIVES:")
    print(f"  • Total Service Time: {eval_results['total_service_time']:.2f}s")
    print(f"  • Total Waiting Time: {eval_results['total_waiting_time']:.2f}s")
    print(f"  • Weighted Objective: {eval_results['weighted_objective']:.2f}")
    print(f"  • Completion Rate: {eval_results['completion_rate']*100:.1f}%")
    print(f"  • Steps Used: {eval_results['step_count']}")
    print("=" * 70)

    # Save results to file
    results_file = os.path.join("./results/", f"results_{weight_id}.txt")
    with open(results_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VRPD Multi-Objective Optimization Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Dataset: {os.path.basename(args.data)}\n")
        f.write(f"  Drone config: {config_manager.problem.drone_config_id}\n")
        f.write(f"  Customers: {config_manager.problem.num_customers}\n")
        f.write(f"  Trucks: {config_manager.problem.num_trucks}\n")
        f.write(f"  Drones: {config_manager.problem.num_drones}\n\n")
        
        f.write(f"Objective Weights:\n")
        f.write(f"  w_service (makespan): {eval_results['w_service']:.3f}\n")
        f.write(f"  w_wait (waiting):     {eval_results['w_wait']:.3f}\n\n")
        
        f.write(f"Primary Objectives:\n")
        f.write(f"  Total Service Time: {eval_results['total_service_time']:.2f}s\n")
        f.write(f"  Total Waiting Time: {eval_results['total_waiting_time']:.2f}s\n")
        f.write(f"  Weighted Objective: {eval_results['weighted_objective']:.2f}\n\n")
        
        f.write(f"Secondary Metrics:\n")
        f.write(f"  Completion Rate: {eval_results['completion_rate']*100:.1f}%\n")
        f.write(f"  Total Travel Distance: {eval_results['total_travel_distance']:.2f}m\n")
        f.write(f"  Steps: {eval_results['step_count']}\n")
        f.write(f"  Invalid Actions: {eval_results['invalid_actions']}\n\n")
        
        f.write(f"Routes:\n")
        # Write all truck routes
        for i, route in enumerate(eval_results['truck_routes']):
            f.write(f"  Truck-{i}: {route}\n")
        for i, r in enumerate(eval_results['drone_routes']):
            f.write(f"  Drone-{i}: {r}\n")
        
        f.write(f"\nTraining Configuration:\n")
        f.write(f"  Total timesteps: {args.total_timesteps:,}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Enable shaping: {args.enable_shaping}\n")
    
    print(f"\n📄 Results saved to {results_file}")
    
    # Save model info
    model_info_file = os.path.join(args.save_dir, f"info_{weight_id}.txt")
    with open(model_info_file, "w") as f:
        f.write(f"Model: maskppo_{weight_id}.zip\n")
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Completion Rate: {eval_results['completion_rate']*100:.1f}%\n")
        f.write(f"Weighted Objective: {eval_results['weighted_objective']:.2f}\n")
    
    print(f"📄 Model info saved to {model_info_file}")


if __name__ == "__main__":
    main()