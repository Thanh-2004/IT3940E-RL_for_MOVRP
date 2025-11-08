"""
Train WA-PPO (MaskablePPO + Attention) on real VRPD data
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
from wa_ppo import WAPPOTrainer, mask_fn
from visualizer import VRPDVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train WA-PPO (Maskable) for VRPD")
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
    parser.add_argument(
        "--total_timesteps", type=int, default=500_000, help="Total training steps"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/", help="Directory for training logs"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models/", help="Directory for saving models"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=10000, help="Evaluation frequency"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def setup_directories(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("./results/", exist_ok=True)


def evaluate_model(model, eval_env, max_steps=10000, debug=False):
    """
    ✅ FIXED: Proper evaluation with action masking
    """
    print("\n" + "="*60)
    print("🔍 EVALUATING MODEL")
    print("="*60)
    
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    invalid_actions = 0
    phase_switches = 0
    last_phase = info.get('phase', 'drone')
    
    while not done and step_count < max_steps:
        # ✅ Get action mask from environment
        action_mask = info["action_mask"]
        
        # ✅ Predict with action mask
        action, _ = model.predict(
            obs, 
            deterministic=True,
            action_masks=action_mask  # ⭐ KEY FIX
        )
        action = int(action)
        
        # ✅ Validate action
        if not action_mask[action]:
            print(f"\n⚠️ WARNING at step {step_count}:")
            print(f"   Model predicted invalid action: {action}")
            print(f"   Valid actions: {np.where(action_mask)[0][:20]}...")
            
            # Force a valid action
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = int(valid_actions[0])  # Take first valid action
                print(f"   Using valid action: {action}")
                invalid_actions += 1
            else:
                print(f"   ❌ NO VALID ACTIONS - Episode will terminate")
                break
        
        # Track phase switches
        current_phase = info.get('phase', 'drone')
        if current_phase != last_phase:
            phase_switches += 1
            print(f"\n🔄 PHASE SWITCH #{phase_switches}: {last_phase} → {current_phase}")
            print(f"   Customers left: {info.get('customers_left', '?')}")
            last_phase = current_phase
        
        # Debug output
        if debug and step_count % 20 == 0:
            print(f"\n[Step {step_count}]")
            print(f"  Phase: {info['phase']}")
            print(f"  Action: {action}")
            print(f"  Customers: {info.get('customers_visited', 0)}/{eval_env.env.N}")
            print(f"  Reward so far: {total_reward:.2f}")
        
        # Execute step
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
        
        # Check for errors
        if 'error' in info:
            print(f"\n❌ ERROR at step {step_count}: {info['error']}")
            break
    
    # ✅ Get final info from unwrapped environment
    inner_env = eval_env.env
    
    # Detailed results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total steps: {step_count}")
    print(f"Invalid actions predicted: {invalid_actions}")
    print(f"Phase switches: {phase_switches}")
    print(f"Customers visited: {info.get('customers_visited', 0)}/{inner_env.N}")
    print(f"Completion rate: {info.get('completion_rate', 0)*100:.1f}%")
    print(f"Final phase: {info.get('phase', '?')}")
    
    # ✅ Validate routes for duplicates
    print("\n" + "="*60)
    print("🔍 ROUTE VALIDATION")
    print("="*60)
    
    drone_customers = set()
    duplicate_in_drone = []
    for d_idx, route in enumerate(info['drone_routes']):
        for cid in route:
            if cid in drone_customers:
                duplicate_in_drone.append((d_idx, cid))
            drone_customers.add(cid)
    
    if duplicate_in_drone:
        print(f"⚠️ Duplicates in drone routes: {duplicate_in_drone}")
    else:
        print(f"✅ No duplicates in drone routes")
    
    truck_customers = set()
    duplicate_in_truck = []
    for cid in info['truck_routes']:
        if cid in truck_customers:
            duplicate_in_truck.append(cid)
        truck_customers.add(cid)
    
    if duplicate_in_truck:
        print(f"⚠️ Duplicates in truck route: {set(duplicate_in_truck)}")
    else:
        print(f"✅ No duplicates in truck route")
    
    # Check drone-truck overlap
    overlap = drone_customers.intersection(truck_customers)
    if overlap:
        print(f"⚠️ Customers served by BOTH drone and truck: {overlap}")
    else:
        print(f"✅ No overlap between drone and truck routes")
    
    # Check missing customers
    all_customers = drone_customers.union(truck_customers)
    missing = set(range(inner_env.N)) - all_customers
    if missing:
        print(f"⚠️ Customers NOT visited: {len(missing)} customers")
        if len(missing) <= 20:
            print(f"   Missing: {sorted(missing)}")
    else:
        print(f"✅ All customers visited")
    
    # Route summary
    print("\n" + "="*60)
    print("📦 ROUTE SUMMARY")
    print("="*60)
    print(f"Drone routes ({len(info['drone_routes'])} drones):")
    for i, route in enumerate(info['drone_routes']):
        if len(route) > 0:
            preview = route[:10]
            suffix = f"... +{len(route)-10} more" if len(route) > 10 else ""
            print(f"  Drone {i}: {preview} {suffix} (total: {len(route)})")
        else:
            print(f"  Drone {i}: [] (empty)")
    
    truck_route = info['truck_routes']
    if len(truck_route) > 0:
        preview = truck_route[:20]
        suffix = f"... +{len(truck_route)-20} more" if len(truck_route) > 20 else ""
        print(f"\nTruck route: {preview} {suffix} (total: {len(truck_route)})")
    else:
        print(f"\nTruck route: [] (empty)")
    
    return {
        'total_reward': total_reward,
        'steps': step_count,
        'completion_rate': info.get('completion_rate', 0),
        'truck_routes': info['truck_routes'],
        'drone_routes': info['drone_routes'],
        'customers_visited': info.get('customers_visited', 0),
        'invalid_actions': invalid_actions,
        'has_duplicates': len(duplicate_in_drone) > 0 or len(duplicate_in_truck) > 0 or len(overlap) > 0,
        'missing_customers': len(missing),
    }


def train_single_weight(args, config_manager, data_loader, weight_id, w_comp, w_wait):
    print(f"\n{'='*60}")
    print(f"Training with weights: completion={w_comp}, waiting={w_wait}")
    print(f"{'='*60}\n")

    # Load dataset
    x_coords, y_coords = data_loader.get_coordinates()
    coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    demands = data_loader.get_customer_demands().astype(np.float32)

    truck_only_ids = data_loader.get_truck_only_customers()
    truck_only = np.zeros_like(demands, dtype=bool)
    for idx in truck_only_ids:
        truck_only[idx - 1] = True

    capacity = getattr(config_manager, "truck_capacity", 50.0)
    env_params = config_manager.get_env_params()
    
    # ✅ Add max_steps based on problem size
    env_params['max_steps'] = len(demands) * 10
    
    # ✅ Remove debug if it exists in env_params to avoid conflict
    env_params.pop('debug', None)

    # Create training environment (WITHOUT ActionMasker - trainer will add it)
    print(f"Creating training environment for {len(demands)} customers...")
    env = ParallelVRPDEnv(
        n_customers=len(demands),
        coords=coords,
        demands=demands,
        truck_only=truck_only,
        calculator=None,
        debug=False,  # Disable during training for speed
        **env_params,
    )

    # Create evaluation environment with debug enabled
    print(f"Creating evaluation environment (debug={args.debug})...")
    eval_env = ParallelVRPDEnv(
        n_customers=len(demands),
        coords=coords,
        demands=demands,
        truck_only=truck_only,
        calculator=None,
        debug=args.debug,  # Enable debug in eval if requested
        **env_params,
    )

    # ✅ Wrap with ActionMasker
    env = ActionMasker(env, mask_fn)
    eval_env = ActionMasker(eval_env, mask_fn)

    # Create trainer
    wa_trainer = WAPPOTrainer(env, eval_env=eval_env)

    print(f"Starting training for {args.total_timesteps:,} timesteps ...\n")
    wa_trainer.learn(total_timesteps=args.total_timesteps)

    # Save model
    model_path = os.path.join(args.save_dir, f"maskppo_{weight_id}.zip")
    wa_trainer.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # ✅ Evaluate with proper action masking
    eval_results = evaluate_model(
        model=wa_trainer.model,
        eval_env=eval_env,
        max_steps=env_params['max_steps'],
        debug=args.debug
    )

    # Visualization
    if eval_results['completion_rate'] > 0.5:  # Only visualize if reasonable solution
        print("\n📊 Generating visualization...")
        vis = VRPDVisualizer(data_loader, save_dir="./results/")
        vis.plot_routes(
            eval_results['truck_routes'],
            eval_results['drone_routes'],
            title=f"VRPD Routes - {weight_id} (completion: {eval_results['completion_rate']*100:.1f}%)",
            save_path=f"./results/routes_{weight_id}.png",
        )
        print(f"✅ Visualization saved to ./results/routes_{weight_id}.png")
    else:
        print(f"\n⚠️ Skipping visualization (completion rate too low: {eval_results['completion_rate']*100:.1f}%)")

    return wa_trainer.model, eval_results


def main():
    args = parse_args()
    setup_directories(args)

    print("=" * 60)
    print("WA-PPO (Maskable) Training for Parallel VRPD")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    print("\nLoading configuration ...")
    config_manager = ConfigManager(
        drone_config_path=args.drone_config,
        truck_config_path=args.truck_config,
    )
    print(config_manager)

    # Load data
    print("\nLoading dataset ...")
    data_loader = DataLoader(args.data)
    print(data_loader.summary())

    # Train with single weight
    weights = (0.5, 0.5)
    weight_id = f"w_{weights[0]}_{weights[1]}"
    model, eval_results = train_single_weight(args, config_manager, data_loader, weight_id, *weights)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Results:")
    print(f"  Completion rate: {eval_results['completion_rate']*100:.1f}%")
    print(f"  Total reward: {eval_results['total_reward']:.2f}")
    print(f"  Customers visited: {eval_results['customers_visited']}")
    print(f"  Invalid actions: {eval_results['invalid_actions']}")
    print(f"  Has duplicates: {eval_results['has_duplicates']}")
    print(f"  Missing customers: {eval_results['missing_customers']}")
    print("=" * 60)


if __name__ == "__main__":
    main()