# """
# Main Training Script for WA-PPO VRPD
# Script chính để train model
# """
# import argparse
# import os
# import numpy as np
# from datetime import datetime

# from config import ConfigManager
# from data_loader import DataLoader
# from environment import ParallelVRPDEnv
# from wa_ppo import WAPPO, MultiObjectiveCallback
# from visualizer import VRPDVisualizer
# from sb3_contrib.common.wrappers import ActionMasker


# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Train WA-PPO for VRPD')
    
#     # Data paths
#     parser.add_argument('--data', type=str, default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/200.40.4.txt",
#                        help='Path to data file')
#     parser.add_argument('--drone_config', type=str, default='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json',
#                        help='Path to drone config')
#     parser.add_argument('--truck_config', type=str, default='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json',
#                        help='Path to truck config')
    
#     # Training hyperparameters
#     parser.add_argument('--lr', type=float, default=3e-4,
#                        help='Learning rate')
#     parser.add_argument('--n_steps', type=int, default=2048,
#                        help='Number of steps per update')
#     parser.add_argument('--batch_size', type=int, default=64,
#                        help='Batch size')
#     parser.add_argument('--n_epochs', type=int, default=10,
#                        help='Number of epochs')
#     parser.add_argument('--total_timesteps', type=int, default=500000,
#                        help='Total training timesteps')
    
#     # Multi-objective weights
#     parser.add_argument('--weights', type=str, default='0.5,0.5',
#                        help='Weights for multi-objective (completion,waiting)')
#     parser.add_argument('--train_multiple', action='store_true',
#                        help='Train with multiple weight configurations')
    
#     # Logging
#     parser.add_argument('--log_dir', type=str, default='./logs/',
#                        help='Directory for logs')
#     parser.add_argument('--save_dir', type=str, default='./models/',
#                        help='Directory to save models')
#     parser.add_argument('--eval_freq', type=int, default=10000,
#                        help='Evaluation frequency')
    
#     return parser.parse_args()

# def setup_directories(args):
#     """Tạo các thư mục cần thiết"""
#     os.makedirs(args.log_dir, exist_ok=True)
#     os.makedirs(args.save_dir, exist_ok=True)
#     os.makedirs('./results/', exist_ok=True)

# def train_single_weight(args, config_manager, data_loader, weight_id, w_comp, w_wait):
#     """Train với single weight configuration"""
#     print(f"\n{'='*60}")
#     print(f"Training with weights: completion={w_comp}, waiting={w_wait}")
#     print(f"{'='*60}\n")
    
#     # Gói env bằng ActionMasker 
#     # def mask_fn(env):
#     #     return env.get_action_mask()

#     # # Create environment
#     # env = ParallelVRPDEnv(data_loader, config_manager)
#     # env = ActionMasker(env, mask_fn)


#     # eval_env = ParallelVRPDEnv(data_loader, config_manager)
#     # eval_env = ActionMasker(eval_env, mask_fn)

#     def mask_fn(env):
#         return env.get_action_mask()

#     env = ParallelVRPDEnv(data_loader, config_manager)
#     env = ActionMasker(env, mask_fn)        # 🔹 mask hoạt động bình thường

#     eval_env = ParallelVRPDEnv(data_loader, config_manager)
#     eval_env = ActionMasker(eval_env, mask_fn)
    
#     # Create WA-PPO
#     wa_ppo = WAPPO(env, config_manager, log_dir=args.log_dir)
#     model = wa_ppo.create_model(weight_id, w_comp, w_wait)
    
#     # Callback
#     callback = MultiObjectiveCallback(
#         eval_env=eval_env,
#         eval_freq=args.eval_freq,
#         verbose=1
#     )
    
#     # Train
#     print(f"Starting training for {args.total_timesteps} timesteps...")

#     # 1) Env base vẫn là gymnasium.Env?
#     print("Base env class:", type(env.unwrapped))

#     # 2) ActionMasker có tồn tại?
#     print("Has get_action_mask:", hasattr(env.env, "get_action_mask") or hasattr(env.env, "_get_action_mask"))

#     # 3) Reset cuối cùng trước learn() trả về gì?
#     test = env.reset()
#     print("reset() returned type:", type(test))
#     # Với VecEnv, reset() của DummyVecEnv sẽ trả obs (np.ndarray batch), nhưng đây là trước khi SB3 bọc VecEnv,
#     # nên ở đây bạn đang in của env đơn lẻ + ActionMasker: phải là (obs, info).

#     print("Env chain:", env)
#     print("Underlying env:", env.env)
#     print("Reset returns:", env.reset())

#     wa_ppo.train_model(
#         weight_id=weight_id,
#         total_timesteps=args.total_timesteps,
#         callback=callback
#     )
    
#     # Save model
#     model_path = os.path.join(args.save_dir, f'model_{weight_id}')
#     wa_ppo.save_model(weight_id, model_path)
#     print(f"Model saved to {model_path}")
    
#     # Evaluate and visualize
#     print("\nEvaluating final model...")
#     obs = eval_env.reset()
#     done = False
    
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = eval_env.step(action)
    
#     # Visualize
#     visualizer = VRPDVisualizer(data_loader, save_dir='./results/')
    
#     truck_routes = info['truck_routes']
#     drone_routes = info['drone_routes']
    
#     # Calculate metrics
#     calc = eval_env.calculator
#     times = []
#     wait = 0
    
#     for route in truck_routes:
#         if route:
#             t, w = calc.calculate_truck_time(route)
#             times.append(t)
#             wait += w
    
#     for route in drone_routes:
#         if route:
#             t, w, _ = calc.calculate_drone_time(route)
#             times.append(t)
#             wait += w
    
#     completion_time = max(times) if times else 0
    
#     print(f"\nFinal Results:")
#     print(f"  Completion time: {completion_time:.2f}s")
#     print(f"  Total waiting time: {wait:.2f}s")
#     print(f"  Truck routes: {truck_routes}")
#     print(f"  Drone routes: {drone_routes}")
    
#     # Plot routes
#     fig, _ = visualizer.plot_routes(
#         truck_routes, drone_routes,
#         title=f"VRPD Routes - {weight_id}",
#         save_path=f'./results/routes_{weight_id}.png'
#     )
#     print(f"Route visualization saved to ./results/routes_{weight_id}.png")
    
#     # Save summary
#     metrics = {
#         'completion_time': completion_time,
#         'waiting_time': wait,
#         'total_objective': completion_time + wait
#     }
#     routes = {
#         'truck_routes': truck_routes,
#         'drone_routes': drone_routes
#     }
#     visualizer.save_summary_report(metrics, routes, f'summary_{weight_id}.txt')
    
#     return model, metrics

# def train_multiple_weights(args, config_manager, data_loader):
#     """Train với nhiều weight configurations"""
#     weight_configs = [
#         ('w_1_0', 1.0, 0.0),
#         ('w_0.8_0.2', 0.8, 0.2),
#         ('w_0.5_0.5', 0.5, 0.5),
#         ('w_0.2_0.8', 0.2, 0.8),
#         ('w_0_1', 0.0, 1.0),
#     ]
    
#     all_results = {}
    
#     for weight_id, w_comp, w_wait in weight_configs:
#         model, metrics = train_single_weight(
#             args, config_manager, data_loader, 
#             weight_id, w_comp, w_wait
#         )
#         all_results[weight_id] = metrics
    
#     # Visualize Pareto front
#     print("\n" + "="*60)
#     print("Creating Pareto Front visualization...")
#     print("="*60)
    
#     visualizer = VRPDVisualizer(data_loader, save_dir='./results/')
#     visualizer.plot_pareto_front(
#         all_results,
#         save_path='./results/pareto_front.png'
#     )
#     print("Pareto front saved to ./results/pareto_front.png")
    
#     return all_results

# def main():
#     """Main function"""
#     args = parse_args()
    
#     # Setup
#     setup_directories(args)
    
#     print("="*60)
#     print("WA-PPO Training for Parallel VRPD")
#     print("="*60)
#     print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
#     # Load configuration
#     print("\nLoading configuration...")
#     config_manager = ConfigManager(
#         drone_config_path=args.drone_config,
#         truck_config_path=args.truck_config
#     )
    
#     # Update RL config from args
#     config_manager.update_rl_config(
#         learning_rate=args.lr,
#         n_steps=args.n_steps,
#         batch_size=args.batch_size,
#         n_epochs=args.n_epochs,
#         total_timesteps=args.total_timesteps
#     )
    
#     print(config_manager)
    
#     # Load data
#     print("\nLoading data...")
#     data_loader = DataLoader(args.data)
#     print(data_loader.summary())
    
#     # Train
#     if args.train_multiple:
#         results = train_multiple_weights(args, config_manager, data_loader)
#     else:
#         weights = [float(w) for w in args.weights.split(',')]
#         w_comp, w_wait = weights[0], weights[1]
#         weight_id = f"w_{w_comp}_{w_wait}"
        
#         model, metrics = train_single_weight(
#             args, config_manager, data_loader,
#             weight_id, w_comp, w_wait
#         )
    
#     print("\n" + "="*60)
#     print("Training completed!")
#     print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("="*60)

# if __name__ == "__main__":
#     main()


"""
Train entrypoint for ParallelVRPDEnv + WA-PPO (MaskablePPO).
- Uses Gymnasium API
- Proper action masking (ActionMasker)
- Clean eval loop in callback
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import torch

from wa_ppo import WAPPO, mask_fn
from sb3_contrib.common.wrappers import ActionMasker

# Project modules you already have:
from environment import ParallelVRPDEnv
from data_loader import DataLoader
from config import ConfigManager


def make_env(cfg: ConfigManager, data: DataLoader, seed: int = 0):
    env = ParallelVRPDEnv(data_loader=data, config_manager=cfg)
    # Do NOT vec-wrap here; WAPPO.create_model will wrap Monitor + DummyVecEnv
    # But we DO add masking for safety & for eval_env symmetry:
    env = ActionMasker(env, mask_fn)
    # Set seed
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
            env.np_random.seed(seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--eval_freq", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config/data (assumes your classes already do the right thing)
    config_manager = ConfigManager()          # adjust constructor if needed
    data_loader = DataLoader(config_manager)  # adjust constructor if needed

    # Create base env (masked) for training & eval
    train_env = make_env(config_manager, data_loader, seed=args.seed)
    eval_env = make_env(config_manager, data_loader, seed=args.seed + 123)

    # WA-PPO
    trainer = WAPPO(env=train_env, config_manager=config_manager, log_dir=args.log_dir, seed=args.seed)
    model = trainer.create_model(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # You can override policy kwargs here if you want:
        # policy_kwargs=dict(net_arch=[256, 256]),
    )

    # Train
    trainer.train_model(
        total_timesteps=args.timesteps,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
