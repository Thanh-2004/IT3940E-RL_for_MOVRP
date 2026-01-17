"""
Example Usage Script
Ví dụ minh họa cách sử dụng các module
"""
import numpy as np
from config import ConfigManager
from data_loader import DataLoader
from environment import ParallelVRPDEnv
from wa_ppo import WAPPO, MultiObjectiveCallback
from visualizer import VRPDVisualizer
from utils import seed_everything, format_time, print_model_summary

def example_1_basic_training():
    """
    Ví dụ 1: Training cơ bản với single weight
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Training")
    print("="*60 + "\n")
    
    # Set seed
    seed_everything(42)
    
    # Load configuration
    config = ConfigManager(
        drone_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json',
        truck_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json'
    )
    
    # Update hyperparameters
    config.update_rl_config(
        learning_rate=3e-4,
        total_timesteps=50000  # Demo với timesteps nhỏ
    )
    
    # Load data
    data = DataLoader('/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/200.40.4.txt')
    print(data.summary())
    
    # Create environment
    env = ParallelVRPDEnv(data, config)
    
    # Create WA-PPO
    wa_ppo = WAPPO(env, config, log_dir='./logs/example1/')
    
    # Create model với weights cân bằng
    model = wa_ppo.create_model('balanced', w_completion=0.5, w_waiting=0.5)
    
    # Print model summary
    print_model_summary(model)
    
    # Train
    print("\nStarting training...")
    callback = MultiObjectiveCallback(env, eval_freq=5000)
    wa_ppo.train_model('balanced', total_timesteps=50000, callback=callback)
    
    # Save model
    wa_ppo.save_model('balanced', 'models/example1_model.zip')
    print("\nModel saved!")
    
    # Evaluate
    print("\nEvaluating...")
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    # Visualize
    visualizer = VRPDVisualizer(data, save_dir='./results/example1/')
    visualizer.plot_routes(
        info['truck_routes'],
        info['drone_routes'],
        title="Example 1 - Balanced Weights",
        save_path='./results/example1/routes.png'
    )
    
    print("\n✓ Example 1 completed!")

def example_2_multi_weight_training():
    """
    Ví dụ 2: Training với nhiều weights để tìm Pareto front
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Weight Training")
    print("="*60 + "\n")
    
    seed_everything(42)
    
    # Setup
    config = ConfigManager(
        drone_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json',
        truck_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json'
    )
    config.update_rl_config(total_timesteps=30000)  # Demo ngắn
    
    data = DataLoader('/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/6.5.1.txt')
    env = ParallelVRPDEnv(data, config)
    
    # Create WA-PPO
    wa_ppo = WAPPO(env, config, log_dir='./logs/example2/')
    
    # Train với 3 weight configurations
    weight_configs = [
        ('completion_focus', 0.8, 0.2),
        ('balanced', 0.5, 0.5),
        ('waiting_focus', 0.2, 0.8)
    ]
    
    results = {}
    
    for weight_id, w_comp, w_wait in weight_configs:
        print(f"\n--- Training {weight_id} ---")
        
        model = wa_ppo.create_model(weight_id, w_comp, w_wait)
        wa_ppo.train_model(weight_id, total_timesteps=30000)
        wa_ppo.save_model(weight_id, f'models/{weight_id}.zip')
        
        # Evaluate
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        # Calculate metrics
        calc = env.calculator
        times = []
        wait = 0
        
        for route in info['truck_routes']:
            if route:
                t, w = calc.calculate_truck_time(route)
                times.append(t)
                wait += w
        
        for route in info['drone_routes']:
            if route:
                t, w, _ = calc.calculate_drone_time(route)
                times.append(t)
                wait += w
        
        results[weight_id] = {
            'completion_time': max(times) if times else 0,
            'waiting_time': wait
        }
    
    # Visualize Pareto front
    visualizer = VRPDVisualizer(data, save_dir='./results/example2/')
    visualizer.plot_pareto_front(
        results,
        save_path='./results/example2/pareto_front.png'
    )
    
    print("\n✓ Example 2 completed!")
    print(f"Results: {results}")

def example_3_evaluation_and_comparison():
    """
    Ví dụ 3: Evaluation và so sánh models
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Evaluation & Comparison")
    print("="*60 + "\n")
    
    # Load configuration
    config = ConfigManager(
        drone_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json',
        truck_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json'
    )
    
    data = DataLoader('/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/6.5.1.txt')
    env = ParallelVRPDEnv(data, config)
    
    # Giả sử đã có trained models
    # Ở đây ta sẽ train nhanh 2 models để demo
    
    wa_ppo = WAPPO(env, config, log_dir='./logs/example3/')
    
    # Model 1: Focus on completion
    print("Training Model 1 (completion focus)...")
    model1 = wa_ppo.create_model('model1', 0.8, 0.2)
    wa_ppo.train_model('model1', total_timesteps=20000)
    
    # Model 2: Focus on waiting
    print("Training Model 2 (waiting focus)...")
    model2 = wa_ppo.create_model('model2', 0.2, 0.8)
    wa_ppo.train_model('model2', total_timesteps=20000)
    
    # Evaluate both
    results = {}
    
    for model_name, model in [('model1', model1), ('model2', model2)]:
        print(f"\nEvaluating {model_name}...")
        
        metrics_list = []
        for _ in range(5):  # 5 episodes
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            
            # Calculate metrics
            calc = env.calculator
            times = []
            wait = 0
            
            for route in info['truck_routes']:
                if route:
                    t, w = calc.calculate_truck_time(route)
                    times.append(t)
                    wait += w
            
            for route in info['drone_routes']:
                if route:
                    t, w, _ = calc.calculate_drone_time(route)
                    times.append(t)
                    wait += w
            
            metrics_list.append({
                'completion': max(times) if times else 0,
                'waiting': wait
            })
        
        # Average metrics
        avg_completion = np.mean([m['completion'] for m in metrics_list])
        avg_waiting = np.mean([m['waiting'] for m in metrics_list])
        
        results[model_name] = {
            'completion_time': avg_completion,
            'waiting_time': avg_waiting,
            'total': avg_completion + avg_waiting
        }
        
        print(f"  Avg Completion: {avg_completion:.2f}s")
        print(f"  Avg Waiting: {avg_waiting:.2f}s")
    
    # Compare
    print("\n--- Comparison ---")
    print(f"Model 1 (completion focus):")
    print(f"  Total objective: {results['model1']['total']:.2f}s")
    print(f"Model 2 (waiting focus):")
    print(f"  Total objective: {results['model2']['total']:.2f}s")
    
    if results['model1']['total'] < results['model2']['total']:
        print("→ Model 1 is better overall")
    else:
        print("→ Model 2 is better overall")
    
    print("\n✓ Example 3 completed!")

def example_4_custom_hyperparameters():
    """
    Ví dụ 4: Tùy chỉnh hyperparameters
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Hyperparameters")
    print("="*60 + "\n")
    
    # Create custom config
    config = ConfigManager(
        drone_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json',
        truck_config_path='/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json'
    )
    
    # Tùy chỉnh hyperparameters
    custom_params = {
        'learning_rate': 5e-4,  # Higher LR
        'n_steps': 1024,        # Smaller steps
        'batch_size': 128,      # Larger batch
        'n_epochs': 15,         # More epochs
        'ent_coef': 0.02,       # More exploration
        'clip_range': 0.3,      # Larger clip
    }
    
    config.update_rl_config(**custom_params)
    
    print("Custom hyperparameters:")
    for key, value in custom_params.items():
        print(f"  {key}: {value}")
    
    # Load data và train
    data = DataLoader('/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/6.5.1.txt')
    env = ParallelVRPDEnv(data, config)
    
    wa_ppo = WAPPO(env, config, log_dir='./logs/example4/')
    model = wa_ppo.create_model('custom', 0.5, 0.5)
    
    print("\nTraining with custom hyperparameters...")
    wa_ppo.train_model('custom', total_timesteps=30000)
    
    print("\n✓ Example 4 completed!")

def main():
    """Chạy tất cả examples"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Which example to run (1-4)')
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_training,
        2: example_2_multi_weight_training,
        3: example_3_evaluation_and_comparison,
        4: example_4_custom_hyperparameters
    }
    
    print("\n" + "="*60)
    print("WA-PPO VRPD - Example Usage")
    print("="*60)
    
    examples[args.example]()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()