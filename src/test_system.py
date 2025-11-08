"""
System Test Suite
Test tất cả các modules và integration
"""
import numpy as np
import os
import sys
from typing import List, Tuple

# Test imports
def test_imports():
    """Test tất cả imports"""
    print("Testing imports...")
    try:
        from config import ConfigManager, DroneConfig, TruckConfig, RLConfig
        from data_loader import DataLoader, Customer
        from route_calculator import RouteCalculator
        from environment import ParallelVRPDEnv
        from wa_ppo import WAPPO, VRPDFeatureExtractor, MultiObjectiveCallback
        from visualizer import VRPDVisualizer
        from utils import (seed_everything, calculate_pareto_front, 
                          validate_route, format_time)
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

# Test ConfigManager
def test_config_manager():
    """Test ConfigManager"""
    print("\nTesting ConfigManager...")
    try:
        # Test with default configs
        config = ConfigManager()
        assert config.drone.capacity > 0
        assert config.truck.v_max > 0
        assert config.rl.learning_rate > 0
        
        # Test updates
        config.update_rl_config(learning_rate=5e-4)
        assert config.rl.learning_rate == 5e-4
        
        config.update_weights(0.8, 0.2)
        assert abs(config.rl.w_completion_time - 0.8) < 1e-6
        
        print("✓ ConfigManager tests passed")
        return True
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False

# Test DataLoader
def test_data_loader():
    """Test DataLoader"""
    print("\nTesting DataLoader...")
    try:
        # Check if test file exists
        if not os.path.exists('6.5.1.txt'):
            print("⚠ Warning: 6.5.1.txt not found, skipping test")
            return True
        
        data = DataLoader('6.5.1.txt')
        
        # Basic checks
        assert len(data.customers) > 0
        assert data.num_staff > 0
        assert data.num_drones > 0
        
        # Distance matrix
        dist_matrix = data.get_distance_matrix()
        assert dist_matrix.shape[0] == len(data.customers) + 1
        assert dist_matrix.shape[1] == len(data.customers) + 1
        
        # Symmetry check
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                assert abs(dist_matrix[i][j] - dist_matrix[j][i]) < 1e-6
        
        # Coordinates
        x_coords, y_coords = data.get_coordinates()
        assert len(x_coords) == len(data.customers) + 1
        assert x_coords[0] == 0 and y_coords[0] == 0  # Depot
        
        print("✓ DataLoader tests passed")
        return True
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return False

# Test RouteCalculator
def test_route_calculator():
    """Test RouteCalculator"""
    print("\nTesting RouteCalculator...")
    try:
        if not os.path.exists('6.5.1.txt'):
            print("⚠ Warning: 6.5.1.txt not found, skipping test")
            return True
        
        config = ConfigManager(
            drone_config_path='drone_linear_config.json',
            truck_config_path='Truck_config.json'
        )
        data = DataLoader('6.5.1.txt')
        
        calc = RouteCalculator(
            distance_matrix=data.get_distance_matrix(),
            truck_config=config.truck,
            drone_config=config.drone
        )
        
        # Test truck route
        truck_route = [1, 2, 3]
        time, wait = calc.calculate_truck_time(truck_route)
        assert time > 0
        assert wait >= 0
        assert time >= wait
        
        # Test drone route
        drone_route = [1]
        time, wait, feasible = calc.calculate_drone_time(drone_route)
        assert time > 0
        assert wait >= 0
        
        # Test capacity check
        demands = [data.customers[i-1].demand for i in drone_route]
        result = calc.check_drone_capacity(drone_route, demands)
        assert isinstance(result, bool)
        
        # Test energy check
        result = calc.check_drone_energy(drone_route, demands)
        assert isinstance(result, bool)
        
        print("✓ RouteCalculator tests passed")
        return True
    except Exception as e:
        print(f"✗ RouteCalculator test failed: {e}")
        return False

# Test Environment
def test_environment():
    """Test Gym Environment"""
    print("\nTesting Environment...")
    try:
        if not os.path.exists('6.5.1.txt'):
            print("⚠ Warning: 6.5.1.txt not found, skipping test")
            return True
        
        config = ConfigManager(
            drone_config_path='drone_linear_config.json',
            truck_config_path='Truck_config.json'
        )
        data = DataLoader('6.5.1.txt')
        env = ParallelVRPDEnv(data, config)
        
        # Test reset
        obs = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        
        # Test step with depot action (always valid)
        obs, reward, done, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'action_mask' in info
        
        # Test action mask
        mask = env._get_action_mask()
        assert len(mask) == env.action_space.n
        assert mask[0] == True  # Depot always valid
        
        # Test full episode
        env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            mask = env._get_action_mask()
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        assert steps < max_steps  # Should finish
        
        print("✓ Environment tests passed")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

# Test WA-PPO
def test_wa_ppo():
    """Test WA-PPO model creation"""
    print("\nTesting WA-PPO...")
    try:
        if not os.path.exists('6.5.1.txt'):
            print("⚠ Warning: 6.5.1.txt not found, skipping test")
            return True
        
        config = ConfigManager(
            drone_config_path='drone_linear_config.json',
            truck_config_path='Truck_config.json'
        )
        data = DataLoader('6.5.1.txt')
        env = ParallelVRPDEnv(data, config)
        
        # Test model creation
        wa_ppo = WAPPO(env, config, log_dir='./test_logs/')
        model = wa_ppo.create_model('test', 0.5, 0.5)
        
        # Test prediction
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert 0 <= action < env.action_space.n
        
        print("✓ WA-PPO tests passed")
        return True
    except Exception as e:
        print(f"✗ WA-PPO test failed: {e}")
        return False

# Test Visualizer
def test_visualizer():
    """Test Visualizer"""
    print("\nTesting Visualizer...")
    try:
        if not os.path.exists('6.5.1.txt'):
            print("⚠ Warning: 6.5.1.txt not found, skipping test")
            return True
        
        data = DataLoader('6.5.1.txt')
        viz = VRPDVisualizer(data, save_dir='./test_results/')
        
        # Test route plotting
        truck_routes = [[1, 2]]
        drone_routes = [[3, 4]]
        
        fig, ax = viz.plot_routes(truck_routes, drone_routes, 
                                  title="Test", save_path=None)
        assert fig is not None
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close('all')
        
        print("✓ Visualizer tests passed")
        return True
    except Exception as e:
        print(f"✗ Visualizer test failed: {e}")
        return False

# Test Utils
def test_utils():
    """Test utility functions"""
    print("\nTesting Utils...")
    try:
        from utils import (format_time, calculate_pareto_front, 
                          validate_route, normalize_objectives)
        
        # Test format_time
        assert format_time(3661) == "1h 1m 1s"
        assert format_time(61) == "1m 1s"
        assert format_time(30) == "30s"
        
        # Test Pareto front
        solutions = [(10, 20), (15, 15), (20, 10), (25, 25)]
        pareto_indices = calculate_pareto_front(solutions)
        assert len(pareto_indices) > 0
        
        # Test validate_route
        route = [1, 2, 3]
        visited = np.zeros(5, dtype=bool)
        assert validate_route(route, 5, visited) == True
        
        visited[0] = True
        assert validate_route(route, 5, visited) == False
        
        # Test normalize
        objectives = np.array([[10, 20], [30, 40], [50, 60]])
        normalized = normalize_objectives(objectives)
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
        print("✓ Utils tests passed")
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False

# Integration test
def test_integration():
    """Test full integration"""
    print("\nTesting Integration...")
    try:
        if not os.path.exists('6.5.1.txt'):
            print("⚠ Warning: 6.5.1.txt not found, skipping test")
            return True
        
        from utils import seed_everything
        seed_everything(42)
        
        # Setup
        config = ConfigManager(
            drone_config_path='drone_linear_config.json',
            truck_config_path='Truck_config.json'
        )
        config.update_rl_config(total_timesteps=100)
        
        data = DataLoader('6.5.1.txt')
        env = ParallelVRPDEnv(data, config)
        
        # Create and train model (very short)
        wa_ppo = WAPPO(env, config, log_dir='./test_logs/')
        model = wa_ppo.create_model('integration_test', 0.5, 0.5)
        
        # Just test that training starts without error
        wa_ppo.train_model('integration_test', total_timesteps=100)
        
        # Test inference
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        print("✓ Integration test passed")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    
    dirs_to_clean = ['test_logs', 'test_results']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("WA-PPO VRPD System Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("ConfigManager", test_config_manager),
        ("DataLoader", test_data_loader),
        ("RouteCalculator", test_route_calculator),
        ("Environment", test_environment),
        ("WA-PPO", test_wa_ppo),
        ("Visualizer", test_visualizer),
        ("Utils", test_utils),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Cleanup
    print("\nCleaning up test files...")
    cleanup_test_files()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)