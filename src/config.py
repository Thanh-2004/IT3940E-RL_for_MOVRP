"""
Configuration Manager for Parallel VRPD
Quản lý các siêu tham số và cấu hình cho bài toán
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class DroneConfig:
    """Cấu hình cho Drone"""
    takeoff_speed: float = 15.6464  # m/s
    cruise_speed: float = 31.2928   # m/s
    landing_speed: float = 7.8232   # m/s
    cruise_alt: float = 50.0        # m
    capacity: float = 2.27          # kg
    battery_power: float = 904033   # Joule
    beta: float = 24.2              # W/kg - năng lượng tiêu hao trên 1kg
    gamma: float = 1392             # W - năng lượng tiêu hao không mang hàng
    speed_type: str = "high"
    range_type: str = "high"
    
    @classmethod
    def from_json(cls, filepath: str, config_id: Optional[str] = None):
        """
        Load drone config from JSON file
        
        Args:
            filepath: Path to JSON config file
            config_id: Config ID (e.g., "1", "2", "3", "4"). If None, uses first config.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # If config_id not specified, use first available config
        if config_id is None:
            config_id = list(data.keys())[0]
        
        # Get config for specific ID
        if str(config_id) not in data:
            raise ValueError(f"Config ID '{config_id}' not found in {filepath}. "
                           f"Available: {list(data.keys())}")
        
        config = data[str(config_id)]
        
        return cls(
            takeoff_speed=config.get("takeoffSpeed [m/s]", 15.6464),
            cruise_speed=config.get("cruiseSpeed [m/s]", 31.2928),
            landing_speed=config.get("landingSpeed [m/s]", 7.8232),
            cruise_alt=config.get("cruiseAlt [m]", 50),
            capacity=config.get("capacity [kg]", 2.27),
            battery_power=config.get("batteryPower [Joule]", 904033),
            beta=config.get("beta(w/kg)", 24.2),
            gamma=config.get("gama(w)", 1392),
            speed_type=config.get("speed_type", "high"),
            range_type=config.get("range", "high"),
        )
    
    def __str__(self):
        return (f"Drone[{self.speed_type}-speed, {self.range_type}-range]: "
                f"capacity={self.capacity}kg, battery={self.battery_power}J, "
                f"cruise_speed={self.cruise_speed}m/s")

@dataclass
class TruckConfig:
    """Cấu hình cho Truck"""
    v_max: float = 15.557  # m/s
    capacity: float = 50.0  # kg
    time_windows: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            v_max=data.get("V_max (m/s)", 15.557),
            capacity=data.get("capacity [kg]", 50.0),
            time_windows=data.get("T (hour)", {})
        )
    
    def get_speed_factor(self, hour: int) -> float:
        """Lấy hệ số tốc độ theo giờ"""
        hour = hour % 12
        key = f"{hour}-{hour+1}"
        return self.time_windows.get(key, 0.7)
    
    def __str__(self):
        return f"Truck: v_max={self.v_max}m/s, capacity={self.capacity}kg"

@dataclass
class RLConfig:
    """Cấu hình cho thuật toán RL"""
    # PPO Hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Multi-objective weights
    w_service: float = 0.5  # renamed from w_completion_time
    w_wait: float = 0.5     # renamed from w_waiting_time
    
    # Training
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000
    
    # Network architecture
    policy_layers: List[int] = field(default_factory=lambda: [256, 256])
    value_layers: List[int] = field(default_factory=lambda: [256, 256])
    
    def to_dict(self) -> Dict:
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
        }

@dataclass
class ProblemConfig:
    """Cấu hình bài toán - được parse từ file data"""
    num_customers: int = 0
    num_trucks: int = 1        # number_staff in file
    num_drones: int = 1        # number_drone in file
    coordinate_range: int = 40  # từ tên file (x.y.z)
    drone_config_id: int = 4    # từ tên file (x.y.z)
    drone_flight_time_limit: int = 3600  # droneLimitationFightTime(s) in file
    service_time_truck: int = 60   # ServiceTimeByTruck(s) in file
    service_time_drone: int = 30   # ServiceTimeByDrone(s) in file
    
    def __post_init__(self):
        self.depot_location = np.array([0.0, 0.0])
    
    def __str__(self):
        return (f"Problem: {self.num_customers} customers, {self.num_trucks} trucks, "
                f"{self.num_drones} drones, coord_range={self.coordinate_range}, "
                f"drone_config={self.drone_config_id}")

class ConfigManager:
    """Quản lý tất cả cấu hình - Auto-parse từ filename và file content"""
    
    def __init__(self, 
                 data_file: str,
                 drone_config_path: str,
                 truck_config_path: str):
        """
        Initialize ConfigManager
        
        Args:
            data_file: Path to data file (format: x.y.z.txt where x=customers, y=range, z=drone_config)
            drone_config_path: Path to drone config JSON
            truck_config_path: Path to truck config JSON
        """
        # Parse problem config from filename and file content
        self.problem = self._parse_problem_config(data_file)
        
        # Load drone config based on parsed config_id
        self.drone = DroneConfig.from_json(
            drone_config_path, 
            config_id=str(self.problem.drone_config_id)
        )
        
        # Load truck config
        self.truck = TruckConfig.from_json(truck_config_path)
        
        # RL config
        self.rl = RLConfig()
        
        print(f"✅ Loaded configuration from: {os.path.basename(data_file)}")
        print(f"   {self.problem}")
        print(f"   {self.drone}")
        print(f"   {self.truck}")
    
    @staticmethod
    def _parse_problem_config(data_file: str) -> ProblemConfig:
        """
        Parse problem configuration from filename and file content
        
        Filename format: x.y.z.txt
        - x: number of customers
        - y: coordinate range (width)
        - z: drone config ID (1, 2, 3, or 4)
        
        File content includes:
        - number_staff (trucks)
        - number_drone
        - droneLimitationFightTime(s)
        - Customers count
        - Service times
        """
        filename = os.path.basename(data_file)
        
        # Parse filename: x.y.z.txt
        match = re.match(r'(\d+)\.(\d+)\.(\d+)\.txt', filename)
        if not match:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                f"Expected format: x.y.z.txt (e.g., 200.40.4.txt)"
            )
        
        n_customers_from_name = int(match.group(1))
        coord_range = int(match.group(2))
        drone_config_id = int(match.group(3))
        
        # Parse file content
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Default values
        num_trucks = 1
        num_drones = 1
        flight_time_limit = 3600
        n_customers_from_file = 0
        service_time_truck = 60
        service_time_drone = 30
        
        for line in lines:
            line = line.strip()
            
            # Parse number_staff (trucks)
            if line.startswith('number_staff'):
                num_trucks = int(line.split()[1])
            
            # Parse number_drone
            elif line.startswith('number_drone'):
                num_drones = int(line.split()[1])
            
            # Parse flight time limit
            elif line.startswith('droneLimitationFightTime'):
                flight_time_limit = int(line.split()[1])
            
            # Parse customers count
            elif line.startswith('Customers'):
                n_customers_from_file = int(line.split()[1])
            
            # Parse service times from first data row
            elif re.match(r'^-?\d+\.?\d*\s+-?\d+\.?\d*\s+\d+\.?\d*\s+[01]\s+\d+\s+\d+', line):
                parts = line.split()
                if len(parts) >= 6:
                    service_time_truck = int(parts[4])
                    service_time_drone = int(parts[5])
                    break  # Only need first data row
        
        # Validate consistency
        if n_customers_from_file > 0 and n_customers_from_name != n_customers_from_file:
            print(f"⚠️  Warning: Filename says {n_customers_from_name} customers, "
                  f"but file content says {n_customers_from_file}. Using file content.")
            n_customers = n_customers_from_file
        else:
            n_customers = n_customers_from_name
        
        return ProblemConfig(
            num_customers=n_customers,
            num_trucks=num_trucks,
            num_drones=num_drones,
            coordinate_range=coord_range,
            drone_config_id=drone_config_id,
            drone_flight_time_limit=flight_time_limit,
            service_time_truck=service_time_truck,
            service_time_drone=service_time_drone,
        )

    def get_env_params(self) -> dict:
        """
        Trả về toàn bộ tham số cần thiết để khởi tạo ParallelVRPDEnv
        """
        return {
            # --- Drone parameters ---
            "n_drones": self.problem.num_drones,
            "drone_battery": self.drone.battery_power,
            "capacity": self.truck.capacity,  # Use truck capacity for environment
            
            # --- Common parameters ---
            "use_drone": self.problem.num_drones > 0,
            "debug": False,
            
            # --- Optional (if env uses them) ---
            # "service_time_drone": self.problem.service_time_drone,
            # "service_time_truck": self.problem.service_time_truck,
            # "drone_flight_time_limit": self.problem.drone_flight_time_limit,
        }
    
    def update_rl_config(self, **kwargs):
        """Cập nhật cấu hình RL"""
        for key, value in kwargs.items():
            if hasattr(self.rl, key):
                setattr(self.rl, key, value)
    
    def update_weights(self, w_service: float, w_wait: float):
        """Cập nhật trọng số cho multi-objective"""
        total = w_service + w_wait
        self.rl.w_service = w_service / total
        self.rl.w_wait = w_wait / total
    
    def save_config(self, filepath: str):
        """Lưu cấu hình"""
        config_dict = {
            'drone': {
                'config_id': self.problem.drone_config_id,
                'speed_type': self.drone.speed_type,
                'range_type': self.drone.range_type,
                'capacity': self.drone.capacity,
                'battery_power': self.drone.battery_power,
                'cruise_speed': self.drone.cruise_speed,
            },
            'truck': {
                'v_max': self.truck.v_max,
                'capacity': self.truck.capacity,
            },
            'problem': {
                'num_customers': self.problem.num_customers,
                'num_trucks': self.problem.num_trucks,
                'num_drones': self.problem.num_drones,
                'coordinate_range': self.problem.coordinate_range,
                'drone_config_id': self.problem.drone_config_id,
            },
            'rl': {
                'learning_rate': self.rl.learning_rate,
                'w_service': self.rl.w_service,
                'w_wait': self.rl.w_wait,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"💾 Configuration saved to {filepath}")
    
    def __str__(self):
        return f"""
                {'='*60}
                Configuration Summary
                {'='*60}
                Dataset: {self.problem.num_customers} customers, range={self.problem.coordinate_range}
                Vehicles: {self.problem.num_trucks} truck(s), {self.problem.num_drones} drone(s)
                Drone Config #{self.problem.drone_config_id}: {self.drone.speed_type}-speed, {self.drone.range_type}-range
                • Capacity: {self.drone.capacity} kg
                • Battery: {self.drone.battery_power} J
                • Cruise speed: {self.drone.cruise_speed} m/s
                Truck:
                • Max speed: {self.truck.v_max} m/s
                • Capacity: {self.truck.capacity} kg
                RL Weights: w_service={self.rl.w_service:.2f}, w_wait={self.rl.w_wait:.2f}
                {'='*60}
                """


def parse_filename_info(filename: str) -> Tuple[int, int, int]:
    """
    Utility function to parse filename
    
    Args:
        filename: Format x.y.z.txt
        
    Returns:
        (n_customers, coord_range, drone_config_id)
    """
    basename = os.path.basename(filename)
    match = re.match(r'(\d+)\.(\d+)\.(\d+)\.txt', basename)
    if not match:
        raise ValueError(f"Invalid filename format: {basename}")
    
    return (
        int(match.group(1)),  # n_customers
        int(match.group(2)),  # coord_range
        int(match.group(3)),  # drone_config_id
    )


# Example usage
if __name__ == "__main__":
    # Test with your file
    config = ConfigManager(
        data_file="../data/random_data/200.40.4.txt",
        drone_config_path="../data/drone_linear_config.json",
        truck_config_path="../data/Truck_config.json"
    )
    
    print(config)
    
    # Get env params
    env_params = config.get_env_params()
    print("\nEnvironment Parameters:")
    for k, v in env_params.items():
        print(f"  {k}: {v}")
    
    # Parse filename info
    n_cust, coord_range, drone_id = parse_filename_info("200.40.4.txt")
    print(f"\nParsed from filename: {n_cust} customers, "
          f"range={coord_range}, drone_config={drone_id}")