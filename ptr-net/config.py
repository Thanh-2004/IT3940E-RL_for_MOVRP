import json
import numpy as np
import torch

class SystemConfig:
    def __init__(self, truck_config_path, drone_config_path, drone_type="1"):
        self.truck_config = self._load_json(truck_config_path)
        self.drone_config_full = self._load_json(drone_config_path)
        
        self.drone_type = str(drone_type)
        if self.drone_type not in self.drone_config_full:
            raise ValueError(f"Drone type {drone_type} not found")
        
        # Load Drone Params
        p = self.drone_config_full[self.drone_type]
        self.drone_params = p
        self.drone_max_energy = p['batteryPower [Joule]']
        self.drone_speed = p['cruiseSpeed [m/s]']
        self.drone_capacity_kg = p['capacity [kg]']
        
        # Pre-calc time for takeoff/landing to save compute
        self.t_takeoff = p['cruiseAlt [m]'] / p['takeoffSpeed [m/s]']
        self.t_landing = p['cruiseAlt [m]'] / p['landingSpeed [m/s]']

        # Truck Time Windows
        self.truck_time_factors = []
        for key, factor in self.truck_config['T (hour)'].items():
            start, end = map(int, key.split('-'))
            self.truck_time_factors.append((start, end, factor))
        self.truck_v_max = self.truck_config['V_max (m/s)']

    def _load_json(self, path):
        with open(path, 'r') as f: return json.load(f)

    def get_truck_speed_batch(self, current_time_seconds):
        """
        Tính vận tốc Truck theo Eq (11).
        Input: Tensor (Batch_Size,)
        Output: Tensor (Batch_Size,)
        """
        if isinstance(current_time_seconds, torch.Tensor):
            hours = (current_time_seconds / 3600) % 24
            factors = torch.ones_like(hours)
            for start, end, f in self.truck_time_factors:
                mask = (hours >= start) & (hours < end)
                factors[mask] = f
            return self.truck_v_max * factors
        else:
            # Fallback scalar
            return self.truck_v_max # Simplification for scalar
