"""
Route Calculator for VRPD Problem
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class TimeDistanceCalculator:
    """
    Calculate travel time and energy for VRPD problem
    """
    
    def __init__(
        self,
        coords: np.ndarray,
        drone_config: Dict,
        truck_config: Dict,
        service_time_drone: float = 30.0,
        service_time_truck: float = 60.0,
    ):
        """
        Initialize calculator
        
        Args:
            coords: (N+1, 2) array with depot at index 0
            drone_config: {
                takeoff_speed, cruise_speed, landing_speed, cruise_alt,
                capacity, battery_power, beta (W/kg), gamma (W)
            }
            truck_config: {
                "V_max (m/s)", "T (hour)": {"0-1": 0.7, ...}
            }
            service_time_drone: Service time in seconds
            service_time_truck: Service time in seconds
        """
        self.coords = coords.astype(np.float32)
        self.N = len(coords) - 1
        
        # Drone parameters
        self.drone_takeoff_speed = float(drone_config["takeoff_speed"])
        self.drone_cruise_speed = float(drone_config["cruise_speed"])
        self.drone_landing_speed = float(drone_config["landing_speed"])
        self.drone_cruise_alt = float(drone_config["cruise_alt"])
        self.drone_capacity = float(drone_config["capacity"])
        self.drone_battery = float(drone_config["battery_power"])  # Joules
        
        # Energy model parameters
        self.drone_beta = float(drone_config["beta"])    # W/kg - power per kg
        self.drone_gamma = float(drone_config["gamma"])  # W - base power
        
        # Truck parameters
        self.truck_v_max = float(truck_config.get("V_max (m/s)", truck_config.get("v_max", 15.557)))
        self.truck_capacity = float(truck_config.get("capacity", 50.0))
        
        # Parse time-based speed multipliers
        self.truck_speed_multipliers = self._parse_truck_speed_config(truck_config)
        
        # Service times
        self.service_time_drone = float(service_time_drone)
        self.service_time_truck = float(service_time_truck)
        
        # Precompute distances
        self._compute_distance_matrix()
        
        print(f"✅ Calculator initialized:")
        print(f"   Drone: cruise={self.drone_cruise_speed:.1f}m/s, battery={self.drone_battery:.0f}J")
        print(f"   Drone energy: P(w) = {self.drone_beta:.1f}*w + {self.drone_gamma:.0f} (W)")
        print(f"   Truck: v_max={self.truck_v_max:.1f}m/s")
    
    def _parse_truck_speed_config(self, truck_config: Dict) -> Optional[Dict[int, float]]:
        """Parse truck speed multipliers from config"""
        t_config = truck_config.get("T (hour)", None)
        if t_config is None:
            return None
        
        multipliers = {}
        for time_range, multiplier in t_config.items():
            try:
                start_hour = int(time_range.split("-")[0])
                multipliers[start_hour] = float(multiplier)
            except (ValueError, IndexError):
                print(f"⚠️  Warning: Cannot parse time range '{time_range}'")
                continue
        
        return multipliers if multipliers else None
    
    def _compute_distance_matrix(self):
        """Precompute Euclidean distances"""
        n = len(self.coords)
        self.dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.dist_matrix[i, j] = np.linalg.norm(
                        self.coords[i] - self.coords[j]
                    )
    
    def get_distance(self, i: int, j: int) -> float:
        """Get Euclidean distance between nodes (meters)"""
        return float(self.dist_matrix[i, j])
    
    # ========== TRUCK METHODS ==========
    
    def _get_truck_speed_at_time(self, current_time_seconds: float) -> float:
        """
        Get truck speed at given time
        
        Args:
            current_time_seconds: Current time in seconds
            
        Returns:
            speed: v_max * sigma (m/s)
        """
        if self.truck_speed_multipliers is None:
            return self.truck_v_max
        
        hours = current_time_seconds / 3600.0
        hour_slot = int(hours) % 12  # Wrap every 12 hours
        multiplier = self.truck_speed_multipliers.get(hour_slot, 1.0)
        
        return self.truck_v_max * multiplier
    
    def get_truck_travel_time(
        self, 
        i: int, 
        j: int, 
        current_time: float = 0.0
    ) -> float:
        """
        Get truck travel time from i to j
        
        Handles time-varying speed correctly:
        - If route crosses hour boundary, split calculation
        
        Args:
            i: Start node
            j: End node
            current_time: Current time in seconds
            
        Returns:
            travel_time: Time in seconds
        """
        distance = self.get_distance(i, j)
        
        if distance < 1e-6:
            return 0.0
        
        if self.truck_speed_multipliers is None:
            # No time-varying speed
            return distance / self.truck_v_max
        
        # Handle hour boundaries like in the C++ code
        time_elapsed = 0.0
        remaining_distance = distance
        current_hour = int(current_time / 3600)
        
        while remaining_distance > 0:
            # Speed in current hour
            hour_slot = current_hour % 12
            multiplier = self.truck_speed_multipliers.get(hour_slot, 1.0)
            speed = self.truck_v_max * multiplier
            
            # Time until next hour boundary
            next_hour_time = (current_hour + 1) * 3600
            time_in_hour = next_hour_time - (current_time + time_elapsed)
            
            # Distance can travel in this hour
            dist_in_hour = time_in_hour * speed
            
            if remaining_distance <= dist_in_hour:
                # Can finish in this hour
                time_elapsed += remaining_distance / speed
                break
            else:
                # Need to continue to next hour
                time_elapsed += time_in_hour
                remaining_distance -= dist_in_hour
                current_hour += 1
        
        return time_elapsed
    
    def calculate_truck_route_time(
        self, 
        route: List[int], 
        start_time: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate truck route completion time and total waiting time
        
        Args:
            route: List of customer indices (0-indexed)
            start_time: Start time in seconds
            
        Returns:
            (completion_time, total_waiting_time) in seconds
        """
        if not route:
            return 0.0, 0.0
        
        # Full route: depot -> customers -> depot
        full_route = [0] + [c + 1 for c in route] + [0]
        
        time_need = start_time
        time_wait = []
        
        for i in range(len(full_route) - 1):
            # Travel to next node
            travel_time = self.get_truck_travel_time(
                full_route[i], 
                full_route[i + 1], 
                time_need
            )
            time_need += travel_time
            
            # Service time (not at depot)
            if i < len(full_route) - 2:
                time_need += self.service_time_truck
                time_wait.append(time_need)
        
        # Remove last service time
        time_need -= self.service_time_truck
        
        # Calculate total waiting time
        total_wait = sum(time_need - t for t in time_wait)
        
        return time_need, total_wait
    
    # ========== DRONE METHODS ==========
    
    def get_drone_travel_time(self, i: int, j: int, weight: float = 0.0) -> float:
        """
        Get drone travel time from i to j
        
        Phases:
        1. Takeoff: vertical climb to cruise altitude
        2. Cruise: horizontal flight
        3. Landing: vertical descent
        
        Args:
            i: Start node
            j: End node
            weight: Current payload weight (kg)
            
        Returns:
            travel_time: Time in seconds
        """
        if i == j:
            return 0.0
        
        horizontal_dist = self.get_distance(i, j)
        
        # Time for each phase
        t_takeoff = self.drone_cruise_alt / self.drone_takeoff_speed
        t_cruise = horizontal_dist / self.drone_cruise_speed
        t_landing = self.drone_cruise_alt / self.drone_landing_speed
        
        return t_takeoff + t_cruise + t_landing
    
    def get_drone_energy(self, i: int, j: int, weight: float = 0.0) -> float:
        """
        Energy model: E = P(w) × t
        where P(w) = β*w + γ (Watts)
        
        Args:
            i: Start node
            j: End node
            weight: Payload weight (kg)
            
        Returns:
            energy: Energy in Joules
        """
        if i == j:
            return 0.0
        
        # Get flight time
        flight_time = self.get_drone_travel_time(i, j, weight)
        
        # Calculate power consumption
        # P(w) = β*w + γ (W)
        power = self.drone_beta * weight + self.drone_gamma
        
        # Energy = Power × Time
        # E = P(w) × t (J)
        energy = power * flight_time
        
        return float(energy)
    
    def calculate_drone_route_energy(
        self, 
        route: List[int], 
        demands: np.ndarray
    ) -> float:
        """
        Calculate total energy for drone route
        
        Args:
            route: List of customer indices (0-indexed)
            demands: Array of customer demands (kg)
            
        Returns:
            total_energy: Energy in Joules
        """
        if not route:
            return 0.0
        
        total_energy = 0.0
        current_node = 0  # Depot
        current_weight = sum(demands[c] for c in route)
        
        for cid in route:
            customer_node = cid + 1
            
            # Energy to customer with current weight
            energy = self.get_drone_energy(current_node, customer_node, current_weight)
            total_energy += energy
            
            # Deliver: reduce weight
            current_weight -= demands[cid]
            current_node = customer_node
        
        # Return to depot (empty)
        total_energy += self.get_drone_energy(current_node, 0, 0.0)
        
        return float(total_energy)
    
    def calculate_drone_route_time(
        self, 
        route: List[int],
        demands: np.ndarray,
        start_time: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate drone route completion time and waiting time
        
        Args:
            route: List of customer indices (0-indexed)
            demands: Array of customer demands
            start_time: Start time in seconds
            
        Returns:
            (completion_time, total_waiting_time) in seconds
        """
        if not route:
            return 0.0, 0.0
        
        time_need = start_time
        time_wait = []
        current_node = 0
        current_weight = sum(demands[c] for c in route)
        
        for cid in route:
            customer_node = cid + 1
            
            # Travel time
            travel_time = self.get_drone_travel_time(current_node, customer_node, current_weight)
            time_need += travel_time
            
            # Service time
            time_need += self.service_time_drone
            time_wait.append(time_need)
            
            # Update state
            current_weight -= demands[cid]
            current_node = customer_node
        
        # Return to depot
        time_need += self.get_drone_travel_time(current_node, 0, 0.0)
        time_need -= self.service_time_drone
        
        # Calculate total waiting time
        total_wait = sum(time_need - t for t in time_wait)
        
        return time_need, total_wait
    
    # ========== FEASIBILITY CHECKS ==========
    
    def feasible_drone_route(
        self, 
        route: List[int], 
        demands: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if drone route is feasible
        
        Args:
            route: List of customer indices (0-indexed)
            demands: Array of customer demands
            
        Returns:
            (is_feasible, reason)
        """
        if not route:
            return True, "Empty route"
        
        # Check capacity
        total_demand = sum(demands[c] for c in route)
        if total_demand > self.drone_capacity:
            return False, f"Capacity exceeded: {total_demand:.2f} > {self.drone_capacity:.2f}"
        
        # Check energy
        total_energy = self.calculate_drone_route_energy(route, demands)
        if total_energy > self.drone_battery:
            return False, f"Energy exceeded: {total_energy:.0f} > {self.drone_battery:.0f}"
        
        return True, "Feasible"
    
    def get_max_drone_range(self, weight: float = 0.0) -> float:
        """
        Calculate maximum range for drone with given weight
        
        Args:
            weight: Payload weight (kg)
            
        Returns:
            max_range: Maximum distance in meters
        """
        # Calculate power consumption
        power = self.drone_beta * weight + self.drone_gamma
        
        # Max time drone can fly
        max_time = self.drone_battery / power
        
        # Subtract takeoff/landing time
        flight_time = max_time - (
            self.drone_cruise_alt / self.drone_takeoff_speed +
            self.drone_cruise_alt / self.drone_landing_speed
        )
        
        if flight_time <= 0:
            return 0.0
        
        # Max horizontal distance
        max_range = flight_time * self.drone_cruise_speed
        
        return float(max_range)
    
    # ========== UTILITY METHODS ==========
    
    def print_summary(self):
        """Print calculator configuration"""
        print("\n" + "=" * 60)
        print("TIME & ENERGY CALCULATOR SUMMARY")
        print("=" * 60)
        print(f"Customers: {self.N}")
        
        print(f"\n📦 Drone:")
        print(f"  Speeds: takeoff={self.drone_takeoff_speed:.1f}, "
              f"cruise={self.drone_cruise_speed:.1f}, "
              f"landing={self.drone_landing_speed:.1f} m/s")
        print(f"  Cruise altitude: {self.drone_cruise_alt:.1f} m")
        print(f"  Capacity: {self.drone_capacity:.2f} kg")
        print(f"  Battery: {self.drone_battery:.0f} J")
        print(f"  Energy model: P(w) = {self.drone_beta:.1f}*w + {self.drone_gamma:.0f} W")
        
        # Example ranges
        print(f"  Max range:")
        for w in [0, 1, 2]:
            r = self.get_max_drone_range(w)
            print(f"    {w} kg: {r:.0f} m")
        
        print(f"  Service time: {self.service_time_drone:.0f} s")
        
        print(f"\n🚚 Truck:")
        print(f"  Max speed: {self.truck_v_max:.1f} m/s")
        if self.truck_speed_multipliers:
            print(f"  Speed multipliers (12-hour cycle):")
            for hour in sorted(self.truck_speed_multipliers.keys()):
                mult = self.truck_speed_multipliers[hour]
                speed = self.truck_v_max * mult
                print(f"    Hour {hour:2d}: {mult:.2f}x = {speed:.2f} m/s")
        else:
            print(f"  Constant speed: {self.truck_v_max:.1f} m/s")
        print(f"  Service time: {self.service_time_truck:.0f} s")
        print("=" * 60)


def load_truck_config(config_path: str) -> Dict:
    """Load truck configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Sample coordinates
    coords = np.array([
        [0, 0],       # Depot
        [1000, 500],  # Customer 1
        [1500, 1000], # Customer 2
    ], dtype=np.float32)
    
    # Drone config
    drone_config = {
        "takeoff_speed": 15.6464,
        "cruise_speed": 31.2928,
        "landing_speed": 7.8232,
        "cruise_alt": 50.0,
        "capacity": 2.27,
        "battery_power": 904033,  # Joules
        "beta": 24.2,   # W/kg
        "gamma": 1392,  # W
    }
    
    # Truck config with time-based speed
    truck_config = {
        "V_max (m/s)": 15.557,
        "capacity": 50.0,
        "T (hour)": {
            "0-1": 1.0, "1-2": 1.0, "2-3": 1.0,
            "3-4": 0.9, "4-5": 0.8, "5-6": 0.7,  # Morning rush
            "6-7": 0.8, "7-8": 0.9, "8-9": 1.0,
            "9-10": 1.0, "10-11": 1.0, "11-12": 1.0,
        }
    }
    
    calc = TimeDistanceCalculator(coords, drone_config, truck_config)
    calc.print_summary()
    
    # Test calculations
    print("\n" + "=" * 60)
    print("TEST CALCULATIONS")
    print("=" * 60)
    
    # Test drone
    demands = np.array([1.0, 1.5])
    route = [0, 1]  # Visit both customers
    
    print(f"\nDrone route: depot -> C1 -> C2 -> depot")
    print(f"Demands: {demands}")
    
    energy = calc.calculate_drone_route_energy(route, demands)
    time, wait = calc.calculate_drone_route_time(route, demands)
    feasible, reason = calc.feasible_drone_route(route, demands)
    
    print(f"  Total energy: {energy:.0f} J ({energy/calc.drone_battery*100:.1f}% of battery)")
    print(f"  Completion time: {time:.1f} s")
    print(f"  Total waiting: {wait:.1f} s")
    print(f"  Feasible: {feasible} - {reason}")
    
    # Test truck
    print(f"\nTruck route: depot -> C1 -> C2 -> depot")
    truck_time, truck_wait = calc.calculate_truck_route_time(route, start_time=0)
    print(f"  Completion time: {truck_time:.1f} s")
    print(f"  Total waiting: {truck_wait:.1f} s")
    
    print("=" * 60)