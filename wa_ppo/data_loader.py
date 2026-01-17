"""
Data Loader for Parallel VRPD
Load và xử lý dữ liệu khách hàng
"""
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Customer:
    """Thông tin khách hàng"""
    id: int
    x: float
    y: float
    demand: float
    only_truck: bool
    service_time_truck: float
    service_time_drone: float
    
    def get_location(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def distance_to(self, other) -> float:
        if isinstance(other, Customer):
            loc = other.get_location()
        else:
            loc = np.array(other)
        dst = np.linalg.norm(self.get_location() - loc)
        # print(f"[DEBUG]: Distance: {dst}, Shape: {dst.shape}")
        return float(dst)

class DataLoader:
    """Load dữ liệu từ file"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.customers: List[Customer] = []
        self.num_staff = 0
        self.num_drones = 0
        self.drone_time_limit = 0
        self.depot = np.array([0.0, 0.0])
        self._load_data()
    
    def _load_data(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        self.num_staff = int(lines[0].split()[1])
        self.num_drones = int(lines[1].split()[1])
        self.drone_time_limit = int(lines[2].split()[1])
        num_customers = int(lines[3].split()[1])
        
        for i in range(num_customers):
            line = lines[5 + i].strip().split()
            customer = Customer(
                id=i + 1,
                x=float(line[0]),
                y=float(line[1]),
                demand=float(line[2]),
                only_truck=bool(int(line[3])),
                service_time_truck=float(line[4]),
                service_time_drone=float(line[5])
            )
            self.customers.append(customer)
    
    def get_distance_matrix(self) -> np.ndarray:
        n = len(self.customers) + 1
        matrix = np.zeros((n, n))
        
        for i, customer in enumerate(self.customers, start=1):
            matrix[0][i] = customer.distance_to(self.depot)
            matrix[i][0] = matrix[0][i]
        
        for i, c1 in enumerate(self.customers, start=1):
            for j, c2 in enumerate(self.customers, start=1):
                if i != j:
                    matrix[i][j] = c1.distance_to(c2)
        
        return matrix
    
    def get_truck_only_customers(self) -> List[int]:
        return [c.id for c in self.customers if c.only_truck]
    
    def get_customer_demands(self) -> np.ndarray:
        return np.array([c.demand for c in self.customers])
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        x_coords = [0.0] + [c.x for c in self.customers]
        y_coords = [0.0] + [c.y for c in self.customers]
        return np.array(x_coords), np.array(y_coords)
    
    def summary(self) -> str:
        truck_only = len(self.get_truck_only_customers())
        total_demand = sum(c.demand for c in self.customers)
        
        return f"""
=== Data Summary ===
Customers: {len(self.customers)}
Trucks: {self.num_staff}
Drones: {self.num_drones}
Drone time limit: {self.drone_time_limit}s
Truck-only: {truck_only}
Total demand: {total_demand:.2f} kg
"""