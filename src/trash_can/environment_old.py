# """
# Gym Environment for Parallel VRPD
# Environment cho Deep RL
# """
# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# from typing import List, Tuple, Dict
# from data_loader import DataLoader
# from route_calculator import RouteCalculator
# from config import ConfigManager


# class ParallelVRPDEnv(gym.Env):
#     """
#     Environment cho bài toán Parallel VRPD
#     State: [customer_features, mask, current_routes]
#     Action: Select next customer for truck/drone
#     """
    
#     def __init__(self, 
#                  data_loader: DataLoader,
#                  config_manager: ConfigManager):
#         super().__init__()
        
#         self.data = data_loader
#         self.config = config_manager
#         self.n_customers = len(data_loader.customers)
        
#         # Initialize route calculator
#         self.calculator = RouteCalculator(
#             distance_matrix=data_loader.get_distance_matrix(),
#             truck_config=config_manager.truck,
#             drone_config=config_manager.drone,
#             service_time_truck=data_loader.customers[0].service_time_truck if self.n_customers > 0 else 60,
#             service_time_drone=data_loader.customers[0].service_time_drone if self.n_customers > 0 else 30
#         )
        
#         # Action space: select customer (0 = depot/end route, 1-n = customers)
#         self.action_space = spaces.Discrete(self.n_customers + 1)
        
#         # Observation space
#         # Features per customer: [x, y, demand, visited, truck_only, distance_to_depot]
#         # UPDATED: Global features now 10 (was 10, still 10 but different content)
#         obs_dim = self.n_customers * 6 + 10
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, 
#             shape=(obs_dim,), 
#             dtype=np.float32
#         )
        
#         # State variables
#         self.truck_routes = [[] for _ in range(self.config.problem.num_trucks)]
#         self.drone_routes = []
#         self.visited = np.zeros(self.n_customers, dtype=bool)
#         self.current_vehicle = 0  # 0 = truck, 1 = drone
#         self.current_vehicle_idx = 0
#         self.step_count = 0
        
#         # Demands
#         self.demands = self.data.get_customer_demands()
#         self.truck_only = set(self.data.get_truck_only_customers())
        
#         # THÊM MỚI: Tracking state
#         self.current_time = 0.0  # Thời gian hiện tại
#         self.current_distance = 0.0  # Tổng distance đã đi
#         self.current_route_demand = 0.0  # Demand của route hiện tại
#         self.drone_energy_remaining = self.config.drone.battery_power  # Năng lượng drone
#         self.last_location = 0  # Vị trí cuối cùng (0 = depot)
        
#     # def reset(self) -> np.ndarray:
#     #     """Reset environment"""
#     #     self.truck_routes = [[] for _ in range(self.config.problem.num_trucks)]
#     #     self.drone_routes = []
#     #     self.visited = np.zeros(self.n_customers, dtype=bool)
#     #     self.current_vehicle = 0
#     #     self.current_vehicle_idx = 0
#     #     self.step_count = 0
        
#     #     # THÊM MỚI: Reset tracking state
#     #     self.current_time = 0.0
#     #     self.current_distance = 0.0
#     #     self.current_route_demand = 0.0
#     #     self.drone_energy_remaining = self.config.drone.battery_power
#     #     self.last_location = 0
        
#     #     return self._get_observation()

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.truck_routes = [[] for _ in range(self.config.problem.num_trucks)]
#         self.drone_routes = []
#         self.visited = np.zeros(self.n_customers, dtype=bool)
#         self.current_vehicle = 0
#         self.current_vehicle_idx = 0
#         self.step_count = 0

#         self.current_time = 0.0
#         self.current_distance = 0.0
#         self.current_route_demand = 0.0
#         self.drone_energy_remaining = self.config.drone.battery_power
#         self.last_location = 0

#         obs = self._get_observation()
#         info = {}
#         return obs, info

        
#     def _get_observation(self) -> np.ndarray:
#         """Tạo observation vector"""
#         obs = []

#         for i in range(self.n_customers):
#             customer = self.data.customers[i]
#             obs.extend([
#                 float(customer.x) / 5000.0,
#                 float(customer.y) / 5000.0,
#                 float(customer.demand) / 2.0,
#                 float(self.visited[i]),
#                 float(customer.only_truck),
#                 float(customer.distance_to(self.data.depot)) / 5000.0
#             ])

#         obs.extend([
#             float(self.current_vehicle),
#             float(self.current_vehicle_idx) / max(self.config.problem.num_trucks, self.config.problem.num_drones),
#             float(np.sum(self.visited)) / self.n_customers,
#             float(len(self.drone_routes)) / 10.0,
#             float(self.current_time) / 10000.0,
#             float(self.current_distance) / 50000.0,
#             float(self.current_route_demand) / 2.0,
#             float(self.drone_energy_remaining) / self.config.drone.battery_power,
#             float(self.last_location) / self.n_customers,
#             float(self.step_count) / (self.n_customers * 2)
#         ])

#         obs_array = np.array(obs, dtype=np.float32)
#         assert obs_array.ndim == 1, f"[ERROR] Obs shape invalid: {obs_array.shape}"
#         return obs_array

    
#     def _get_action_mask(self) -> np.ndarray:
#         """Tạo mask cho actions hợp lệ"""
#         mask = np.ones(self.action_space.n, dtype=bool)

#         # if hasattr(self, "last_location") and self.last_location != 0:
#         #     mask[self.last_location] = False  # không chọn lại chính node đó

#         # # Nếu tất cả đều False (đã xong tour) → ít nhất mở depot
#         # if not np.any(mask):
#         #     mask[:] = False
#         #     mask[0] = True  # return to depot
        
#         # Action 0 (depot) luôn hợp lệ để kết thúc route
        
#         for i in range(self.n_customers):
#             customer_id = i + 1
            
#             # Nếu đã visit thì không hợp lệ
#             if self.visited[i]:
#                 mask[customer_id] = False
#                 continue
            
#             # Nếu drone và customer chỉ cho truck
#             if self.current_vehicle == 1 and customer_id in self.truck_only:
#                 mask[customer_id] = False
#                 continue
            
#             # Kiểm tra capacity của drone
#             if self.current_vehicle == 1:
#                 current_route = self.drone_routes[-1] if self.drone_routes else []
#                 test_route = current_route + [customer_id]
                
#                 # Safe demand extraction with bounds check
#                 test_demands = []
#                 for c in test_route:
#                     if 0 < c <= len(self.demands):
#                         test_demands.append(self.demands[c-1])
                
#                 if not test_demands or not self.calculator.check_drone_capacity(test_route, test_demands):
#                     mask[customer_id] = False
#                     continue
                
#                 # THÊM MỚI: Kiểm tra năng lượng có đủ không
#                 # Estimate energy để đi đến customer này và về depot
#                 if self.last_location < len(self.calculator.M) and customer_id < len(self.calculator.M[0]):
#                     distance_to_customer = self.calculator.M[self.last_location][customer_id]
#                     distance_to_depot = self.calculator.M[customer_id][0]
                    
#                     # Tính energy cần (simplified)
#                     weight = sum(test_demands)
#                     power = self.calculator.drone_config.beta * weight + self.calculator.drone_config.gamma
#                     time_needed = (distance_to_customer + distance_to_depot) / self.calculator.drone_config.cruise_speed
#                     energy_needed = power * time_needed
                    
#                     if energy_needed > self.drone_energy_remaining:
#                         mask[customer_id] = False

#         if self.step_count % 500 == 0:
#             print(f"[DEBUG] Mask valid actions: {np.sum(mask)}")

#         if not np.all(self.visited):
#             if self.current_vehicle == 0:
#                 current_route = self.truck_routes[self.current_vehicle_idx]
#             else:
#                 current_route = self.drone_routes[-1] if self.drone_routes else []
#             if len(current_route) == 0:
#                 mask[0] = False


#         print(f"[DEBUG MASK] valid actions: {np.sum(mask)} / {len(mask)}")
                
#         return mask

#     def get_action_mask(self) -> np.ndarray:
#         """Hàm public cho MaskablePPO wrapper"""
#         return self._get_action_mask()


    
#     def step(self, action: int):
#         """Execute action"""
#         print(f"[DEBUG] M.shape={self.calculator.M.shape}, last_location={self.last_location}, customer_id={action}")
#         print(f"[DEBUG] Current tracked route: {self.visited}")
#         self.step_count += 1
        
#         # Action 0 = kết thúc route hiện tại
#         if action == 0:
#             print(f"[DEBUG]: End of Route.")
#             reward = self._finish_current_route()
#             done = np.all(self.visited)
            
#             if not done:
#                 self._switch_vehicle()
#         else:
#             # Thêm customer vào route
#             customer_id = action
#             customer_idx = customer_id - 1
            
#             if self.visited[customer_idx]:
#                 # Invalid action penalty
#                 reward = -10.0
#                 done = False
#             else:
#                 # THÊM MỚI: Update tracking state
#                 if self.last_location < len(self.calculator.M) and customer_id < len(self.calculator.M[0]):
#                     # distance_traveled = self.calculator.M[self.last_location][customer_id]
#                     if isinstance(self.last_location, (list, np.ndarray)):
#                         print(f"[DEBUG]: last_location={self.last_location}")
#                         last_loc = int(self.last_location[0])  
#                     else:
#                         print(f"[DEBUG]: last_location_type={type(self.last_location)}")
#                         last_loc = int(self.last_location)
#                     distance_traveled = self.calculator.M[last_loc][customer_id]

#                     self.current_distance += distance_traveled
                    
#                     # Safe demand access
#                     if customer_idx < len(self.demands):
#                         self.current_route_demand += self.demands[customer_idx]
                    
#                     # Update time (simplified - actual calculation in route_calculator)
#                     if self.current_vehicle == 0:  # Truck
#                         speed = self.config.truck.v_max * 0.7  # Estimate
#                         if speed > 0:
#                             self.current_time += distance_traveled / speed
#                     else:  # Drone
#                         if self.config.drone.cruise_speed > 0:
#                             self.current_time += distance_traveled / self.config.drone.cruise_speed
                            
#                             # Update drone energy
#                             weight = self.current_route_demand
#                             power = self.calculator.drone_config.beta * weight + self.calculator.drone_config.gamma
#                             time_segment = distance_traveled / self.config.drone.cruise_speed
#                             self.drone_energy_remaining -= power * time_segment
                    
#                     self.last_location = customer_id
                
#                 self.visited[customer_idx] = True
                
#                 if self.current_vehicle == 0:  # Truck
#                     self.truck_routes[self.current_vehicle_idx].append(customer_id)
#                 else:  # Drone
#                     if not self.drone_routes or len(self.drone_routes[-1]) == 0:
#                         self.drone_routes.append([customer_id])
#                     else:
#                         self.drone_routes[-1].append(customer_id)
                
#                 reward = -0.1  # Small penalty for each step
#                 done = np.all(self.visited)
        
#         # Final reward khi hoàn thành
#         if done:
#             reward += self._calculate_final_reward()
        
#         obs = self._get_observation()
#         info = {
#             'action_mask': self.get_action_mask(),
#             'truck_routes': self.truck_routes,
#             'drone_routes': self.drone_routes,
#             'current_time': self.current_time,
#             'current_distance': self.current_distance,
#             'drone_energy': self.drone_energy_remaining
#         }
#         if self.step_count % 1000 == 0:
#             print(f"[DEBUG] Step {self.step_count}: visited={np.sum(self.visited)}/{self.n_customers}, done={done} \n Action: {action}")

#         terminated = np.all(self.visited)
#         truncated = False
#         return obs, reward, terminated, truncated, info
        
#         # return obs, reward, done, info
    
#     def _switch_vehicle(self):
#         """Chuyển sang vehicle khác"""
#         # THÊM MỚI: Reset tracking state khi đổi vehicle
#         self.current_distance = 0.0
#         self.current_route_demand = 0.0
#         self.last_location = 0
        
#         if self.current_vehicle == 0:  # Truck -> Drone
#             if self.current_vehicle_idx < self.config.problem.num_trucks - 1:
#                 self.current_vehicle_idx += 1
#             else:
#                 self.current_vehicle = 1
#                 self.current_vehicle_idx = 0
#                 if not self.drone_routes:
#                     self.drone_routes.append([])
#                 # Reset drone energy khi bắt đầu trip mới
#                 self.drone_energy_remaining = self.config.drone.battery_power
#         else:  # Drone -> next drone trip
#             self.drone_routes.append([])
#             # Reset drone energy cho trip mới
#             self.drone_energy_remaining = self.config.drone.battery_power
    
#     def _finish_current_route(self) -> float:
#         """Kết thúc route hiện tại và tính reward"""
#         if self.current_vehicle == 0:  # Truck
#             route = self.truck_routes[self.current_vehicle_idx]
#             if route:
#                 time, wait = self.calculator.calculate_truck_time(route)
#                 return -0.01 * (time + wait)
#         else:  # Drone
#             route = self.drone_routes[-1] if self.drone_routes else []
#             if route:
#                 demands = [self.demands[c-1] for c in route]
#                 if not self.calculator.check_drone_energy(route, demands):
#                     return -100.0  # Penalty for infeasible
#                 time, wait, feasible = self.calculator.calculate_drone_time(route)
#                 if not feasible:
#                     return -100.0
#                 return -0.01 * (time + wait)
        
#         return 0.0
    
#     def _calculate_final_reward(self) -> float:
#         """Tính reward cuối cùng dựa trên multi-objective"""
#         # Tính completion time
#         truck_times = []
#         for route in self.truck_routes:
#             if route:
#                 time, _ = self.calculator.calculate_truck_time(route)
#                 truck_times.append(time)
        
#         drone_times = []
#         for route in self.drone_routes:
#             if route:
#                 time, _, _ = self.calculator.calculate_drone_time(route)
#                 drone_times.append(time)
        
#         all_times = truck_times + drone_times
#         completion_time = max(all_times) if all_times else 0
        
#         # Tính total waiting time
#         total_wait = 0.0
#         for route in self.truck_routes:
#             if route:
#                 _, wait = self.calculator.calculate_truck_time(route)
#                 total_wait += wait
        
#         for route in self.drone_routes:
#             if route:
#                 _, wait, _ = self.calculator.calculate_drone_time(route)
#                 total_wait += wait
        
#         # Multi-objective reward with weights
#         w1 = self.config.rl.w_completion_time
#         w2 = self.config.rl.w_waiting_time
        
#         # Normalize and negate (minimize -> maximize)
#         reward = -(w1 * completion_time / 10000.0 + w2 * total_wait / 10000.0)

#         if not np.isfinite(completion_time) or not np.isfinite(total_wait):
#             print("[WARN] NaN/inf in reward", completion_time, total_wait)

        
#         return reward
    
#     def render(self, mode='human'):
#         """Render environment state"""
#         print(f"\n=== Step {self.step_count} ===")
#         print(f"Current vehicle: {'Truck' if self.current_vehicle == 0 else 'Drone'} {self.current_vehicle_idx}")
#         print(f"Visited: {np.sum(self.visited)}/{self.n_customers}")
#         print(f"Truck routes: {self.truck_routes}")
#         print(f"Drone routes: {self.drone_routes}")


"""
Gym Environment for Parallel VRPD (Gymnasium API)
- Observation: 1D vector gồm đặc trưng khách + đặc trưng toàn cục
- Action space: Discrete(n_customers + 1), trong đó:
    0   = quay về depot / kết thúc route hiện tại
    1..n= chọn khách hàng (id = action - 1 trong mảng visited)
- Masking:
    * Không cho chọn khách đã visited
    * Depot (0) CHỈ mở khi current_route có >=1 node (tránh agent spam end-route ngay bước đầu)
    * Với drone: chặn khách truck-only, kiểm tra năng lượng/capacity (mức đơn giản an toàn)
- API:
    reset(...) -> (obs, info)
    step(a)    -> (obs, reward, terminated, truncated, info)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

# Bạn đã có sẵn các module này trong project:
from data_loader import DataLoader
from route_calculator import RouteCalculator
from config import ConfigManager


class ParallelVRPDEnv(gym.Env):
    """
    Environment cho bài toán Parallel VRPD
    State: [customer_features, global_features]
    Action: 0 = end route (về depot), 1..n_customers = chọn khách tiếp theo
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------
    def __init__(
        self,
        data_loader: DataLoader,
        config_manager: ConfigManager,
        debug_every: int = 500,
    ):
        super().__init__()

        self.data: DataLoader = data_loader
        self.config: ConfigManager = config_manager
        self.n_customers: int = len(data_loader.customers)
        self.debug_every: int = max(1, int(debug_every))

        # Tạo RouteCalculator
        self.calculator: RouteCalculator = RouteCalculator(
            distance_matrix=self.data.get_distance_matrix(),
            truck_config=self.config.truck,
            drone_config=self.config.drone,
            service_time_truck=self.data.customers[0].service_time_truck if self.n_customers > 0 else 60,
            service_time_drone=self.data.customers[0].service_time_drone if self.n_customers > 0 else 30,
        )

        # Action space: 0..n_customers
        self.action_space = spaces.Discrete(self.n_customers + 1)

        # Observation space
        # Mỗi khách: [x, y, demand, visited, truck_only, dist_to_depot] => 6
        # Global: 10 đặc trưng
        obs_dim = self.n_customers * 6 + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Dữ liệu trợ giúp
        self.demands: np.ndarray = self.data.get_customer_demands().astype(float) if self.n_customers > 0 else np.zeros(0)
        # truck_only là set các id (1..n) tương ứng action, tiện so khớp
        self.truck_only: set[int] = set(self.data.get_truck_only_customers())

        # Biến trạng thái
        self.truck_routes: List[List[int]] = []
        self.drone_routes: List[List[int]] = []
        self.visited: np.ndarray = np.zeros(self.n_customers, dtype=bool)
        self.current_vehicle: int = 0            # 0=truck, 1=drone
        self.current_vehicle_idx: int = 0        # index truck (0..num_trucks-1) hoặc drone-trip index
        self.step_count: int = 0

        # Tracking
        self.current_time: float = 0.0
        self.current_distance: float = 0.0
        self.current_route_demand: float = 0.0
        self.drone_energy_remaining: float = self.config.drone.battery_power
        self.last_location: int = 0  # 0=depot, 1..n=khách

        # Ngắt sớm nếu có số bước tối đa cấu hình (nếu không có thì đặt None)
        self.max_steps: Optional[int] = getattr(self.config.rl, "max_steps", None)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _current_route_ref(self) -> List[int]:
        """Lấy reference list của route hiện tại theo loại phương tiện."""
        if self.current_vehicle == 0:
            return self.truck_routes[self.current_vehicle_idx]
        else:
            if not self.drone_routes:
                self.drone_routes.append([])
            return self.drone_routes[-1]

    # ------------------------------------------------------------
    # Reset theo Gymnasium API
    # ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        self.truck_routes = [[] for _ in range(self.config.problem.num_trucks)]
        self.drone_routes = []
        self.visited = np.zeros(self.n_customers, dtype=bool)

        self.current_vehicle = 0
        self.current_vehicle_idx = 0
        self.step_count = 0

        self.current_time = 0.0
        self.current_distance = 0.0
        self.current_route_demand = 0.0
        self.drone_energy_remaining = self.config.drone.battery_power
        self.last_location = 0

        obs = self._get_observation()
        info = {"action_mask": self.get_action_mask()}

        print(f"[RESET] visited.sum={int(self.visited.sum())}, in_route={len(self._current_route_ref())>0}, last_location={self.last_location}")
        return obs, info

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        obs: List[float] = []

        # Đặc trưng từng khách
        depot = self.data.depot
        for i in range(self.n_customers):
            c = self.data.customers[i]
            obs.extend([
                float(c.x) / 5000.0,
                float(c.y) / 5000.0,
                float(c.demand) / 2.0,
                float(self.visited[i]),
                float(c.only_truck),
                float(c.distance_to(depot)) / 5000.0,
            ])

        # Đặc trưng toàn cục (10)
        total_vehicles = max(self.config.problem.num_trucks, self.config.problem.num_drones, 1)
        obs.extend([
            float(self.current_vehicle),
            float(self.current_vehicle_idx) / float(total_vehicles),
            float(self.visited.sum()) / float(max(self.n_customers, 1)),
            float(len(self.drone_routes)) / 10.0,
            float(self.current_time) / 10000.0,
            float(self.current_distance) / 50000.0,
            float(self.current_route_demand) / 2.0,
            float(self.drone_energy_remaining) / max(self.config.drone.battery_power, 1e-6),
            float(self.last_location) / float(max(self.n_customers, 1)),
            float(self.step_count) / float(max(self.n_customers * 2, 1)),
        ])

        obs_array = np.asarray(obs, dtype=np.float32)
        assert obs_array.ndim == 1, f"[ERROR] Obs shape invalid: {obs_array.shape}"
        return obs_array

    # ------------------------------------------------------------
    # Action Masking
    # ------------------------------------------------------------
    def _get_action_mask(self) -> np.ndarray:
        """Tạo mask cho hành động hợp lệ (bool array với shape (n_actions,))."""
        n_actions = self.action_space.n
        mask = np.ones(n_actions, dtype=bool)  # mặc định: tất cả True

        # Chặn khách đã thăm
        if self.n_customers > 0:
            visited_idx = np.where(self.visited)[0]
            mask[visited_idx + 1] = False

        # Depot chỉ mở nếu route hiện tại đã có ít nhất 1 khách
        current_route = self._current_route_ref()
        mask[0] = len(current_route) > 0

        # Nếu drone: chặn khách truck-only
        if self.current_vehicle == 1 and len(self.truck_only) > 0:
            for cid in self.truck_only:
                if 1 <= cid <= self.n_customers:
                    mask[cid] = False

        # (Tuỳ chọn) kiểm tra năng lượng drone mức tối thiểu (an toàn)
        if self.current_vehicle == 1:
            for cid in range(1, self.n_customers + 1):
                if not mask[cid]:
                    continue
                # ước lượng quãng bay đến khách + về depot
                if self.last_location <= self.n_customers:
                    last_id = self.last_location
                else:
                    last_id = 0
                M = self.calculator.M
                if last_id < M.shape[0] and cid < M.shape[1]:
                    dist_to_c = M[last_id, cid]
                    dist_to_depot = M[cid, 0]
                    speed = max(self.calculator.drone_config.cruise_speed, 1e-6)
                    time_need = (dist_to_c + dist_to_depot) / speed
                    # công suất xấp xỉ theo trọng lượng hiện tại
                    weight = max(float(self.current_route_demand), 0.0)
                    power = self.calculator.drone_config.beta * weight + self.calculator.drone_config.gamma
                    energy_need = power * time_need
                    if energy_need > self.drone_energy_remaining:
                        mask[cid] = False

        # Nếu tất cả False → mở depot để tránh kẹt
        if not np.any(mask):
            mask[:] = False
            mask[0] = True

        if self.step_count % self.debug_every == 0:
            print(
                f"[DEBUG MASK] valid={int(mask.sum())}/{n_actions} | "
                f"in_route={len(current_route)>0} | "
                f"visited={int(self.visited.sum())}/{self.n_customers} | "
                f"depot_open={bool(mask[0])}"
            )

        return mask

    def get_action_mask(self) -> np.ndarray:
        """Public mask getter cho MaskablePPO / ActionMasker."""
        mask = self._get_action_mask()
        if mask is None or mask.dtype != bool or mask.shape != (self.action_space.n,):
            raise ValueError("[ERROR] Invalid action mask shape/dtype.")
        return mask

    # ------------------------------------------------------------
    # Step theo Gymnasium API
    # ------------------------------------------------------------
    def step(self, action: int):
        # Mask kiểm tra trước: nếu invalid → phạt và giữ nguyên trạng thái
        mask = self.get_action_mask()
        if not mask[action]:
            # Hạn chế lặp invalid nhiều lần: phạt nhẹ nhưng không thay đổi route
            print(f"[WARN] Chosen action {action} was invalid under mask. Penalize.")
            obs = self._get_observation()
            reward = -10.0
            terminated = False
            truncated = False
            info = {"action_mask": mask}
            return obs, reward, terminated, truncated, info

        self.step_count += 1
        reward = -0.1
        terminated = False
        truncated = False

        # Ngắt sớm theo max_steps (nếu cấu hình)
        if self.max_steps is not None and self.step_count >= int(self.max_steps):
            truncated = True

        if action == 0:
            # Kết thúc route hiện tại, về depot
            reward += self._finish_current_route()
            terminated = bool(self.visited.all())
            if not terminated:
                self._switch_vehicle()
            else:
                reward += self._calculate_final_reward()
        else:
            # Thăm khách hàng (action 1..n -> index 0..n-1)
            cid = int(action)
            idx = cid - 1
            # Update quãng đường + thời gian + năng lượng (an toàn)
            self._accumulate_move_cost(self.last_location, cid)
            # Đánh dấu visited + push vào route
            self.visited[idx] = True
            route_ref = self._current_route_ref()
            route_ref.append(cid)
            self.last_location = cid
            # Nếu đã xong hết khách → kết thúc khi quay về depot (tuỳ luật)
            if self.visited.all():
                # Khuyến khích kết thúc sớm bằng depot
                reward += 0.0

        if self.step_count % (self.debug_every * 2) == 0:
            print(
                f"[DEBUG] Step {self.step_count}: action={action} | "
                f"visited={int(self.visited.sum())}/{self.n_customers} | "
                f"term={terminated} | trunc={truncated}"
            )

        obs = self._get_observation()
        info = {
            "action_mask": self.get_action_mask(),
            "truck_routes": self.truck_routes,
            "drone_routes": self.drone_routes,
            "current_time": self.current_time,
            "current_distance": self.current_distance,
            "drone_energy": self.drone_energy_remaining,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Chi phí di chuyển & cập nhật tracking
    # ------------------------------------------------------------
    def _accumulate_move_cost(self, last_node: int, next_node: int) -> None:
        """Cộng dồn quãng đường, thời gian, năng lượng theo bước đi từ last_node -> next_node."""
        M = self.calculator.M
        if last_node < 0 or last_node >= M.shape[0]:
            last_node = 0
        if next_node < 0 or next_node >= M.shape[1]:
            next_node = 0
        distance = float(M[last_node, next_node])
        self.current_distance += distance

        if self.current_vehicle == 0:
            # Truck: thời gian ~ distance / speed
            speed = max(self.config.truck.v_max * 0.7, 1e-6)
            self.current_time += distance / speed
            # Cập nhật nhu cầu (nếu next_node là khách)
            if 1 <= next_node <= self.n_customers:
                self.current_route_demand += float(self.demands[next_node - 1])
        else:
            # Drone: thời gian theo cruise_speed + trừ năng lượng
            v = max(self.config.drone.cruise_speed, 1e-6)
            dt = distance / v
            self.current_time += dt
            weight = max(self.current_route_demand, 0.0)
            power = self.calculator.drone_config.beta * weight + self.calculator.drone_config.gamma
            self.drone_energy_remaining -= power * dt
            # Drone vẫn có thể phục vụ khách (cộng nhu cầu) nếu là khách
            if 1 <= next_node <= self.n_customers:
                self.current_route_demand += float(self.demands[next_node - 1])

    # ------------------------------------------------------------
    # Kết thúc route hiện tại
    # ------------------------------------------------------------
    def _finish_current_route(self) -> float:
        """Kết thúc route hiện tại và tính phần thưởng phụ thuộc route."""
        if self.current_vehicle == 0:
            # Truck route
            route = self.truck_routes[self.current_vehicle_idx]
            if route:
                t, w = self.calculator.calculate_truck_time(route)
                # thưởng âm theo thời gian + chờ
                return -0.01 * float(t + w)
            return 0.0
        else:
            # Drone route
            route = self.drone_routes[-1] if self.drone_routes else []
            if route:
                demands = [float(self.demands[c - 1]) for c in route if 1 <= c <= self.n_customers]
                if not demands:
                    return 0.0
                # Kiểm tra ràng buộc năng lượng
                if not self.calculator.check_drone_energy(route, demands):
                    return -100.0
                t, w, feasible = self.calculator.calculate_drone_time(route)
                if not feasible:
                    return -100.0
                return -0.01 * float(t + w)
            return 0.0

    # ------------------------------------------------------------
    # Đổi phương tiện / mở route mới
    # ------------------------------------------------------------
    def _switch_vehicle(self) -> None:
        """Chuyển sang phương tiện/route tiếp theo và reset tracking của route."""
        # Reset tracking cho route mới
        self.current_distance = 0.0
        self.current_route_demand = 0.0
        self.last_location = 0

        if self.current_vehicle == 0:
            # Truck -> truck tiếp theo, hoặc chuyển sang drone
            if self.current_vehicle_idx < self.config.problem.num_trucks - 1:
                self.current_vehicle_idx += 1
            else:
                self.current_vehicle = 1
                self.current_vehicle_idx = 0
                if not self.drone_routes:
                    self.drone_routes.append([])
                self.drone_energy_remaining = self.config.drone.battery_power
        else:
            # Drone -> bắt đầu trip drone mới
            self.drone_routes.append([])
            self.drone_energy_remaining = self.config.drone.battery_power

    # ------------------------------------------------------------
    # Phần thưởng cuối cùng khi hoàn tất
    # ------------------------------------------------------------
    def _calculate_final_reward(self) -> float:
        """Tính phần thưởng cuối cùng theo multi-objective (hoàn tất episode)."""
        truck_times: List[float] = []
        for r in self.truck_routes:
            if r:
                t, _ = self.calculator.calculate_truck_time(r)
                truck_times.append(float(t))

        drone_times: List[float] = []
        for r in self.drone_routes:
            if r:
                t, _, _ = self.calculator.calculate_drone_time(r)
                drone_times.append(float(t))

        all_times = truck_times + drone_times
        completion_time = float(max(all_times)) if all_times else 0.0

        total_wait = 0.0
        for r in self.truck_routes:
            if r:
                _, w = self.calculator.calculate_truck_time(r)
                total_wait += float(w)
        for r in self.drone_routes:
            if r:
                _, w, _ = self.calculator.calculate_drone_time(r)
                total_wait += float(w)

        w1 = float(getattr(self.config.rl, "w_completion_time", 1.0))
        w2 = float(getattr(self.config.rl, "w_waiting_time", 1.0))

        reward = -(w1 * (completion_time / 10000.0) + w2 * (total_wait / 10000.0))

        if not np.isfinite(completion_time) or not np.isfinite(total_wait):
            print(f"[WARN] NaN/inf in reward completion={completion_time}, wait={total_wait}")

        return float(reward)

    # ------------------------------------------------------------
    # Render (debug)
    # ------------------------------------------------------------
    def render(self):
        print(f"\n=== Step {self.step_count} ===")
        print(f"Vehicle: {'Truck' if self.current_vehicle == 0 else 'Drone'} idx={self.current_vehicle_idx}")
        print(f"Visited: {int(self.visited.sum())}/{self.n_customers}")
        print(f"Truck routes: {self.truck_routes}")
        print(f"Drone routes: {self.drone_routes}")
        print(f"time={self.current_time:.2f}, dist={self.current_distance:.2f}, droneE={self.drone_energy_remaining:.2f}")
