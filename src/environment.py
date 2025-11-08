"""
Parallel VRPD Environment - Multi-objective optimization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Any


class ParallelVRPDEnv(gym.Env):
    """
    Parallel VRPD Environment - Multi-objective optimization
    
    OBJECTIVES (to minimize):
    1. Total Service Time: Max time when all vehicles return to depot
    2. Total Waiting Time: Sum of waiting times calculated when vehicle returns to depot
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_customers: int,
        coords: np.ndarray,
        demands: np.ndarray,
        truck_only: np.ndarray,
        capacity: float,
        calculator: Optional[Any] = None, 
        use_drone: bool = True,
        n_drones: int = 1,
        n_trucks: int = 1,
        drone_battery: float = 0.0,
        max_steps: int = 10_000,
        # Objective weights
        w_service: float = 0.5,
        w_wait: float = 0.5,
        # Reward shaping
        enable_shaping: bool = True,
        step_penalty: float = 0.1,
        progress_reward: float = 1.0,
        idle_penalty: float = 0.5,
        # Misc
        include_weight_in_obs: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()

        assert coords.shape == (n_customers + 1, 2), "coords must include depot at index 0"
        assert demands.shape == (n_customers,), "demands shape must be (N,)"
        assert truck_only.shape == (n_customers,), "truck_only shape must be (N,)"
        
        if calculator is None:
            raise ValueError(
                "Calculator must be provided for proper time and energy calculation. "
                "Use TimeDistanceCalculator from calculator.py"
            )

        # Problem data
        self.N = int(n_customers)
        self.coords = coords.astype(np.float32)
        self.demands = demands.astype(np.float32)
        self.truck_only = truck_only.astype(bool)
        self.capacity = float(capacity)

        # Vehicles
        self.use_drone = bool(use_drone)
        self.D = int(n_drones) if (self.use_drone and n_drones > 0) else 0
        self.T = int(n_trucks) if n_trucks > 0 else 1
        self.drone_battery = float(drone_battery)

        self.V = self.D + self.T
        assert self.V > 0, "Need at least one vehicle"

        # Objective weights
        self.w_service = float(w_service)
        self.w_wait = float(w_wait)
        weight_sum = self.w_service + self.w_wait
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights must sum to 1.0, got {weight_sum}"

        # Reward shaping
        self.enable_shaping = enable_shaping
        self.step_penalty = float(step_penalty)
        self.progress_reward = float(progress_reward)
        self.idle_penalty = float(idle_penalty)
        self.include_weight_in_obs = include_weight_in_obs

        # Episode config
        self.max_steps = int(max_steps)
        self.step_count = 0

        self.calculator = calculator 
        self.debug = debug

        # Spaces
        self.action_space = spaces.Discrete(self.V * (self.N + 1))

        # Observation dims
        self.customer_feat_dim = 6
        self.vehicle_feat_dim = 5
        base_global_dim = 1
        weight_dim = 2 if self.include_weight_in_obs else 0
        self.global_dim = base_global_dim + weight_dim + self.V * self.vehicle_feat_dim

        obs_dim = self.N * self.customer_feat_dim + self.global_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._rng = None
        
        # Metrics tracking
        self.episode_metrics = {
            "total_service_time": 0.0,
            "total_waiting_time": 0.0,
            "total_travel_distance": 0.0,
        }
        
        self.consecutive_idles = 0
        self.max_consecutive_idles = 20
        
        self.reset(seed=None)

    def _norm_coords(self, xy: np.ndarray) -> np.ndarray:
        mins = self.coords.min(axis=0)
        maxs = self.coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        return (xy - mins) / span

    def _get_distance(self, i: int, j: int) -> float:
        """Get Euclidean distance between nodes i and j (in meters)"""
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def _get_truck_travel_time(self, i: int, j: int, current_time: float = None) -> float:
        """Get truck travel time from node i to j (in seconds)"""
        if current_time is None:
            current_time = 0.0
        
        if hasattr(self.calculator, 'get_truck_travel_time'):
            return float(self.calculator.get_truck_travel_time(i, j, current_time))
        else:
            distance = self._get_distance(i, j)
            return distance / 15.557

    def _get_drone_travel_time(self, i: int, j: int, demand: float = 0.0) -> float:
        """Get drone travel time from node i to j with given demand (in seconds)"""
        if hasattr(self.calculator, 'get_drone_travel_time'):
            return float(self.calculator.get_drone_travel_time(i, j, demand))
        else:
            distance = self._get_distance(i, j)
            return distance / 31.2928

    def _get_drone_energy(self, i: int, j: int, demand: float = 0.0) -> float:
        """Get drone energy consumption from node i to j with given demand (in Joules)"""
        if hasattr(self.calculator, 'get_drone_energy'):
            return float(self.calculator.get_drone_energy(i, j, demand))
        else:
            distance = self._get_distance(i, j)
            return distance * 100.0

    def _get_service_time(self, is_drone: bool) -> float:
        """Get service time for vehicle type (in seconds)"""
        if hasattr(self.calculator, 'service_time_drone') and hasattr(self.calculator, 'service_time_truck'):
            return float(self.calculator.service_time_drone if is_drone else self.calculator.service_time_truck)
        else:
            return 30.0 if is_drone else 60.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.consecutive_idles = 0
        
        self._reserved_customers = set()

        # Optional: override weights per-episode
        if options is not None:
            w = options.get("weights")
            if w is not None:
                if isinstance(w, dict):
                    self.w_service = float(w.get("w_service", self.w_service))
                    self.w_wait = float(w.get("w_wait", self.w_wait))
                elif isinstance(w, (list, tuple)) and len(w) == 2:
                    self.w_service, self.w_wait = map(float, w)
                weight_sum = self.w_service + self.w_wait
                if weight_sum > 0:
                    self.w_service /= weight_sum
                    self.w_wait /= weight_sum

        # Customers
        self.visited = np.zeros(self.N, dtype=bool)
        self.customers_left = self.N
        self.service_time = np.full(self.N, np.nan, dtype=np.float32)

        # Time
        self.current_time = 0.0

        # Vehicles
        self.veh_is_drone = np.zeros(self.V, dtype=bool)
        if self.D > 0:
            self.veh_is_drone[:self.D] = True

        self.veh_node = np.zeros(self.V, dtype=np.int32)
        self.veh_time = np.zeros(self.V, dtype=np.float32)  
        self.veh_energy = np.zeros(self.V, dtype=np.float32)
        if self.D > 0:
            self.veh_energy[:self.D] = self.drone_battery

        self.truck_loads = np.zeros(self.T, dtype=np.float32)

        # Track current trip customers for waiting time calculation
        self.current_trip_customers: List[List[int]] = [[] for _ in range(self.V)]
        
        self.drone_routes: List[List[int]] = [[] for _ in range(self.D)]
        self.truck_routes: List[List[int]] = [[] for _ in range(self.T)]

        # Accumulate waiting time as vehicles return to depot
        self.accumulated_waiting_time = 0.0

        # Reset metrics
        self.episode_metrics = {
            "total_service_time": 0.0,
            "total_waiting_time": 0.0,
            "total_travel_distance": 0.0,
        }

        obs = self._get_observation()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def _get_observation(self) -> np.ndarray:
        coords_norm = self._norm_coords(self.coords)
        depot = coords_norm[0]

        # Customers
        feat = []
        for i in range(self.N):
            xy = coords_norm[i + 1]
            demand_norm = float(self.demands[i] / max(self.capacity, 1e-6))
            visited = 1.0 if self.visited[i] else 0.0
            only_truck = 1.0 if self.truck_only[i] else 0.0
            d_depot = float(np.linalg.norm(xy - depot))
            feat.extend([xy[0], xy[1], demand_norm, visited, only_truck, d_depot])

        # Vehicles
        vfeat = []
        for v in range(self.V):
            is_drone = 1.0 if self.veh_is_drone[v] else 0.0
            node = int(self.veh_node[v])
            xy = coords_norm[node]
            if self.veh_is_drone[v]:
                if self.drone_battery > 0:
                    energy_norm = float(self.veh_energy[v] / max(self.drone_battery, 1e-6))
                else:
                    energy_norm = 0.0
                load_norm = 0.0
            else:
                truck_idx = v - self.D
                energy_norm = 0.0
                load_norm = float(self.truck_loads[truck_idx] / max(self.capacity, 1e-6))
            vfeat.extend([is_drone, xy[0], xy[1], energy_norm, load_norm])

        time_norm = float(np.tanh(self.current_time / 1e4))
        g = [time_norm, *vfeat]

        if self.include_weight_in_obs:
            g.extend([float(self.w_service), float(self.w_wait)])

        obs = np.array([*feat, *g], dtype=np.float32)
        assert obs.shape == self.observation_space.shape
        return obs

    def _feasible_for_drone(self, v: int, cid: int) -> bool:
        if self.visited[cid]:
            return False
        
        if hasattr(self, '_reserved_customers') and cid in self._reserved_customers:
            return False
            
        if self.truck_only[cid]:
            return False

        cur = int(self.veh_node[v])
        target = cid + 1
        if cur == target:
            return False

        demand = float(self.demands[cid])
        energy_needed = self._get_drone_energy(cur, target, demand)
        if self.veh_energy[v] + 1e-9 < energy_needed:
            return False

        if hasattr(self.calculator, "feasible_drone_add"):
            if not self.calculator.feasible_drone_add(self.drone_routes[v], target):
                return False

        return True

    def _feasible_for_truck(self, truck_veh_id: int, cid: int) -> bool:
        if self.visited[cid]:
            return False
        
        if hasattr(self, '_reserved_customers') and cid in self._reserved_customers:
            return False

        truck_node = int(self.veh_node[truck_veh_id])
        target = cid + 1
        if truck_node == target:
            return False

        truck_idx = truck_veh_id - self.D
        new_load = self.truck_loads[truck_idx] + float(self.demands[cid])
        if new_load - 1e-9 > self.capacity:
            return False

        if hasattr(self.calculator, "feasible_truck_add"):
            if not self.calculator.feasible_truck_add(self.truck_routes[truck_idx], target):
                return False

        return True

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.customers_left == 0:
            return mask

        for v in range(self.V):
            base = v * (self.N + 1)
            node = int(self.veh_node[v])

            if node != 0 or self.consecutive_idles < self.max_consecutive_idles:
                mask[base + 0] = True

            for cid in range(self.N):
                if self.visited[cid]:
                    continue
                if self.veh_is_drone[v]:
                    if self._feasible_for_drone(v, cid):
                        mask[base + (cid + 1)] = True
                else:
                    if self._feasible_for_truck(v, cid):
                        mask[base + (cid + 1)] = True

        return mask

    def _calculate_trip_waiting_time(self, veh_id: int, return_time: float) -> float:
        """
        Calculate waiting time for current trip when vehicle returns to depot
        
        Args:
            veh_id: Vehicle ID
            return_time: Time when vehicle returns to depot
            
        Returns:
            Total waiting time for customers in current trip
        """
        trip_waiting = 0.0
        
        for cid in self.current_trip_customers[veh_id]:
            service_time_i = self.service_time[cid]
            if not np.isnan(service_time_i):
                waiting_time_i = return_time - service_time_i
                trip_waiting += waiting_time_i
        
        return trip_waiting

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        if self.debug:
            print(f"\n[STEP {self.step_count}] action={action}")

        mask = self.get_action_mask()
        if action < 0 or action >= self.action_space.n or not mask[action]:
            reward = -1e4
            if self.debug:
                print(f"  ❌ INVALID ACTION {action}, valid={np.where(mask)[0]}")
            obs = self._get_observation()
            info = {
                "action_mask": mask,
                "error": "invalid_action",
                "time": self.current_time,
                "customers_left": int(self.customers_left),
            }
            return obs, float(reward), True, False, info

        veh_id = action // (self.N + 1)
        choice = action % (self.N + 1)
        cur_node = int(self.veh_node[veh_id])

        travel_dist = 0.0
        travel_time = 0.0
        served_customer = False
        is_idle_action = False

        # Execute action
        if choice == 0:
            # Return to depot or noop
            if cur_node != 0:
                travel_dist = self._get_distance(cur_node, 0)
                
                if self.veh_is_drone[veh_id]:
                    travel_time = self._get_drone_travel_time(cur_node, 0, 0.0)
                    energy_used = self._get_drone_energy(cur_node, 0, 0.0)
                    self.veh_energy[veh_id] = max(0.0, self.veh_energy[veh_id] - energy_used)
                    self.veh_energy[veh_id] = self.drone_battery
                else:
                    travel_time = self._get_truck_travel_time(
                        cur_node, 0, 
                        current_time=self.veh_time[veh_id]
                    )
                    truck_idx = veh_id - self.D
                    self.truck_loads[truck_idx] = 0.0
                
                self.veh_node[veh_id] = 0
                self.veh_time[veh_id] += travel_time
                
                # CALCULATE WAITING TIME WHEN RETURNING TO DEPOT
                if len(self.current_trip_customers[veh_id]) > 0:
                    trip_waiting = self._calculate_trip_waiting_time(
                        veh_id, 
                        self.veh_time[veh_id]
                    )
                    self.accumulated_waiting_time += trip_waiting
                    
                    if self.debug:
                        print(f"  🏠 Vehicle-{veh_id} returned to depot at t={self.veh_time[veh_id]:.1f}s")
                        print(f"     Trip waiting time: {trip_waiting:.1f}s")
                        print(f"     Accumulated waiting: {self.accumulated_waiting_time:.1f}s")
                    
                    # Clear current trip
                    self.current_trip_customers[veh_id] = []
                
                if self.customers_left > 0 and self.enable_shaping:
                    reward -= self.idle_penalty
                    is_idle_action = True
            else:
                is_idle_action = True
                if self.customers_left > 0 and self.enable_shaping:
                    reward -= self.idle_penalty * 2

        else:
            cid = choice - 1
            target = cid + 1
            demand = float(self.demands[cid])
            
            if self.visited[cid]:
                reward = -1e4
                terminated = True
                if self.debug:
                    print(f"  ❌ Customer {cid} already visited!")
                obs = self._get_observation()
                info = {
                    "action_mask": self.get_action_mask(),
                    "error": "customer_already_visited",
                    "time": self.current_time,
                    "customers_left": int(self.customers_left),
                }
                return obs, float(reward), terminated, truncated, info

            if self.veh_is_drone[veh_id]:
                if not self._feasible_for_drone(veh_id, cid):
                    reward = -1e4
                    terminated = True
                    if self.debug:
                        print(f"  ❌ Drone-{veh_id} infeasible for cid={cid}")
                else:
                    travel_dist = self._get_distance(cur_node, target)
                    travel_time = self._get_drone_travel_time(cur_node, target, demand)
                    service_time = self._get_service_time(is_drone=True)
                    energy_used = self._get_drone_energy(cur_node, target, demand)
                    
                    self.visited[cid] = True
                    self.customers_left -= 1
                    
                    self.veh_energy[veh_id] -= energy_used
                    self.veh_node[veh_id] = target
                    self.veh_time[veh_id] += travel_time + service_time
                    
                    # Track service time and add to current trip
                    self.service_time[cid] = self.veh_time[veh_id]
                    self.current_trip_customers[veh_id].append(cid)
                    
                    self.drone_routes[veh_id].append(cid)
                    served_customer = True
                    
                    if self.debug:
                        print(f"  ✅ Drone-{veh_id} served cid={cid}, "
                              f"travel_time={travel_time:.1f}s, service_time={service_time:.1f}s, "
                              f"energy_used={energy_used:.1f}J")

            else:
                truck_idx = veh_id - self.D
                
                if not self._feasible_for_truck(veh_id, cid):
                    reward = -1e4
                    terminated = True
                    if self.debug:
                        print(f"  ❌ Truck-{truck_idx} infeasible for cid={cid}")
                else:
                    travel_dist = self._get_distance(cur_node, target)
                    
                    travel_time = self._get_truck_travel_time(
                        cur_node, target,
                        current_time=self.veh_time[veh_id]
                    )
                    service_time = self._get_service_time(is_drone=False)
                    
                    self.visited[cid] = True
                    self.customers_left -= 1
                    
                    self.veh_node[veh_id] = target
                    self.truck_loads[truck_idx] += demand
                    self.veh_time[veh_id] += travel_time + service_time
                    
                    # Track service time and add to current trip
                    self.service_time[cid] = self.veh_time[veh_id]
                    self.current_trip_customers[veh_id].append(cid)
                    
                    self.truck_routes[truck_idx].append(cid)
                    served_customer = True
                    
                    if self.debug:
                        print(f"  ✅ Truck-{truck_idx} served cid={cid}, "
                              f"travel_time={travel_time:.1f}s, service_time={service_time:.1f}s")

        # Track consecutive idle actions
        if is_idle_action:
            self.consecutive_idles += 1
        else:
            self.consecutive_idles = 0

        # Update metrics
        self.episode_metrics["total_travel_distance"] += travel_dist
        self.current_time = float(np.max(self.veh_time))

        # Reward shaping
        if self.enable_shaping:
            reward -= self.step_penalty
            if served_customer:
                reward += self.progress_reward
                reward += 0.1 * (1.0 - self.current_time / 10000.0)

        # Terminal conditions
        if self.customers_left == 0:
            # Make all vehicles to return to depot if not already there
            for v in range(self.V):
                if self.veh_node[v] != 0:
                    # Calculate return time
                    cur_node = int(self.veh_node[v])
                    
                    if self.veh_is_drone[v]:
                        return_time = self._get_drone_travel_time(cur_node, 0, 0.0)
                    else:
                        return_time = self._get_truck_travel_time(
                            cur_node, 0, 
                            current_time=self.veh_time[v]
                        )
                    
                    self.veh_time[v] += return_time
                    
                    # Calculate waiting time for remaining customers in trip
                    if len(self.current_trip_customers[v]) > 0:
                        trip_waiting = self._calculate_trip_waiting_time(v, self.veh_time[v])
                        self.accumulated_waiting_time += trip_waiting
                        self.current_trip_customers[v] = []
            
            # Final objectives
            total_service_time = float(np.max(self.veh_time))  
            total_waiting_time = float(self.accumulated_waiting_time)
            
            self.episode_metrics["total_service_time"] = total_service_time
            self.episode_metrics["total_waiting_time"] = total_waiting_time
            
            objective_reward = -(
                self.w_service * total_service_time + 
                self.w_wait * total_waiting_time
            )
            reward += objective_reward
            
            terminated = True
            if self.debug:
                print(f"  🎉 DONE: service_time={total_service_time:.2f}s, "
                      f"wait_time={total_waiting_time:.2f}s, R={reward:.2f}")

        if not terminated:
            next_mask = self.get_action_mask()
            if not next_mask.any():
                remaining = int(self.customers_left)
                avg_time = 120.0
                estimated_time = remaining * avg_time
                
                penalty = (
                    self.w_service * (self.current_time + estimated_time) +
                    self.w_wait * (self.accumulated_waiting_time + estimated_time * remaining)
                )
                reward -= penalty
                terminated = True
                if self.debug:
                    print(f"  ⚠️ DEADLOCK: remaining={remaining}, penalty={penalty:.2f}")
            
            elif self.step_count >= self.max_steps:
                remaining = int(self.customers_left)
                avg_time = 120.0
                estimated_time = remaining * avg_time
                
                penalty = (
                    self.w_service * (self.current_time + estimated_time) +
                    self.w_wait * (self.accumulated_waiting_time + estimated_time * remaining)
                )
                reward -= penalty
                truncated = True
                if self.debug:
                    print(f"  ⏱️ TIMEOUT: remaining={remaining}, penalty={penalty:.2f}")

        obs = self._get_observation()
        info = {
            "action_mask": self.get_action_mask(),
            "time": self.current_time,
            "customers_left": int(self.customers_left),
            "customers_visited": int(self.N - self.customers_left),
            "completion_rate": float((self.N - self.customers_left) / self.N),
            "step_count": int(self.step_count),
            "drone_routes": self.drone_routes,
            "truck_routes": self.truck_routes,
            "weights": {
                "w_service": self.w_service,
                "w_wait": self.w_wait,
            },
            "total_service_time": self.episode_metrics["total_service_time"],
            "total_waiting_time": self.episode_metrics["total_waiting_time"],
            "total_travel_distance": self.episode_metrics["total_travel_distance"],
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        print(f"[t={self.current_time:.1f}s] step={self.step_count}, left={self.customers_left}/{self.N}")
        for v in range(self.V):
            node = self.veh_node[v]
            veh_time = self.veh_time[v]
            if self.veh_is_drone[v]:
                print(f"  DRONE-{v}: node={node}, time={veh_time:.1f}s, "
                      f"E={self.veh_energy[v]:.0f}/{self.drone_battery:.0f}J")
            else:
                truck_idx = v - self.D
                print(f"  TRUCK-{truck_idx}: node={node}, time={veh_time:.1f}s, "
                      f"load={self.truck_loads[truck_idx]:.1f}/{self.capacity:.1f}kg")