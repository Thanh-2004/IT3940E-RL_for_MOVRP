import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Any

class ParallelVRPDEnv(gym.Env):
    """
    Parallel VRPD: Drone-first optimization, then truck completes remaining customers.
    
    Design:
    - Phase 1 (Drone): All drones operate in round-robin until no drone can serve any customer
    - Phase 2 (Truck): Single truck serves ALL remaining customers
    - Constraint: Each customer visited EXACTLY ONCE (no duplicates)
    """
    metadata = {"render.modes": ["human"]}

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
        drone_battery: float = 0.0,
        max_steps: int = 1000,
        debug: bool = False
    ) -> None:
        super().__init__()
        assert coords.shape == (n_customers + 1, 2), "coords must include depot at index 0"
        assert demands.shape == (n_customers,), "demands shape must be (N,)"
        assert truck_only.shape == (n_customers,), "truck_only shape must be (N,)"

        self.N = n_customers
        self.coords = coords.astype(np.float32)
        self.demands = demands.astype(np.float32)
        self.truck_only = truck_only.astype(np.bool_)
        self.capacity = float(capacity)

        self.use_drone = bool(use_drone)
        self.D = int(max(1, n_drones)) if self.use_drone else 0
        self.drone_battery = float(drone_battery)
        
        self.max_steps = max_steps
        self.step_count = 0

        self.calculator = calculator
        self.debug = debug

        # Action/Obs spaces
        self.action_space = spaces.Discrete(self.N + 1)
        g_dim = 2 + 1 + 1 + (self.D if self.D > 0 else 1) + 1
        obs_dim = self.N * 6 + g_dim
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.reset(seed=None)

    # ---------------------- Helpers ----------------------
    def _norm_coords(self, xy: np.ndarray) -> np.ndarray:
        mins = self.coords.min(axis=0)
        maxs = self.coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        return (xy - mins) / span

    def _dist(self, i: int, j: int) -> float:
        if self.calculator is not None and hasattr(self.calculator, "distance"):
            return float(self.calculator.distance(i, j))
        a, b = self.coords[i], self.coords[j]
        return float(np.linalg.norm(a - b))

    # ---------------------- Episode init ----------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # ✅ CRITICAL: Global visited status (shared between drone & truck)
        self.visited = np.zeros(self.N, dtype=bool)
        self.customers_left = self.N
        self.current_time = 0.0
        self.step_count = 0

        # Phase: 'drone' -> 'truck'
        self.phase = 'drone' if (self.use_drone and self.D > 0) else 'truck'

        # Truck state
        self.truck_node = 0
        self.truck_routes: List[int] = []
        self.current_load = 0.0

        # Drone fleet state
        self.drone_node = np.zeros(self.D, dtype=np.int32)
        self.drone_energy = np.full(self.D, self.drone_battery, dtype=np.float32)
        self.drone_routes: List[List[int]] = [[] for _ in range(self.D)]
        self.current_drone = 0

        obs = self._get_observation()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    # ---------------------- Observation ----------------------
    def _get_observation(self) -> np.ndarray:
        coords_norm = self._norm_coords(self.coords)
        depot = coords_norm[0]

        feat = []
        for i in range(self.N):
            xy = coords_norm[i + 1]
            demand = self.demands[i] / max(self.capacity, 1e-6)
            visited = 1.0 if self.visited[i] else 0.0
            only_truck = 1.0 if self.truck_only[i] else 0.0
            d_depot = float(np.linalg.norm(xy - depot))
            feat.extend([xy[0], xy[1], demand, visited, only_truck, d_depot])

        phase_onehot = [1.0, 0.0] if self.phase == 'drone' else [0.0, 1.0]

        if self.D > 0:
            drone_onehot = [1.0 if d == self.current_drone and self.phase == 'drone' else 0.0
                            for d in range(self.D)]
            cur_energy = (self.drone_energy[self.current_drone] / max(self.drone_battery, 1e-6)
                          if self.phase == 'drone' and self.drone_battery > 0 else 0.0)
        else:
            drone_onehot = [0.0]
            cur_energy = 0.0

        g = [
            *phase_onehot,
            min(self.current_load / max(self.capacity, 1e-6), 1.0),
            np.tanh(self.current_time / 1e4),
            *drone_onehot,
            np.clip(cur_energy, 0.0, 1.0),
        ]
        return np.array([*feat, *g], dtype=np.float32)

    # ---------------------- Feasibility ----------------------
    def _feasible_truck_add(self, cid: int) -> bool:
        """Check if truck can add customer cid (0-indexed)"""
        # ✅ CRITICAL: Check visited status FIRST
        if self.visited[cid]:
            if self.debug:
                print(f"    [TRUCK] Customer {cid} already visited - REJECT")
            return False
        
        # ✅ Cannot visit customer at current position
        customer_node = cid + 1
        if self.truck_node == customer_node:
            if self.debug:
                print(f"    [TRUCK] Already at customer {cid} - REJECT")
            return False
        
        # Capacity check
        new_load = self.current_load + float(self.demands[cid])
        if new_load - 1e-9 > self.capacity:
            if self.debug:
                print(f"    [TRUCK] Customer {cid} exceeds capacity ({new_load:.1f} > {self.capacity:.1f}) - REJECT")
            return False
        
        # External calculator check
        if self.calculator is not None and hasattr(self.calculator, "feasible_truck_add"):
            feasible = bool(self.calculator.feasible_truck_add(self.truck_routes, cid + 1))
            if not feasible and self.debug:
                print(f"    [TRUCK] Customer {cid} rejected by calculator")
            return feasible
        
        return True

    def _feasible_drone_add_for(self, d: int, cid: int) -> bool:
        """Check if drone d can add customer cid (0-indexed)"""
        # ✅ CRITICAL: Check visited status FIRST
        if self.visited[cid]:
            if self.debug:
                print(f"    [DRONE-{d}] Customer {cid} already visited - REJECT")
            return False
        
        # Truck-only customers cannot be served by drone
        if self.truck_only[cid]:
            if self.debug:
                print(f"    [DRONE-{d}] Customer {cid} is truck-only - REJECT")
            return False
        
        # ✅ Cannot visit customer at current position
        customer_node = cid + 1
        if self.drone_node[d] == customer_node:
            if self.debug:
                print(f"    [DRONE-{d}] Already at customer {cid} - REJECT")
            return False

        # Energy check: can reach customer from current position
        cur = int(self.drone_node[d])
        need = self._dist(cur, customer_node)
        if self.drone_energy[d] + 1e-9 < need:
            if self.debug:
                print(f"    [DRONE-{d}] Customer {cid} unreachable (need {need:.1f}, have {self.drone_energy[d]:.1f}) - REJECT")
            return False

        # External calculator check
        if self.calculator is not None and hasattr(self.calculator, "feasible_drone_add"):
            feasible = bool(self.calculator.feasible_drone_add(self.drone_routes[d], cid + 1))
            if not feasible and self.debug:
                print(f"    [DRONE-{d}] Customer {cid} rejected by calculator")
            return feasible
        
        return True

    # ---------------------- Masking ----------------------
    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.phase == 'drone':
            d = self.current_drone
            
            # Action 0: Return to depot (only valid if not at depot)
            allow_zero = (self.drone_node[d] != 0)

            # Actions 1..N: Visit customers
            any_feasible = False
            for i in range(self.N):
                if self._feasible_drone_add_for(d, i):
                    mask[i + 1] = True
                    any_feasible = True

            # Safety: If at depot and no customers feasible, allow action 0 to switch drone
            if not any_feasible and self.drone_node[d] == 0:
                allow_zero = True

            mask[0] = allow_zero
            
            # Guarantee at least one valid action
            if not mask.any():
                mask[0] = True
            
            return mask

        else:  # truck phase
            # Action 0: Return to depot (only valid if not at depot)
            allow_zero = (self.truck_node != 0)

            # Actions 1..N: Visit customers
            any_feasible = False
            for i in range(self.N):
                if self._feasible_truck_add(i):
                    mask[i + 1] = True
                    any_feasible = True

            # Safety: If at depot and no customers feasible, allow action 0 to end
            if not any_feasible and self.truck_node == 0:
                allow_zero = True

            mask[0] = allow_zero
            
            # Guarantee at least one valid action
            if not mask.any():
                mask[0] = True
            
            return mask

    def get_action_mask(self) -> np.ndarray:
        m = self._get_action_mask()
        assert m.dtype == bool and m.shape == (self.action_space.n,)
        return m

    # ---------------------- Phase utility ----------------------
    def _any_drone_has_feasible(self) -> bool:
        """Check if ANY drone can serve ANY remaining customer"""
        for d in range(self.D):
            for i in range(self.N):
                if self._feasible_drone_add_for(d, i):
                    return True
        return False

    def _rotate_to_next_drone(self):
        self.current_drone = (self.current_drone + 1) % self.D

    # ---------------------- Step ----------------------
    def step(self, action: int):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        if self.debug:
            print(f"\n[STEP {self.step_count}] Phase={self.phase}, Action={action}")

        # ✅ VALIDATE ACTION BEFORE EXECUTION
        action_mask = self.get_action_mask()
        if not action_mask[action]:
            # Invalid action - apply heavy penalty and return
            reward = -10000.0
            if self.debug:
                print(f"  ❌ INVALID ACTION {action} (not in mask)")
                print(f"  Valid actions: {np.where(action_mask)[0]}")
            
            obs = self._get_observation()
            info = {
                "action_mask": action_mask,
                "phase": self.phase,
                "time": self.current_time,
                "truck_routes": self.truck_routes,
                "drone_routes": self.drone_routes,
                "current_drone": self.current_drone if self.phase == 'drone' else None,
                "customers_left": self.customers_left,
                "customers_visited": self.N - self.customers_left,
                "completion_rate": (self.N - self.customers_left) / self.N,
                "step_count": self.step_count,
                "error": "invalid_action"
            }
            return obs, float(reward), True, False, info

        # ==================== DRONE PHASE ====================
        if self.phase == 'drone':
            d = self.current_drone

            if action == 0:
                # Return to depot and recharge
                if self.drone_node[d] != 0:
                    cost = self._dist(int(self.drone_node[d]), 0)
                    self.drone_energy[d] = max(0.0, self.drone_energy[d] - cost)
                    reward -= cost
                    self.current_time += cost
                    self.drone_node[d] = 0
                    if self.debug:
                        print(f"  Drone {d} returned to depot (cost={cost:.1f})")
                
                # Recharge at depot
                self.drone_energy[d] = self.drone_battery
                
                # Rotate to next drone
                self._rotate_to_next_drone()
                if self.debug:
                    print(f"  Switched to drone {self.current_drone}")

                # Check if any drone can still serve customers
                if not self._any_drone_has_feasible():
                    self.phase = 'truck'
                    if self.debug:
                        print(f"  🚚 PHASE SWITCH: Drone → Truck ({self.customers_left} customers remaining)")
                    
                    # Reset truck to depot with empty load
                    if self.truck_node != 0:
                        cost = self._dist(self.truck_node, 0)
                        reward -= cost
                        self.current_time += cost
                        self.truck_node = 0
                    self.current_load = 0.0

            else:
                # Visit customer
                cid = action - 1
                
                # ✅ DOUBLE-CHECK visited status
                if self.visited[cid]:
                    reward = -10000.0
                    if self.debug:
                        print(f"  ❌ ERROR: Customer {cid} already visited!")
                    terminated = True
                else:
                    target_node = cid + 1
                    move_cost = self._dist(int(self.drone_node[d]), target_node)

                    # Execute move
                    self.drone_energy[d] -= move_cost
                    reward -= move_cost
                    self.current_time += move_cost
                    self.drone_node[d] = target_node

                    # ✅ Mark as visited IMMEDIATELY
                    self.visited[cid] = True
                    self.customers_left -= 1
                    self.drone_routes[d].append(cid)
                    
                    if self.debug:
                        print(f"  ✅ Drone {d} visited customer {cid} (cost={move_cost:.1f}, left={self.customers_left})")

                    # Check completion
                    if self.customers_left == 0:
                        terminated = True
                        if self.debug:
                            print(f"  🎉 ALL CUSTOMERS SERVED!")

                # Check phase switch
                if not terminated and not self._any_drone_has_feasible():
                    self.phase = 'truck'
                    if self.debug:
                        print(f"  🚚 PHASE SWITCH: Drone → Truck ({self.customers_left} customers remaining)")
                    
                    if self.truck_node != 0:
                        cost = self._dist(self.truck_node, 0)
                        reward -= cost
                        self.current_time += cost
                        self.truck_node = 0
                    self.current_load = 0.0

        # ==================== TRUCK PHASE ====================
        else:
            if action == 0:
                # Return to depot and unload
                if self.truck_node != 0:
                    cost = self._dist(self.truck_node, 0)
                    reward -= cost
                    self.current_time += cost
                    self.truck_node = 0
                    if self.debug:
                        print(f"  Truck returned to depot (cost={cost:.1f})")
                
                # Reset load
                self.current_load = 0.0

            else:
                # Visit customer
                cid = action - 1
                
                # ✅ DOUBLE-CHECK visited status
                if self.visited[cid]:
                    reward = -10000.0
                    if self.debug:
                        print(f"  ❌ ERROR: Customer {cid} already visited!")
                    terminated = True
                else:
                    target_node = cid + 1
                    cost = self._dist(self.truck_node, target_node)

                    # Execute move
                    reward -= cost
                    self.current_time += cost
                    self.truck_node = target_node

                    # ✅ Mark as visited IMMEDIATELY
                    self.truck_routes.append(cid)
                    self.visited[cid] = True
                    self.customers_left -= 1
                    self.current_load += float(self.demands[cid])
                    
                    if self.debug:
                        print(f"  ✅ Truck visited customer {cid} (cost={cost:.1f}, load={self.current_load:.1f}, left={self.customers_left})")

                    # Check completion
                    if self.customers_left == 0:
                        # Return to depot to close tour
                        if self.truck_node != 0:
                            c2 = self._dist(self.truck_node, 0)
                            reward -= c2
                            self.current_time += c2
                            self.truck_node = 0
                            if self.debug:
                                print(f"  Final return to depot (cost={c2:.1f})")
                        terminated = True
                        if self.debug:
                            print(f"  🎉 ALL CUSTOMERS SERVED!")

        # ✅ CHECK DEADLOCK OR TIMEOUT
        if not terminated:
            action_mask = self.get_action_mask()
            
            # Deadlock: no feasible actions
            if not action_mask.any():
                unvisited_penalty = self.customers_left * 1000.0
                reward -= unvisited_penalty
                terminated = True
                if self.debug:
                    print(f"  ⚠️ DEADLOCK: {self.customers_left} customers unvisited. Penalty: {unvisited_penalty:.1f}")
            
            # Timeout
            elif self.step_count >= self.max_steps:
                unvisited_penalty = self.customers_left * 500.0
                reward -= unvisited_penalty
                truncated = True
                if self.debug:
                    print(f"  ⏱️ TIMEOUT: {self.customers_left} customers unvisited. Penalty: {unvisited_penalty:.1f}")

        obs = self._get_observation()
        info = {
            "action_mask": self.get_action_mask(),
            "phase": self.phase,
            "time": self.current_time,
            "truck_routes": self.truck_routes,
            "drone_routes": self.drone_routes,
            "current_drone": self.current_drone if self.phase == 'drone' else None,
            "customers_left": self.customers_left,
            "customers_visited": self.N - self.customers_left,
            "completion_rate": (self.N - self.customers_left) / self.N,
            "step_count": self.step_count,
        }
        
        return obs, float(reward), terminated, truncated, info

    # ---------------------- Render ----------------------
    def render(self):
        if self.phase == 'drone':
            print(f"[t={self.current_time:.1f}] DRONE-{self.current_drone} @ node={self.drone_node[self.current_drone]} "
                  f"E={self.drone_energy[self.current_drone]:.1f}/{self.drone_battery:.1f} "
                  f"left={self.customers_left}/{self.N} step={self.step_count}")
        else:
            print(f"[t={self.current_time:.1f}] TRUCK @ node={self.truck_node} "
                  f"load={self.current_load:.1f}/{self.capacity:.1f} "
                  f"left={self.customers_left}/{self.N} step={self.step_count}")