# wadrl/envs/pvrpd_env.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
from environment.config import *


@dataclass
class TransitionData:
    obs: Dict[str, torch.Tensor]
    reward: torch.Tensor
    done: torch.Tensor
    info: Dict


class ParallelVRPDEnvironment:
    def __init__(
        self,
        cfg: Config,
        *,
        maintain_trip_cycles: bool = True,
        parallel_mode: bool = False,
        enable_idle_action: bool = False,
        max_drone_trips: Optional[int] = None,
        # reward normalization toggles (you can keep these False first):
        normalize_targets: bool = False,
        # weight handling:
        weight_strategy: str = "sample",
        predefined_weights: Tuple[float, float] = (0.5, 0.5),
        dirichlet_params: Tuple[float, float] = (1.0, 1.0),
        device: Optional[torch.device] = None,
        # max episode length
        max_episode_steps: int | None = 512, 
        timeout_cost: float = -1e7,
        display_actions: bool = False,
        penalty_unserved: float = 1e6
    ):
        """
        Args:
          cfg: Config from your parser.
          maintain_trip_cycles: if True, when drone returns to depot, E_cur resets and trip_idx += 1
          parallel_mode: if True, step expects per-vehicle actions in one call
          enable_idle_action: if True, add an idle token to masks (last column)
          max_drone_trips: optional hard cap on number of trips each drone may start
          normalize_targets: if True, maintains running min/max for F1,F2 to normalize
          weight_strategy: "sample" samples (w1,w2) ~ Dirichlet(alpha1,alpha2) at reset; "fixed" uses predefined_weights
          predefined_weights: weights used in inference if weight_strategy="fixed"
          dirichlet_params: alpha parameters for sampling w1,w2 during training
        """

        self.display_actions = display_actions
        self.penalty_unserved = penalty_unserved

        self.max_episode_steps = max_episode_steps
        self.timeout_cost = float(timeout_cost)
        self.current_step = 0

        self.cfg = cfg
        self.maintain_trip_cycles = maintain_trip_cycles
        self.parallel_mode = parallel_mode
        self.enable_idle_action = enable_idle_action
        self.max_drone_trips = max_drone_trips
        self.normalize_targets = normalize_targets

        # weights
        self.weight_strategy = weight_strategy
        self.predefined_weights = tuple(float(x) for x in predefined_weights)
        self.dirichlet_params = np.asarray(dirichlet_params, dtype=np.float64)

        self.num_customers = cfg.customers.N
        self.num_trucks = cfg.trucks.num_trucks
        self.num_drones = cfg.drones.D
        self.device = torch.device(cfg.device) if device is None else device

        # geometry (index 0 is depot)
        self.location_coords = [(0.0, 0.0)] + [c.coord() for c in cfg.customers.items]
        self.customer_demands = [0.0] + [c.demand for c in cfg.customers.items]
        self.truck_only_flag = np.array([0] + [int(c.only_truck) for c in cfg.customers.items], dtype=np.int64)
        self.truck_service_times = np.array([0.0] + [c.truck_service_time_s for c in cfg.customers.items], dtype=np.float32)
        self.drone_service_times  = np.array([0.0] + [c.drone_service_time_s  for c in cfg.customers.items], dtype=np.float32)

        # per-trip time cap (optional) from your meta: cfg.drones.max_trip_time_s (may be None)
        self.max_trip_duration = float(cfg.drones.max_trip_time_s) if cfg.drones.max_trip_time_s is not None else None


        # ---------- feature normalization (obs only, not physics) ----------

        # coords: center at depot, scale by max radius from depot
        coords_array = np.array(self.location_coords, dtype=np.float32)  # (V,2)
        depot_position = coords_array[0]
        relative_coords = coords_array - depot_position
        max_distance = float(np.linalg.norm(relative_coords, axis=1).max())
        if max_distance < 1e-6:
            max_distance = 1.0
        self._coordinate_offset = torch.tensor(depot_position, dtype=torch.float32, device=self.device)
        self._coordinate_normalization = torch.tensor(1.0 / max_distance, dtype=torch.float32, device=self.device)

        # demands: divide by max demand
        demands_array = np.asarray(self.customer_demands, dtype=np.float32)
        peak_demand = float(demands_array.max())
        if peak_demand <= 0:
            peak_demand = 1.0
        self._demand_normalization = torch.tensor(1.0 / peak_demand, dtype=torch.float32, device=self.device)

        # service times: divide by global max (truck/drone)
        service_array = np.stack([self.truck_service_times, self.drone_service_times], axis=-1)  # (V,2)
        peak_service = float(np.max(service_array))
        if peak_service <= 0:
            peak_service = 1.0
        self._service_normalization = torch.tensor(1.0 / peak_service, dtype=torch.float32, device=self.device)

        # drones: speeds, capacities, energy
        takeoff_velocity = [d.takeoff_speed for d in cfg.drones.drones]
        cruise_velocity  = [d.cruise_speed  for d in cfg.drones.drones]
        landing_velocity = [d.landing_speed for d in cfg.drones.drones]
        max_velocity = float(max(takeoff_velocity + cruise_velocity + landing_velocity))
        if max_velocity <= 0:
            max_velocity = 1.0
        self._drone_speed_normalization = torch.tensor(1.0 / max_velocity, dtype=torch.float32, device=self.device)

        max_capacity = float(max(d.capacity for d in cfg.drones.drones))
        if max_capacity <= 0:
            max_capacity = 1.0
        self._drone_capacity_normalization = torch.tensor(1.0 / max_capacity, dtype=torch.float32, device=self.device)

        max_energy = float(max(d.battery_power for d in cfg.drones.drones))
        if max_energy <= 0:
            max_energy = 1.0
        self._drone_energy_normalization = torch.tensor(1.0 / max_energy, dtype=torch.float32, device=self.device)

        # trips and intra-trip time (for features, not constraints)
        if self.max_drone_trips is not None:
            self._drone_trip_normalization = torch.tensor(1.0 / float(self.max_drone_trips),
                                               dtype=torch.float32, device=self.device)
        else:
            # heuristic: treat "no limit" as ~10 trips scale
            self._drone_trip_normalization = torch.tensor(0.01, dtype=torch.float32, device=self.device)

        if self.max_trip_duration is not None:
            self._trip_duration_normalization = torch.tensor(1.0 / self.max_trip_duration,
                                                 dtype=torch.float32, device=self.device)
        else:
            # fall back to a coarse scale; you can tune this
            self._trip_duration_normalization = torch.tensor(1e-3, dtype=torch.float32, device=self.device)

        # ---------- end normalization block ----------

        # precompute pairwise distance
        self.distance_matrix = self._calculate_pairwise_distances(self.location_coords)  # (V,V), float32 torch
        # drone time/energy matrices from catalog (D,V,V)
        drone_time_matrix = precompute_drone_times(cfg.drones, self.location_coords)
        self.drone_time_matrix = torch.tensor(drone_time_matrix, dtype=torch.float32, device=self.device)

        # running normalization for F1/F2 if enabled
        self._objective_min = torch.tensor([float('inf'), float('inf')], dtype=torch.float32, device=self.device)
        self._objective_max = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)

        # idle token index (last column if enabled)
        self.idle_token_index = self.num_customers + 1  # after nodes 0..N

        self._previous_objective1 = 0.0
        self._previous_objective2_normalized = 0.0

        self._initialize_state_variables()
        self._initialize_objective_weights()

    # -------------------- public helpers --------------------

    def configure_fixed_weights(self, weight1: float, weight2: float):
        """Use for inference: sets weight_strategy='fixed' and stores (w1,w2) normalized."""
        total = float(weight1) + float(weight2)
        if total <= 0:
            raise ValueError("w1+w2 must be positive.")
        self.weight_strategy = "fixed"
        self.predefined_weights = (float(weight1) / total, float(weight2) / total)

    def generate_random_weights(self):
        """Call to sample new (w1,w2) from Dirichlet (for training curricula)."""
        weight1 = float(np.random.choice(np.linspace(0.0, 1.0, 11)))
        weight2 = 1.0 - weight1
        self.objective_weights = torch.tensor([weight1, weight2], dtype=torch.float32, device=self.device)
        return self.objective_weights


    # -------------------- core API --------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.current_step = 0
        if self.max_episode_steps is None:
            # heuristic: enough to visit all customers + returns, but bounded
            self.max_episode_steps = 5 * (self.num_customers + self.num_trucks + self.num_drones)

        self.truck_cargo: List[List[int]] = [[] for _ in range(self.num_trucks)]
        self.drone_cargo: List[List[int]] = [[] for _ in range(self.num_drones)]        
        self.drone_payload_mass = torch.zeros(self.num_drones, dtype=torch.float32, device=self.device)

        self.customer_return_time = torch.full((self.num_customers+1,), float('nan'), device=self.device)
        self._initialize_state_variables()

        obj1_initial, obj2_initial = self._evaluate_objectives(include_unserved_penalty=False)
        self._previous_objective1 = float(obj1_initial)
        self._previous_objective2_normalized = float(obj2_initial / max(1, self.num_customers))

        self._initialize_objective_weights()
        return self._construct_observation()

    def step(self, action) -> TransitionData:
        self.current_step += 1
        if self.current_step > self.max_episode_steps and not self._is_episode_complete():
            print("Max episode len exceed!")
            step_reward = self.timeout_cost
            observation = self._construct_observation()
            episode_info = self._generate_info(True)
            return TransitionData(
                obs=observation,
                reward=torch.tensor([step_reward], dtype=torch.float32, device=self.device),
                done=torch.tensor([True], dtype=torch.bool, device=self.device),
                info=episode_info
            )
        if self.parallel_mode:
            result = self._execute_parallel_step(action)
        else:
            result = self._execute_serialized_step(action)
        return result

    # -------------------- internals: state --------------------

    def _initialize_state_variables(self):
        num_locations = self.num_customers + 1

        # served flags
        self.service_status = torch.zeros(num_locations, dtype=torch.bool, device=self.device)
        self.service_status[0] = True

        # vehicle clocks
        self.truck_clock = torch.zeros(self.num_trucks, dtype=torch.float32, device=self.device)
        self.drone_clock  = torch.zeros(self.num_drones, dtype=torch.float32, device=self.device)

        # positions (start at depot)
        self.truck_position = torch.zeros(self.num_trucks, dtype=torch.long, device=self.device)
        self.drone_position  = torch.zeros(self.num_drones, dtype=torch.long, device=self.device)

        # energy and trip bookkeeping (per drone)
        self.drone_battery_level = torch.tensor([d.battery_power for d in self.cfg.drones.drones],
                                 dtype=torch.float32, device=self.device)
        self.drone_current_trip = torch.zeros(self.num_drones, dtype=torch.long, device=self.device)  # current trip number
        self.drone_trip_duration = torch.zeros(self.num_drones, dtype=torch.float32, device=self.device)  # seconds spent within current trip

        # finished flags (optional if you want to close each tour)
        self.truck_finished = torch.zeros(self.num_trucks, dtype=torch.bool, device=self.device)
        self.drone_finished  = torch.zeros(self.num_drones, dtype=torch.bool, device=self.device)

        self.drone_cargo: List[List[int]] = [[] for _ in range(self.num_drones)]        
        self.drone_payload_mass = torch.zeros(self.num_drones, dtype=torch.float32, device=self.device)
        

        # accounting for objectives
        self.customer_service_time = torch.full((num_locations,), float('nan'), dtype=torch.float32, device=self.device)  # service completion
        self.customer_return_time = torch.full((num_locations,), float('nan'), dtype=torch.float32, device=self.device)  # return-to-depot times (optional exact)

    def _initialize_objective_weights(self):
        if self.weight_strategy == "fixed":
            w1, w2 = self.predefined_weights
            total = max(1e-9, w1 + w2)
            self.objective_weights = torch.tensor([w1 / total, w2 / total], dtype=torch.float32, device=self.device)
        else:
            # sample (training)
            weight1 = float(np.random.choice(np.linspace(0.0, 1.0, 5)))
            weight2 = 1.0 - weight1
            self.objective_weights = torch.tensor([weight1, weight2], dtype=torch.float32, device=self.device)


    def _calculate_drone_flight_time(self, drone_idx: int, origin: int, destination: int) -> float:
        return float(self.drone_time_matrix[drone_idx, origin, destination].item())

    def _calculate_drone_energy_consumption(self, drone_idx: int, origin: int, destination: int, cargo_weight: float) -> float:
        flight_distance = float(self.distance_matrix[origin, destination].item())
        return float(self.cfg.drones.drones[drone_idx].energy_j(flight_distance, cargo_weight))

    # -------------------- internals: utilities --------------------

    def _calculate_pairwise_distances(self, coordinates):
        num_nodes = len(coordinates)
        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist_matrix[i, j] = euclid(coordinates[i], coordinates[j])
        return torch.tensor(dist_matrix, dtype=torch.float32, device=self.device)

    def _calculate_truck_travel_duration(self, origin: int, destination: int, departure_time: float) -> float:
        return self.cfg.trucks.calc_truck_travel_time(start_time_s=departure_time, distance_m=float(self.distance_matrix[origin, destination].item()))

    # -------------------- observation & masks --------------------

    def _construct_observation(self) -> Dict[str, torch.Tensor]:
        num_locations = self.num_customers + 1

        # -------- Graph features: (x,y, only_truck, demand, svc_trk, svc_dr, served) --------
        node_features = torch.empty((num_locations, 7), dtype=torch.float32, device=self.device)

        # coords: center at depot + scale by max radius
        coordinates = torch.tensor(self.location_coords, dtype=torch.float32, device=self.device)  # (V,2)
        coordinates = (coordinates - self._coordinate_offset) * self._coordinate_normalization
        node_features[:, 0:2] = coordinates

        node_features[:, 2] = torch.tensor(self.truck_only_flag, dtype=torch.float32, device=self.device)

        demands = torch.tensor(self.customer_demands, dtype=torch.float32, device=self.device)
        node_features[:, 3] = demands * self._demand_normalization

        truck_svc = torch.tensor(self.truck_service_times, dtype=torch.float32, device=self.device)
        drone_svc  = torch.tensor(self.drone_service_times,  dtype=torch.float32, device=self.device)
        node_features[:, 4] = truck_svc * self._service_normalization
        node_features[:, 5] = drone_svc  * self._service_normalization

        node_features[:, 6] = self.service_status.float()
        graph_context = node_features.unsqueeze(0)  # (1,V,7)

        # -------- Trucks context: (x,y, available, slot_sin, slot_cos) --------
        num_trucks = self.num_trucks
        truck_features = torch.zeros((num_trucks, 5), dtype=torch.float32, device=self.device)

        truck_coords = torch.tensor([self.location_coords[i] for i in self.truck_position.tolist()],
                                    dtype=torch.float32, device=self.device)  # (K,2)
        truck_coords = (truck_coords - self._coordinate_offset) * self._coordinate_normalization
        truck_features[:, 0:2] = truck_coords

        truck_features[:, 2] = (~self.truck_finished).float()

        num_slots = self.cfg.trucks.L
        time_slot = torch.floor(self.truck_clock / self.cfg.trucks.slot_len_seconds) % num_slots
        angle = 2.0 * math.pi * time_slot / num_slots
        truck_features[:, 3] = torch.sin(angle)
        truck_features[:, 4] = torch.cos(angle)
        trucks_context = truck_features.unsqueeze(0)  # (1,K,5)

        # -------- Drones context --------
        # (x,y, v_to, v_cru, v_ld, cap, E_max, E_cur, trip_idx, avail, trip_elapsed, trips_left)
        num_drones = self.num_drones
        drone_features = torch.zeros((num_drones, 12), dtype=torch.float32, device=self.device)

        drone_coords = torch.tensor([self.location_coords[i] for i in self.drone_position.tolist()],
                                    dtype=torch.float32, device=self.device)  # (D,2)
        drone_coords = (drone_coords - self._coordinate_offset) * self._coordinate_normalization
        drone_features[:, 0:2] = drone_coords

        remaining_trips = torch.full((num_drones,), float('inf'), dtype=torch.float32, device=self.device)
        if self.max_drone_trips is not None:
            remaining_trips = (self.max_drone_trips - self.drone_current_trip).clamp(min=0).float()

        for drone_id, drone_spec in enumerate(self.cfg.drones.drones):
            drone_features[drone_id, 2] = drone_spec.takeoff_speed * self._drone_speed_normalization
            drone_features[drone_id, 3] = drone_spec.cruise_speed  * self._drone_speed_normalization
            drone_features[drone_id, 4] = drone_spec.landing_speed * self._drone_speed_normalization
            drone_features[drone_id, 5] = drone_spec.capacity      * self._drone_capacity_normalization
            drone_features[drone_id, 6] = drone_spec.battery_power * self._drone_energy_normalization

        drone_features[:, 7]  = self.drone_battery_level * self._drone_energy_normalization
        drone_features[:, 8]  = self.drone_current_trip.float() * self._drone_trip_normalization
        drone_features[:, 9]  = (~self.drone_finished).float()
        drone_features[:, 10] = self.drone_trip_duration * self._trip_duration_normalization
        drone_features[:, 11] = remaining_trips * self._drone_trip_normalization
        drones_context = drone_features.unsqueeze(0)  # (1,D,12)

        # -------- masks --------
        truck_action_mask = self._generate_truck_mask()    # (1,K,V [+1 if idle])
        drone_action_mask  = self._generate_drone_mask()    # (1,D,V [+1 if idle])

        # weights (1,2) to be embedded by the policy (WADRL)
        weight_vector = self.objective_weights.view(1, 2)

        return dict(
            graph_ctx=graph_context,
            trucks_ctx=trucks_context,
            drones_ctx=drones_context,
            mask_trk=truck_action_mask,
            mask_dr=drone_action_mask,
            weights=weight_vector,
        )


    def _generate_truck_mask(self):
        num_locations = self.num_customers + 1
        num_trucks = self.num_trucks

        extra_columns = 1 if self.enable_idle_action else 0
        action_mask = torch.zeros((1, num_trucks, num_locations + extra_columns), dtype=torch.bool, device=self.device)

        # feasibility: can visit any unserved customer, plus depot (always)
        # (You can add more truck-specific constraints here if needed)
        # customers 1..N:
        unserved_customers = (~self.service_status[1:]).unsqueeze(0).unsqueeze(0).expand(1, num_trucks, self.num_customers)  # (1,K,N)
        action_mask[:, :, 1:1+self.num_customers] = unserved_customers

        # depot allowed only after all customers served
        depot_accessible = bool(self.service_status[1:].all().item())
        action_mask[:, :, 0] = depot_accessible


        if self.enable_idle_action:
            action_mask[:, :, -1] = True  # idle

        # if a truck is marked done, only idle (if enabled) or depot (to keep it harmless)
        finished_trucks = torch.where(self.truck_finished)[0].tolist()
        for truck_id in finished_trucks:
            action_mask[:, truck_id, :] = False
            action_mask[:, truck_id, 0] = True
            if self.enable_idle_action:
                action_mask[:, truck_id, -1] = True

        for truck_id in range(num_trucks):
            current_pos = int(self.truck_position[truck_id].item())
            action_mask[:, truck_id, current_pos] = False          # forbid self-loop everywhere

        return action_mask

    def _generate_drone_mask(self):
        num_locations = self.num_customers + 1
        num_drones = self.num_drones
        extra_columns = 1 if self.enable_idle_action else 0
        action_mask = torch.zeros((1, num_drones, num_locations + extra_columns), dtype=torch.bool, device=self.device)

        # base feasibility: not truck-only and not served
        drone_accessible = torch.tensor(1 - self.truck_only_flag, dtype=torch.bool, device=self.device)  # 1 for drone-allowed
        base_feasibility = (drone_accessible & (~self.service_status)).unsqueeze(0).unsqueeze(0).expand(1, num_drones, num_locations)  # (1,D,V)
        action_mask[:, :, :] = False
        action_mask[:, :, :] |= base_feasibility

        # depot always allowed
        action_mask[:, :, 0] = True

        # energy + return reachability + per-trip time cap + max_trips constraint
        for drone_id in range(num_drones):
            if self.drone_finished[drone_id]:
                action_mask[:, drone_id, :] = False
                action_mask[:, drone_id, 0] = True
                if self.enable_idle_action:
                    action_mask[:, drone_id, -1] = True
                continue

            current_location = int(self.drone_position[drone_id].item())
            current_battery = float(self.drone_battery_level[drone_id].item())
            elapsed_duration = float(self.drone_trip_duration[drone_id].item())

            # If max_drone_trips reached and drone is at depot, forbid leaving depot:
            if (self.max_drone_trips is not None) and (self.drone_current_trip[drone_id] >= self.max_drone_trips) and (current_location == 0):
                action_mask[:, drone_id, :] = False
                action_mask[:, drone_id, 0] = True  # can sit at depot only
                if self.enable_idle_action:
                    action_mask[:, drone_id, -1] = True
                continue

            # Check each candidate j:
            for target_location in range(num_locations):
                current_payload = float(self.drone_payload_mass[drone_id].item())
                drone_capacity = self.cfg.drones.drones[drone_id].capacity
                current_battery = float(self.drone_battery_level[drone_id].item())

                if target_location == current_location:
                    action_mask[:, drone_id, target_location] = False  # forbid self-loop
                    continue

                if target_location == 0:
                    # Depot return feasibility
                    energy_to_depot = self._calculate_drone_energy_consumption(drone_id, current_location, 0, current_payload)
                    if current_battery < energy_to_depot: 
                        action_mask[:, drone_id, 0] = False
                        continue
                    if self.max_trip_duration is not None:
                        time_to_depot = self._calculate_drone_flight_time(drone_id, current_location, 0)
                        if self.drone_trip_duration[drone_id] + time_to_depot > self.max_trip_duration:
                            action_mask[:, drone_id, 0] = False
                    continue

                # customer j
                if self.service_status[target_location] or self.truck_only_flag[target_location]:
                    action_mask[:, drone_id, target_location] = False
                    continue
                # capacity after pickup
                if current_payload + self.customer_demands[target_location] > drone_capacity:
                    action_mask[:, drone_id, target_location] = False
                    continue

                # energy forward + guaranteed return (after serving j)
                energy_to_customer = self._calculate_drone_energy_consumption(drone_id, current_location, target_location, current_payload)
                energy_from_customer = self._calculate_drone_energy_consumption(drone_id, target_location, 0, current_payload + self.customer_demands[target_location])
                if current_battery < energy_to_customer + energy_from_customer + 1e-9:
                    action_mask[:, drone_id, target_location] = False
                    continue

                # per-trip time cap (forward+svc+return)
                if self.max_trip_duration is not None:
                    time_to_customer = self._calculate_drone_flight_time(drone_id, current_location, target_location)
                    time_from_customer = self._calculate_drone_flight_time(drone_id, target_location, 0)
                    service_duration  = float(self.drone_service_times[target_location]) if not isinstance(self.drone_service_times, torch.Tensor) else float(self.drone_service_times[target_location].item())
                    if self.drone_trip_duration[drone_id] + time_to_customer + service_duration + time_from_customer > self.max_trip_duration:
                        action_mask[:, drone_id, target_location] = False
                        continue

                action_mask[:, drone_id, target_location] = True

            if self.enable_idle_action:
                action_mask[:, drone_id, -1] = True  # idle

        # served customers masked:
        action_mask[:, :, 1:1+self.num_customers] &= (~self.service_status[1:]).unsqueeze(0).unsqueeze(0)

        # truck-only customers masked:
        truck_restricted = torch.tensor(self.truck_only_flag, dtype=torch.bool, device=self.device)
        action_mask[:, :, :] &= torch.logical_not(truck_restricted).unsqueeze(0).unsqueeze(0) | torch.tensor(
            [[True] + [False]*self.num_customers], device=self.device
        ).unsqueeze(1)

        return action_mask

    # -------------------- stepping --------------------

    def _execute_truck_action(self, truck_id: int, destination: int):
        origin = int(self.truck_position[truck_id].item())
        if origin == destination and origin != 0:
            raise ValueError("Self loop tour! Start != End is must!")
        departure_time = float(self.truck_clock[truck_id].item())

        # if idle token used:
        if self.enable_idle_action and (destination == self.idle_token_index):
            # do nothing, could add small time tick if desired
            return

        travel_duration = self._calculate_truck_travel_duration(origin, destination, departure_time)
        self.truck_clock[truck_id] = departure_time + travel_duration + float(self.truck_service_times[destination])
        self.truck_position[truck_id] = destination

        if destination != 0:
            self.service_status[destination] = True
            self.customer_service_time[destination] = self.truck_clock[truck_id]
            self.truck_cargo[truck_id].append(destination)
        elif origin != 0:  # returning from a route end (or multi-tour)
            arrival_time = float(self.truck_clock[truck_id].item())
            for customer_id in self.truck_cargo[truck_id]:
                self.customer_return_time[customer_id] = arrival_time
            self.truck_cargo[truck_id].clear()
            self.truck_finished[truck_id] = True



    def _execute_drone_action(self, drone_id: int, destination: int):
        origin = int(self.drone_position[drone_id].item())
        if origin == destination:
            raise ValueError("Self loop tour! Start != End is must!")

        current_payload = float(self.drone_payload_mass[drone_id].item())

        if self.enable_idle_action and (destination == self.idle_token_index):
            return

        if destination == 0:
            # Return to depot with current payload
            flight_time = self._calculate_drone_flight_time(drone_id, origin, 0)
            energy_cost = self._calculate_drone_energy_consumption(drone_id, origin, 0, current_payload)
            # safety mirrors mask:
            if float(self.drone_battery_level[drone_id].item()) < energy_cost:
                raise ValueError("Energy violation on return")
            if self.max_trip_duration is not None and (self.drone_trip_duration[drone_id] + flight_time > self.max_trip_duration):
                raise ValueError("Trip time cap violation on return")

            self.drone_clock[drone_id] += flight_time
            self.drone_trip_duration[drone_id] += flight_time
            self.drone_battery_level[drone_id] -= energy_cost
            self.drone_position[drone_id] = 0

            # finalize waiting times for all onboard goods
            arrival_time = float(self.drone_clock[drone_id].item())
            for customer_id in self.drone_cargo[drone_id]:
                self.customer_return_time[customer_id] = arrival_time
            self.drone_cargo[drone_id].clear()
            self.drone_payload_mass[drone_id] = 0.0

            if self.maintain_trip_cycles:
                self.drone_current_trip[drone_id] += 1
                self.drone_trip_duration[drone_id] = 0.0
                self.drone_battery_level[drone_id] = self.cfg.drones.drones[drone_id].battery_power
            return

        # Move to customer j
        flight_time = self._calculate_drone_flight_time(drone_id, origin, destination)
        energy_cost = self._calculate_drone_energy_consumption(drone_id, origin, destination, current_payload)
        service_duration  = float(self.drone_service_times[destination]) if not isinstance(self.drone_service_times, torch.Tensor) else float(self.drone_service_times[destination].item())

        # safety mirrors mask (including guaranteed return)
        energy_return = self._calculate_drone_energy_consumption(drone_id, destination, 0, current_payload + self.customer_demands[destination])
        if float(self.drone_battery_level[drone_id].item()) < energy_cost + energy_return:
            raise ValueError("Energy violation forward+return")
        if current_payload + self.customer_demands[destination] > self.cfg.drones.drones[drone_id].capacity:
            raise ValueError("Capacity violation")
        if self.max_trip_duration is not None:
            return_time = self._calculate_drone_flight_time(drone_id, destination, 0)
            if self.drone_trip_duration[drone_id] + flight_time + service_duration + return_time > self.max_trip_duration:
                raise ValueError("Trip time cap violation")

        # apply
        self.drone_clock[drone_id] += flight_time + service_duration
        self.drone_trip_duration[drone_id] += flight_time + service_duration
        self.drone_battery_level[drone_id] -= energy_cost
        self.drone_position[drone_id] = destination

        # mark service & load
        self.service_status[destination] = True
        self.customer_service_time[destination] = self.drone_clock[drone_id]
        self.drone_cargo[drone_id].append(destination)
        self.drone_payload_mass[drone_id] += float(self.customer_demands[destination])


    def _execute_serialized_step(self, action) -> TransitionData:
        """
        action: tuple/list (veh_type: 0=truck,1=drone, instance_idx, next_node_id)
                next_node_id in [0..N] or idle index if enable_idle_action=True
        """
        vehicle_type, vehicle_index, target_node = map(int, action)
        if self.display_actions:
            print(f"Action: {vehicle_type} with index {vehicle_index} move to {target_node}")
        if vehicle_type == 0:
            self._execute_truck_action(vehicle_index, target_node)
        else:
            self._execute_drone_action(vehicle_index, target_node)

        episode_complete = self._is_episode_complete()
        if episode_complete and self.display_actions:
            print("Episode done!")
        step_reward = self._calculate_step_reward(episode_complete)
        observation = self._construct_observation()
        episode_info = self._generate_info(episode_complete)
        return TransitionData(
            obs=observation,
            reward=torch.tensor([step_reward], dtype=torch.float32, device=self.device),
            done=torch.tensor([episode_complete], dtype=torch.bool, device=self.device),
            info=episode_info
        )

    def _execute_parallel_step(self, action) -> TransitionData:
        """
        action: dict {'truck_nodes': LongTensor[K], 'drone_nodes': LongTensor[D]}
                each entry is next node id (0..N) or idle index (if enabled)
        """
        truck_targets = action['truck_nodes'].tolist()
        drone_targets = action['drone_nodes'].tolist()

        for truck_id, target in enumerate(truck_targets):
            self._execute_truck_action(truck_id, int(target))
        for drone_id, target in enumerate(drone_targets):
            self._execute_drone_action(drone_id, int(target))

        episode_complete = self._is_episode_complete()
        step_reward = self._calculate_step_reward(episode_complete)
        observation = self._construct_observation()
        episode_info = self._generate_info(episode_complete)
        return TransitionData(
            obs=observation,
            reward=torch.tensor([step_reward], dtype=torch.float32, device=self.device),
            done=torch.tensor([episode_complete], dtype=torch.bool, device=self.device),
            info=episode_info
        )

    # -------------------- termination, objectives, reward --------------------

    def _is_episode_complete(self) -> bool:
        # Here: episode ends when all customers are served.
        # (Optionally also require all vehicles to be back at depot.)
        return bool(self.service_status[1:].all().item())

    def _evaluate_objectives(self, include_unserved_penalty=True) -> Tuple[float, float]:
        # customer_return_time currently has NaNs for cargo still onboard; we fill those with
        # a conservative "return now" completion time per vehicle.
        return_times = self.customer_return_time.clone()

        # Trucks: if carrying, estimate immediate return completion for those samples
        for truck_id in range(self.num_trucks):
            if self.truck_cargo[truck_id]:
                current_pos = int(self.truck_position[truck_id].item())
                current_time = float(self.truck_clock[truck_id].item())
                return_duration = self._calculate_truck_travel_duration(current_pos, 0, current_time)
                estimated_return = current_time + return_duration
                for customer_id in self.truck_cargo[truck_id]:
                    return_times[customer_id] = estimated_return

        # Drones: if carrying, estimate immediate return time using _calculate_drone_flight_time
        for drone_id in range(self.num_drones):
            if self.drone_cargo[drone_id]:
                current_pos = int(self.drone_position[drone_id].item())
                estimated_return = float(self.drone_clock[drone_id].item()) + self._calculate_drone_flight_time(drone_id, current_pos, 0)
                for customer_id in self.drone_cargo[drone_id]:
                    return_times[customer_id] = estimated_return


        # Valid served customers (exclude depot=0)
        customers_served = self.service_status.clone()
        customers_served[0] = False

        customers_unserved = ~self.service_status.clone()
        customers_unserved[0] = False
        count_unserved = int(customers_unserved.sum().item())

        # ---------- F1: makespan over served customers ----------
        # valid entries are those that are served and not NaN in customer_return_time
        valid_objective1 = customers_served & (~torch.isnan(return_times))
        if valid_objective1.any():
            # replace invalid entries with a very negative number, then max
            very_small = torch.tensor(-1e30, device=self.device, dtype=return_times.dtype)
            safe_return_times = torch.where(valid_objective1, return_times, very_small)
            objective1 = float(safe_return_times.max().item())
        else:
            objective1 = 0.0

        # ---------- F2: total waiting time sum_i (customer_return_time[i] - customer_service_time[i]) ----------
        waiting_time = return_times - self.customer_service_time
        valid_objective2 = customers_served & (~torch.isnan(waiting_time))
        objective2 = float(torch.where(valid_objective2, waiting_time, torch.tensor(0.0, device=self.device)).sum().item())
        if count_unserved > 0 and include_unserved_penalty:
            objective2 += count_unserved * self.penalty_unserved
        return objective1 / 3600.0, objective2 / 3600.0


    def _calculate_step_reward(self, episode_complete: bool) -> float:
        # Compute current F1, F2 (exclude unserved penalty mid-episode).
        objective1, objective2 = self._evaluate_objectives(include_unserved_penalty=episode_complete)
        objective2_normalized = objective2 / max(1, self.num_customers)  # normalize F2 by N to stabilize scale
        
        # Shaped, telescoping reward on the DELTA of the weighted objective.
        weight1, weight2 = float(self.objective_weights[0].item()), float(self.objective_weights[1].item())
        current_value = max(weight1 * objective1, weight2 * objective2_normalized)
        previous_value = max(weight1 * self._previous_objective1, weight2 * self._previous_objective2_normalized)
        reward_delta = -(current_value - previous_value)

        # Update prev states
        self._previous_objective1 = objective1
        self._previous_objective2_normalized = objective2_normalized
        return reward_delta



    def _generate_info(self, episode_complete: bool) -> Dict:
        objective1, objective2 = self._evaluate_objectives()
        objective2_normalized = objective2 / max(1, self.num_customers)

        # Shaped, telescoping reward on the DELTA of the weighted objective.
        weight1, weight2 = float(self.objective_weights[0].item()), float(self.objective_weights[1].item())
        total_reward = max(weight1 * objective1, weight2 * objective2_normalized)
        
        return dict(
            reward=total_reward,
            F1=objective1, F2=objective2, F2n=objective2_normalized,
            w1=float(self.objective_weights[0].item()), w2=float(self.objective_weights[1].item()),
            served=int(self.service_status[1:].sum().item()),
        )