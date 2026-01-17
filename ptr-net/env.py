# import torch
# import numpy as np

# # from config import SystemConfig
# # from model import MOPVRP_Actor
# # from dataloader import get_rl_dataloader
# # from visualizer import visualize_mopvrp

# class MOPVRPEnvironment:
#     def __init__(self, config, dataloader, device='cpu'):
#         self.config = config
#         self.dataloader = dataloader
#         self.device = device
#         self.data_iter = iter(dataloader)
        
#         # State placeholders
#         self.static = None       # [B, 4, N]
#         self.dynamic_truck = None # [B, 2, K]
#         self.dynamic_drone = None # [B, 4, D]
#         self.mask_cust = None     # [B, N]
#         self.scale = None         # [B, 1]
#         self.weights = None       # [B, 2]

#     def reset(self):
#         batch_data = next(self.data_iter)
#         self.static, self.dynamic_truck, self.dynamic_drone, \
#         self.mask_cust, mask_veh, self.scale, self.weights = batch_data
        
#         self.batch_size = self.static.size(0)
#         self.num_nodes = self.static.size(2)
#         self.num_trucks = self.dynamic_truck.size(2)
#         self.num_drones = self.dynamic_drone.size(2)
        
#         self.routes = [[{'trucks': [[0] for _ in range(self.num_trucks)], 
#                          'drones': [[0] for _ in range(self.num_drones)]} 
#                         for _ in range(self.batch_size)]]
        
#         self.total_waiting_time = torch.zeros(self.batch_size, device=self.device)
        
#         return (self.static, self.dynamic_truck, self.dynamic_drone, self.mask_cust, self._update_vehicle_mask())

#     def step(self, selected_vehicle_idx, selected_node_idx):
#         is_drone = selected_vehicle_idx >= self.num_trucks
#         drone_idx = selected_vehicle_idx - self.num_trucks
#         truck_idx = selected_vehicle_idx
        
#         # 1. Update Physics
#         self._update_truck_state(truck_idx, selected_node_idx, ~is_drone)
#         self._update_drone_state(drone_idx, selected_node_idx, is_drone)
        
#         is_customer = (selected_node_idx != 0)
        
#         if is_customer.any():
#             current_times = torch.zeros(self.batch_size, device=self.device)
            
#             if (~is_drone).any():
#                 b_tr = torch.where(~is_drone)[0]
#                 tr_ids = truck_idx[b_tr]
#                 current_times[b_tr] = self.dynamic_truck[b_tr, 1, tr_ids]
                
#             # Drone times
#             if is_drone.any():
#                 b_dr = torch.where(is_drone)[0]
#                 dr_ids = drone_idx[b_dr]
#                 current_times[b_dr] = self.dynamic_drone[b_dr, 1, dr_ids]
            
#             self.total_waiting_time += current_times * is_customer.float()

#         # 2. Update Customer Mask
#         # Nếu chọn Node != 0 thì đánh dấu đã thăm
#         node_mask_update = torch.nn.functional.one_hot(selected_node_idx.long(), num_classes=self.num_nodes)
#         not_depot = (selected_node_idx != 0).float().unsqueeze(1)
#         node_mask_update = node_mask_update * not_depot
#         self.mask_cust = self.mask_cust * (1 - node_mask_update)
        
#         # 3. CHECK DONE CONDITION
#         unvisited_count = self.mask_cust[:, 1:].sum(dim=1)
        
#         # Check Truck Location
#         truck_locs = self.dynamic_truck[:, 0, :]
#         all_trucks_home = (truck_locs == 0).all(dim=1)
        
#         # Check Drone Location
#         drone_locs = self.dynamic_drone[:, 0, :]
#         all_drones_home = (drone_locs == 0).all(dim=1)
        
#         # Điều kiện dừng đầy đủ
#         done = (unvisited_count == 0) & all_trucks_home & all_drones_home
        
#         # Check Force Stop (Deadlock): Nếu còn khách hoặc còn xe ngoài đường, 
#         next_mask_veh = self._update_vehicle_mask()
#         no_valid_vehicle = (next_mask_veh.sum(dim=1) == 0)
        
#         # Nếu bị deadlock (chưa xong việc mà hết xe), ta vẫn trả về done=True để reset môi trường, nhưng sẽ phạt nặng ở reward.
#         done = done | no_valid_vehicle
        
#         # 4. Reward
#         step_reward = torch.full((self.batch_size,), -0.1, device=self.device)
        
#         if done.any():
#             final_rewards = self._calculate_terminal_reward()
#             # Nếu done do deadlock (vẫn còn khách hoặc xe chưa về): Phạt nặng
#             is_failure = (unvisited_count > 0) | (~all_trucks_home) | (~all_drones_home)
#             penalty = torch.where(is_failure, torch.tensor(-10.0, device=self.device), torch.tensor(0.0, device=self.device))
            
#             step_reward = torch.where(done, final_rewards + penalty, step_reward)
            
#         self._update_routes_history(selected_vehicle_idx, selected_node_idx)
        
#         next_state = (self.static, self.dynamic_truck, self.dynamic_drone, self.mask_cust, next_mask_veh)
#         return next_state, step_reward, done, {}

#     def get_valid_customer_mask(self, selected_vehicle_idx):
#         """
#         Logic Mask Node:
#         - Nếu HẾT khách: Bắt buộc chọn Node 0 (cho cả Truck và Drone).
#         """
#         valid_mask = self.mask_cust.clone()
        
#         unvisited_count = self.mask_cust[:, 1:].sum(dim=1)
#         is_all_served = (unvisited_count == 0)
        
#         # --- END LOGIC ---
#         if is_all_served.any():
#             served_indices = torch.where(is_all_served)[0]
#             valid_mask[served_indices, :] = 0
#             valid_mask[served_indices, 0] = 1
        
#         # --- NORMAL LOGIC (Khi còn khách) ---
#         batch_indices = torch.arange(self.batch_size, device=self.device)
#         is_drone = selected_vehicle_idx >= self.num_trucks
        
#         # Truck Logic (Start at Depot restriction)
#         if (~is_drone).any():
#             t_indices = torch.where(~is_drone)[0]
#             t_ids = selected_vehicle_idx[t_indices]
#             current_times = self.dynamic_truck[t_indices, 1, t_ids]
            
#             # Nếu đang ở Depot (Time=0) -> Không được chọn Node 0
#             at_start = (current_times == 0)
#             if at_start.any():
#                 start_indices = t_indices[at_start]
#                 valid_mask[start_indices, 0] = 0
        
#         if is_drone.any():
#             d_indices = torch.where(is_drone)[0]
#             d_ids = selected_vehicle_idx[d_indices] - self.num_trucks
            
#             # 1. Enable Node 0 (Luôn cho phép về Depot sạc/chờ)
#             valid_mask[d_indices, 0] = 1
            
#             # 2. Mask Truck-only customers
#             truck_only_flags = self.static[d_indices, 3, :]
#             valid_mask[d_indices] *= (1 - truck_only_flags)
            
#             # 3. CHECK ENERGY: (Curr -> Node) + (Node -> Depot)
#             p = self.config.drone_params
#             speed = self.config.drone_speed
#             t_const = self.config.t_takeoff + self.config.t_landing
            
#             # A. Tính chặng đi: Current -> Candidate Node
#             curr_nodes = self.dynamic_drone[d_indices, 0, d_ids].long()
#             coords = self.static[d_indices, :2, :] # (N_drone, 2, N_nodes)
            
#             curr_xy = self._gather_coords(coords, curr_nodes)
#             curr_xy_exp = curr_xy.unsqueeze(2) 
            
#             # Khoảng cách đi (N_drone, N_nodes)
#             dist_go = torch.norm(coords - curr_xy_exp, dim=1) * self.scale[d_indices, 0].unsqueeze(1)
            
#             # B. Tính chặng về: Candidate Node -> Depot (Node 0)
#             depot_xy = coords[:, :, 0].unsqueeze(2) # (N_drone, 2, 1)
#             dist_return = torch.norm(coords - depot_xy, dim=1) * self.scale[d_indices, 0].unsqueeze(1)
            
#             # C. Tính Năng Lượng
#             payloads = self.static[d_indices, 2, :] 
            
#             # Công suất (Power)
#             power = p['gama(w)'] + p['beta(w/kg)'] * payloads
            
#             # Năng lượng chặng đi
#             time_go = dist_go / speed + t_const
#             energy_go = power * time_go
            
#             # Năng lượng chặng về
#             time_return = dist_return / speed + t_const
#             energy_return = power * time_return
            
#             # Tổng năng lượng cần thiết
#             total_energy_req = energy_go + energy_return

#             # --- Xử lý ngoại lệ cho Node 0 (Depot) ---
#             # Nếu ĐANG Ở Depot và chọn Node 0 (Đứng chờ): Energy req gần như bằng 0 (hoặc rất nhỏ).
#             is_at_depot = (curr_nodes == 0).unsqueeze(1)
#             target_is_depot = torch.zeros_like(total_energy_req, dtype=torch.bool)
#             target_is_depot[:, 0] = True
            
#             # Nếu (Đang ở 0) VÀ (Chọn 0) -> Energy Req = 0
#             stay_mask = is_at_depot & target_is_depot
#             total_energy_req[stay_mask] = 0.0
            
#             # D. So sánh với Pin hiện tại
#             energy_req_norm = total_energy_req / self.config.drone_max_energy
#             curr_energy = self.dynamic_drone[d_indices, 2, d_ids].unsqueeze(1)
            
#             # Tạo mask
#             energy_mask = (curr_energy >= energy_req_norm).float()
#             valid_mask[d_indices] *= energy_mask

#             # 4. Check payloads

#             # Payload hiện tại
#             curr_load = self.dynamic_drone[d_indices, 3, d_ids].unsqueeze(1) # (N, 1)
#             # Demand của các khách hàng tiềm năng
#             demands = self.static[d_indices, 2, :] # (N, Nodes)
#             # Capacity Max
#             max_cap = self.config.drone_capacity_kg 
            
#             # Điều kiện: Load sau khi nhặt <= Max Cap
#             cap_mask = (curr_load + demands) <= max_cap
#             valid_mask[d_indices] *= cap_mask.float()

#         # Re-apply End Game logic (để chắc chắn không bị logic energy làm sai mask node 0)
#         if is_all_served.any():
#             served_indices = torch.where(is_all_served)[0]
#             valid_mask[served_indices, 1:] = 0
            
#         return valid_mask

#     def _update_vehicle_mask(self):
#         """
#         Logic chọn xe:
#         - Xe hết pin/hư hỏng -> Mask 0.
#         - Nếu Hết khách (End Game):
#             + Xe ĐANG Ở Node 0 -> Mask 0 (Đã về đích, không chọn nữa).
#             + Xe CHƯA Ở Node 0 -> Mask 1 (Cần chọn để đưa về).
#         """
#         mask = torch.ones(self.batch_size, self.num_trucks + self.num_drones, device=self.device)
        
#         # 1. Check Energy (Drone)
#         energies = self.dynamic_drone[:, 2, :]
#         mask[:, self.num_trucks:] = (energies > 0.05).float()
        
#         # 2. Check End Game Status
#         unvisited_count = self.mask_cust[:, 1:].sum(dim=1)
#         is_all_served = (unvisited_count == 0)
        
#         if is_all_served.any():
#             served_idx = torch.where(is_all_served)[0]
            
#             # Logic: Chỉ kích hoạt những xe CHƯA về nhà
#             # Truck locations
#             t_locs = self.dynamic_truck[served_idx, 0, :]
#             t_needs_return = (t_locs != 0).float()
#             mask[served_idx, :self.num_trucks] = t_needs_return
            
#             # Drone locations
#             d_locs = self.dynamic_drone[served_idx, 0, :]
#             d_needs_return = (d_locs != 0).float()
            
#             # Kết hợp với mask energy cũ
#             mask[served_idx, self.num_trucks:] *= d_needs_return
            
#         return mask

#     def _update_drone_state(self, veh_idx, node_idx, active_mask):
#         """Drone: Cập nhật vị trí, thời gian, năng lượng"""
#         if not active_mask.any(): return
#         b_idx = torch.where(active_mask)[0]
#         coords = self.static[b_idx, :2, :]
#         curr_nodes = self.dynamic_drone[b_idx, 0, veh_idx[b_idx]].long()
#         target_nodes = node_idx[b_idx]
        
#         # Check: Đang ở Depot VÀ Đi đến Depot -> Đứng yên (Wait/Recharge)
#         is_staying = (curr_nodes == 0) & (target_nodes == 0)
#         move_factor = (~is_staying).float()
        
#         curr_xy = self._gather_coords(coords, curr_nodes)
#         target_xy = self._gather_coords(coords, target_nodes)
        
#         dist = torch.norm(target_xy - curr_xy, dim=1) * self.scale[b_idx, 0]
#         payloads = self.static[b_idx, 2, target_nodes] 
        
#         p = self.config.drone_params
#         power = p['gama(w)'] + p['beta(w/kg)'] * payloads
#         t_takeoff, t_landing = self.config.t_takeoff, self.config.t_landing
#         t_cruise = dist / self.config.drone_speed
        
#         # Tính năng lượng: Chỉ tốn khi di chuyển
#         energy_joule = power * ((t_takeoff + t_landing) * move_factor + t_cruise)
#         norm_cost = energy_joule / self.config.drone_max_energy
        
#         # Tính thời gian: Di chuyển tốn time bay, Đứng yên tốn time chờ (ví dụ 30s)
#         wait_time = 30.0 
#         total_time = ((t_takeoff + t_landing) * move_factor + t_cruise)
#         total_time = torch.where(is_staying, torch.tensor(wait_time, device=self.device), total_time)

#         # Tính payloads
#         current_payloads = self.dynamic_drone[b_idx, 3, veh_idx[b_idx]]
#         target_demands = self.static[b_idx, 2, target_nodes]
#         new_payloads = current_payloads + target_demands
        
#         # Update State
#         self.dynamic_drone[b_idx, 0, veh_idx[b_idx]] = target_nodes.float()
#         self.dynamic_drone[b_idx, 1, veh_idx[b_idx]] += total_time
#         self.dynamic_drone[b_idx, 2, veh_idx[b_idx]] -= norm_cost
        
#         # Recharge Logic: Cứ về Depot là đầy pin
#         is_back_depot = (target_nodes == 0)
#         new_payloads = torch.where(is_back_depot, torch.zeros_like(new_payloads), new_payloads)
#         # Update payloads
#         self.dynamic_drone[b_idx, 3, veh_idx[b_idx]] = new_payloads

#         if is_back_depot.any():
#             idx_reset = b_idx[is_back_depot]
#             d_ids_reset = veh_idx[idx_reset]
#             self.dynamic_drone[idx_reset, 2, d_ids_reset] = 1.0
#             self.dynamic_drone[idx_reset, 3, d_ids_reset] = 0.0

#     def _update_truck_state(self, veh_idx, node_idx, active_mask):
#         """Truck: Cập nhật vị trí, thời gian"""
#         if not active_mask.any(): return
#         b_idx = torch.where(active_mask)[0]
#         coords = self.static[b_idx, :2, :]
#         curr_nodes = self.dynamic_truck[b_idx, 0, veh_idx[b_idx]].long()
#         target_nodes = node_idx[b_idx]
#         curr_xy = self._gather_coords(coords, curr_nodes)
#         target_xy = self._gather_coords(coords, target_nodes)
#         dist = torch.norm(target_xy - curr_xy, dim=1) * self.scale[b_idx, 0]
#         current_times = self.dynamic_truck[b_idx, 1, veh_idx[b_idx]]
#         speeds = self.config.get_truck_speed_batch(current_times)
#         travel_time = dist / speeds
#         self.dynamic_truck[b_idx, 0, veh_idx[b_idx]] = target_nodes.float()
#         self.dynamic_truck[b_idx, 1, veh_idx[b_idx]] += travel_time

#     def _gather_coords(self, coords_batch, node_indices):
#         idx_expanded = node_indices.view(-1, 1, 1).expand(-1, 2, -1)
#         return torch.gather(coords_batch, 2, idx_expanded).squeeze(2)


#     def _calculate_terminal_reward(self):
#         w1 = self.weights[:, 0]
#         w2 = self.weights[:, 1]
        
#         # Makespan: Max thời gian của các xe
#         truck_times = self.dynamic_truck[:, 1, :]
#         drone_times = self.dynamic_drone[:, 1, :]
#         all_times = torch.cat([truck_times, drone_times], dim=1)
#         makespan, _ = torch.max(all_times, dim=1)
        
#         waiting_time = self.total_waiting_time
        
#         # Reward = -(w1 * Makespan + w2 * WaitingTime)
#         reward = -(w1 * makespan + w2 * waiting_time)
        
#         return reward / 1000.0

#     def _update_routes_history(self, veh_idx, node_idx):
#         v_idx = veh_idx.cpu().numpy()
#         n_idx = node_idx.cpu().numpy()
#         for b in range(self.batch_size):
#             node_val = int(n_idx[b])
#             if v_idx[b] < self.num_trucks:
#                 self.routes[0][b]['trucks'][v_idx[b]].append(node_val)
#             else:
#                 d_id = v_idx[b] - self.num_trucks
#                 self.routes[0][b]['drones'][d_id].append(node_val)


import torch
import numpy as np

# from config import SystemConfig
# from model import MOPVRP_Actor
# from dataloader import get_rl_dataloader
# from visualizer import visualize_mopvrp

class MOPVRPEnvironment:
    def __init__(self, config, dataloader, device='cpu'):
        self.config = config
        self.dataloader = dataloader
        self.device = device
        self.data_iter = iter(dataloader)
        
        # State placeholders
        self.static = None       # [B, 4, N]
        self.dynamic_truck = None # [B, 2, K]
        self.dynamic_drone = None # [B, 4, D]
        self.mask_cust = None     # [B, N]
        self.scale = None         # [B, 1]
        self.weights = None       # [B, 2]

        # Scenario Configuraion
        self.scenario = 5     # Default


    def reset(self):
        batch_data = next(self.data_iter)
        static_src, self.dynamic_truck, self.dynamic_drone, \
        self.mask_cust, mask_veh, self.scale, self.weights = batch_data
        
        self.static = static_src.clone()
        self.batch_size = self.static.size(0)
        self.num_nodes = self.static.size(2)
        self.num_trucks = self.dynamic_truck.size(2)
        self.num_drones = self.dynamic_drone.size(2)

        # print(f"Current Scenario: {self.scenario}, Num Trucks: {self.num_trucks}, Num Drones: {self.num_drones}")
        
        self.routes = [[{'trucks': [[0] for _ in range(self.num_trucks)], 
                         'drones': [[0] for _ in range(self.num_drones)]} 
                        for _ in range(self.batch_size)]]
        
        self.total_waiting_time = torch.zeros(self.batch_size, device=self.device)
        
        return (self.static, self.dynamic_truck, self.dynamic_drone, self.mask_cust, self._update_vehicle_mask())

    def step(self, selected_vehicle_idx, selected_node_idx):
        is_drone = selected_vehicle_idx >= self.num_trucks
        drone_idx = selected_vehicle_idx - self.num_trucks
        truck_idx = selected_vehicle_idx

        prev_truck_times = self.dynamic_truck[:, 1, :].clone()
        prev_drone_times = self.dynamic_drone[:, 1, :].clone()
        
        # 1. Update Physics
        self._update_truck_state(truck_idx, selected_node_idx, ~is_drone)
        self._update_drone_state(drone_idx, selected_node_idx, is_drone)

        # --- Cập nhật Demand trong Static Feature ---
        # Logic: Node được chọn -> Demand giảm về 0 (Đã phục vụ xong)
        
        batch_indices = torch.arange(self.batch_size, device=self.device)
        # Chỉ update nếu node được chọn không phải Depot (0)
        is_customer_node = (selected_node_idx != 0)
        
        if is_customer_node.any():
            active_batch = batch_indices[is_customer_node]
            active_nodes = selected_node_idx[is_customer_node]
            
            # Gán Demand = 0.0 trực tiếp vào Static tensor
            print("Customer: ", self.static[active_batch])
            self.static[active_batch, 2, active_nodes] = 0.0
        # --------------------------------------------------------
        
        is_customer = (selected_node_idx != 0)
        
        if is_customer.any():
            current_times = torch.zeros(self.batch_size, device=self.device)

            if (~is_drone).any():
                b_tr = torch.where(~is_drone)[0]
                tr_ids = truck_idx[b_tr]
                current_times[b_tr] = self.dynamic_truck[b_tr, 1, tr_ids]
                
            # Drone times
            if is_drone.any():
                b_dr = torch.where(is_drone)[0]
                dr_ids = drone_idx[b_dr]
                current_times[b_dr] = self.dynamic_drone[b_dr, 1, dr_ids]
            
            self.total_waiting_time += current_times * is_customer.float()

        # 2. Update Customer Mask
        # Nếu chọn Node != 0 thì đánh dấu đã thăm
        node_mask_update = torch.nn.functional.one_hot(selected_node_idx.long(), num_classes=self.num_nodes)
        not_depot = (selected_node_idx != 0).float().unsqueeze(1)
        node_mask_update = node_mask_update * not_depot
        self.mask_cust = self.mask_cust * (1 - node_mask_update)
        
        # 3. CHECK DONE CONDITION
        unvisited_count = self.mask_cust[:, 1:].sum(dim=1)
        
        # Check Truck Location
        truck_locs = self.dynamic_truck[:, 0, :]
        all_trucks_home = (truck_locs == 0).all(dim=1)
        
        # Check Drone Location
        drone_locs = self.dynamic_drone[:, 0, :]
        all_drones_home = (drone_locs == 0).all(dim=1)
        
        # Điều kiện dừng đầy đủ
        done = (unvisited_count == 0) & all_trucks_home & all_drones_home
        
        # Check Force Stop (Deadlock): Nếu còn khách hoặc còn xe ngoài đường, 
        next_mask_veh = self._update_vehicle_mask()
        no_valid_vehicle = (next_mask_veh.sum(dim=1) == 0)
        
        # Nếu bị deadlock (chưa xong việc mà hết xe), ta vẫn trả về done=True để reset môi trường, nhưng sẽ phạt nặng ở reward.
        done = done | no_valid_vehicle

        delta_truck = (self.dynamic_truck[:, 1, :] - prev_truck_times).sum(dim=1)
        delta_drone = (self.dynamic_drone[:, 1, :] - prev_drone_times).sum(dim=1)
        
        # Tổng thời gian tiêu tốn của cả đội trong bước này
        step_cost = delta_truck + delta_drone
        
        # 4. Reward
        step_reward = torch.full((self.batch_size,), -0.1, device=self.device)
        step_reward = -(step_cost / 1000.0)
        
        if done.any():
            final_rewards = self._calculate_terminal_reward()
            # Nếu done do deadlock (vẫn còn khách hoặc xe chưa về): Phạt nặng
            is_failure = (unvisited_count > 0) | (~all_trucks_home) | (~all_drones_home)
            penalty = torch.where(is_failure, torch.tensor(-10.0, device=self.device), torch.tensor(0.0, device=self.device))
            
            step_reward = torch.where(done, final_rewards + penalty, step_reward)
            
        self._update_routes_history(selected_vehicle_idx, selected_node_idx)
        
        next_state = (self.static, self.dynamic_truck, self.dynamic_drone, self.mask_cust, next_mask_veh)
        return next_state, step_reward, done, {}

    def get_valid_customer_mask(self, selected_vehicle_idx):
        """
        Logic Mask Node:
        - Nếu HẾT khách: Bắt buộc chọn Node 0 (cho cả Truck và Drone).
        """
        valid_mask = self.mask_cust.clone()
        
        unvisited_count = self.mask_cust[:, 1:].sum(dim=1)
        is_all_served = (unvisited_count == 0)
        
        # --- END LOGIC ---
        if is_all_served.any():
            served_indices = torch.where(is_all_served)[0]
            valid_mask[served_indices, :] = 0
            valid_mask[served_indices, 0] = 1
        
        # --- NORMAL LOGIC (Khi còn khách) ---
        batch_indices = torch.arange(self.batch_size, device=self.device)
        is_drone = selected_vehicle_idx >= self.num_trucks
        
        # Truck Logic (Start at Depot restriction)
        if (~is_drone).any():
            t_indices = torch.where(~is_drone)[0]
            t_ids = selected_vehicle_idx[t_indices]
            current_times = self.dynamic_truck[t_indices, 1, t_ids]
            
            # Nếu đang ở Depot (Time=0) -> Không được chọn Node 0
            at_start = (current_times == 0)
            if at_start.any():
                start_indices = t_indices[at_start]
                valid_mask[start_indices, 0] = 0
        
        if is_drone.any():
            d_indices = torch.where(is_drone)[0]
            d_ids = selected_vehicle_idx[d_indices] - self.num_trucks
            
            # 1. Enable Node 0 (Luôn cho phép về Depot sạc/chờ)
            valid_mask[d_indices, 0] = 1
            
            # 2. Mask Truck-only customers
            truck_only_flags = self.static[d_indices, 3, :]
            valid_mask[d_indices] *= (1 - truck_only_flags)
            
            # 3. CHECK ENERGY: (Curr -> Node) + (Node -> Depot)
            p = self.config.drone_params
            speed = self.config.drone_speed
            t_const = self.config.t_takeoff + self.config.t_landing
            
            # A. Tính chặng đi: Current -> Candidate Node
            curr_nodes = self.dynamic_drone[d_indices, 0, d_ids].long()
            coords = self.static[d_indices, :2, :] # (N_drone, 2, N_nodes)
            
            curr_xy = self._gather_coords(coords, curr_nodes)
            curr_xy_exp = curr_xy.unsqueeze(2) 
            
            # Khoảng cách đi (N_drone, N_nodes)
            dist_go = torch.norm(coords - curr_xy_exp, dim=1) * self.scale[d_indices, 0].unsqueeze(1)
            
            # B. Tính chặng về: Candidate Node -> Depot (Node 0)
            depot_xy = coords[:, :, 0].unsqueeze(2) # (N_drone, 2, 1)
            dist_return = torch.norm(coords - depot_xy, dim=1) * self.scale[d_indices, 0].unsqueeze(1)
            
            # C. Tính Năng Lượng
            payloads = self.static[d_indices, 2, :] 
            
            # Công suất (Power)
            power = p['gama(w)'] + p['beta(w/kg)'] * payloads
            
            # Năng lượng chặng đi
            time_go = dist_go / speed + t_const
            energy_go = power * time_go
            
            # Năng lượng chặng về
            time_return = dist_return / speed + t_const
            energy_return = power * time_return
            
            # Tổng năng lượng cần thiết
            total_energy_req = energy_go + energy_return

            # --- Xử lý ngoại lệ cho Node 0 (Depot) ---
            # Nếu ĐANG Ở Depot và chọn Node 0 (Đứng chờ): Energy req gần như bằng 0 (hoặc rất nhỏ).
            is_at_depot = (curr_nodes == 0).unsqueeze(1)
            target_is_depot = torch.zeros_like(total_energy_req, dtype=torch.bool)
            target_is_depot[:, 0] = True
            
            # Nếu (Đang ở 0) VÀ (Chọn 0) -> Energy Req = 0
            stay_mask = is_at_depot & target_is_depot
            total_energy_req[stay_mask] = 0.0
            
            # D. So sánh với Pin hiện tại
            energy_req_norm = total_energy_req / self.config.drone_max_energy
            curr_energy = self.dynamic_drone[d_indices, 2, d_ids].unsqueeze(1)
            
            # Tạo mask
            energy_mask = (curr_energy >= energy_req_norm).float()
            valid_mask[d_indices] *= energy_mask

            # 4. Check payloads

            # Payload hiện tại
            curr_load = self.dynamic_drone[d_indices, 3, d_ids].unsqueeze(1) # (N, 1)
            # Demand của các khách hàng tiềm năng
            demands = self.static[d_indices, 2, :] # (N, Nodes)
            # Capacity Max
            max_cap = self.config.drone_capacity_kg 
            
            # Điều kiện: Load sau khi nhặt <= Max Cap
            cap_mask = (curr_load + demands) <= max_cap
            valid_mask[d_indices] *= cap_mask.float()

        # Re-apply End Game logic (để chắc chắn không bị logic energy làm sai mask node 0)
        if is_all_served.any():
            served_indices = torch.where(is_all_served)[0]
            valid_mask[served_indices, 1:] = 0
            
        return valid_mask

    def _update_vehicle_mask(self):
        """
        Logic chọn xe:
        - Xe hết pin/hư hỏng -> Mask 0.
        - Nếu Hết khách (End Game):
            + Xe ĐANG Ở Node 0 -> Mask 0 (Đã về đích, không chọn nữa).
            + Xe CHƯA Ở Node 0 -> Mask 1 (Cần chọn để đưa về).
        """
        mask = torch.ones(self.batch_size, self.num_trucks + self.num_drones, device=self.device)
        
        # 1. Check Energy (Drone)
        energies = self.dynamic_drone[:, 2, :]
        mask[:, self.num_trucks:] = (energies > 0.05).float()
        
        # 2. Check End Game Status
        unvisited_count = self.mask_cust[:, 1:].sum(dim=1)
        is_all_served = (unvisited_count == 0)
        
        if is_all_served.any():
            served_idx = torch.where(is_all_served)[0]
            
            # Logic: Chỉ kích hoạt những xe CHƯA về nhà
            # Truck locations
            t_locs = self.dynamic_truck[served_idx, 0, :]
            t_needs_return = (t_locs != 0).float()
            mask[served_idx, :self.num_trucks] = t_needs_return
            
            # Drone locations
            d_locs = self.dynamic_drone[served_idx, 0, :]
            d_needs_return = (d_locs != 0).float()
            
            # Kết hợp với mask energy cũ
            mask[served_idx, self.num_trucks:] *= d_needs_return
            
        return mask

    def _update_drone_state(self, veh_idx, node_idx, active_mask):
        """Drone: Cập nhật vị trí, thời gian, năng lượng"""
        if not active_mask.any(): return
        b_idx = torch.where(active_mask)[0]
        coords = self.static[b_idx, :2, :]
        curr_nodes = self.dynamic_drone[b_idx, 0, veh_idx[b_idx]].long()
        target_nodes = node_idx[b_idx]
        
        # Check: Đang ở Depot VÀ Đi đến Depot -> Đứng yên (Wait/Recharge)
        is_staying = (curr_nodes == 0) & (target_nodes == 0)
        move_factor = (~is_staying).float()
        
        curr_xy = self._gather_coords(coords, curr_nodes)
        target_xy = self._gather_coords(coords, target_nodes)
        
        dist = torch.norm(target_xy - curr_xy, dim=1) * self.scale[b_idx, 0]
        payloads = self.static[b_idx, 2, target_nodes] 
        
        p = self.config.drone_params
        power = p['gama(w)'] + p['beta(w/kg)'] * payloads
        t_takeoff, t_landing = self.config.t_takeoff, self.config.t_landing
        t_cruise = dist / self.config.drone_speed
        
        # Tính năng lượng: Chỉ tốn khi di chuyển
        energy_joule = power * ((t_takeoff + t_landing) * move_factor + t_cruise)
        norm_cost = energy_joule / self.config.drone_max_energy
        
        # Tính thời gian: Di chuyển tốn time bay, Đứng yên tốn time chờ (ví dụ 30s)
        wait_time = 30.0 
        total_time = ((t_takeoff + t_landing) * move_factor + t_cruise)
        total_time = torch.where(is_staying, torch.tensor(wait_time, device=self.device), total_time)

        # Tính payloads
        current_payloads = self.dynamic_drone[b_idx, 3, veh_idx[b_idx]]
        target_demands = self.static[b_idx, 2, target_nodes]
        new_payloads = current_payloads + target_demands
        
        # Update State
        self.dynamic_drone[b_idx, 0, veh_idx[b_idx]] = target_nodes.float()
        self.dynamic_drone[b_idx, 1, veh_idx[b_idx]] += total_time
        self.dynamic_drone[b_idx, 2, veh_idx[b_idx]] -= norm_cost
        
        
        # Recharge Logic: Cứ về Depot là đầy pin
        is_back_depot = (target_nodes == 0)
        new_payloads = torch.where(is_back_depot, torch.zeros_like(new_payloads), new_payloads)
        # Update payloads
        self.dynamic_drone[b_idx, 3, veh_idx[b_idx]] = new_payloads

        if is_back_depot.any():
            idx_reset = b_idx[is_back_depot]
            d_ids_reset = veh_idx[idx_reset]
            self.dynamic_drone[idx_reset, 2, d_ids_reset] = 1.0
            self.dynamic_drone[idx_reset, 3, d_ids_reset] = 0.0

    def _update_truck_state(self, veh_idx, node_idx, active_mask):
        """Truck: Cập nhật vị trí, thời gian"""
        if not active_mask.any(): return
        b_idx = torch.where(active_mask)[0]
        coords = self.static[b_idx, :2, :]
        curr_nodes = self.dynamic_truck[b_idx, 0, veh_idx[b_idx]].long()
        target_nodes = node_idx[b_idx]
        curr_xy = self._gather_coords(coords, curr_nodes)
        target_xy = self._gather_coords(coords, target_nodes)
        dist = torch.norm(target_xy - curr_xy, dim=1) * self.scale[b_idx, 0]
        current_times = self.dynamic_truck[b_idx, 1, veh_idx[b_idx]]
        speeds = self.config.get_truck_speed_batch(current_times)
        travel_time = dist / speeds
        self.dynamic_truck[b_idx, 0, veh_idx[b_idx]] = target_nodes.float()
        self.dynamic_truck[b_idx, 1, veh_idx[b_idx]] += travel_time

        

    def _gather_coords(self, coords_batch, node_indices):
        idx_expanded = node_indices.view(-1, 1, 1).expand(-1, 2, -1)
        return torch.gather(coords_batch, 2, idx_expanded).squeeze(2)


    def _calculate_terminal_reward(self):
        w1 = self.weights[:, 0]
        w2 = self.weights[:, 1]
        
        # Makespan: Max thời gian của các xe
        truck_times = self.dynamic_truck[:, 1, :]
        drone_times = self.dynamic_drone[:, 1, :]
        all_times = torch.cat([truck_times, drone_times], dim=1)
        makespan, _ = torch.max(all_times, dim=1)
        
        waiting_time = self.total_waiting_time
        
        # Reward = -(w1 * Makespan + w2 * WaitingTime)
        reward = -(w1 * makespan + w2 * waiting_time)
        
        return reward / 100000.0
        # return reward

    def _update_routes_history(self, veh_idx, node_idx):
        v_idx = veh_idx.cpu().numpy()
        n_idx = node_idx.cpu().numpy()
        for b in range(self.batch_size):
            node_val = int(n_idx[b])
            if v_idx[b] < self.num_trucks:
                self.routes[0][b]['trucks'][v_idx[b]].append(node_val)
            else:
                d_id = v_idx[b] - self.num_trucks
                self.routes[0][b]['drones'][d_id].append(node_val)