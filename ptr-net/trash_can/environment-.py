import torch
import numpy as np
from config import SystemConfig

class MOPVRPEnvironment:
    def __init__(self, static, dynamic_trucks, dynamic_drones, weights, scale, 
                 config_paths, device):
        self.device = device
        self.batch_size, _, self.num_nodes = static.shape
        self.num_customers = self.num_nodes - 1
        self.num_trucks = dynamic_trucks.size(2)
        self.num_drones = dynamic_drones.size(2)
        
        self.sys_config = SystemConfig(config_paths['truck'], config_paths['drone'])
        
        # Data
        self.scale = scale.view(self.batch_size, 1, 1).to(device)
        self.coords_real = static[:, :2, :] * self.scale 
        self.demands_kg = static[:, 2, :] * 50.0 
        self.truck_only = static[:, 3, :].bool()
        self.weights = weights
        
        # State
        self.truck_state = dynamic_trucks.clone()
        self.drone_state = dynamic_drones.clone()
        self.drone_state[:, 2, :] = 1.0 
        self.drone_state[:, 3, :] = 0.0 
        
        # Tracking
        self.visited = torch.zeros(self.batch_size, self.num_nodes, dtype=torch.bool, device=device)
        self.visited[:, 0] = True 
        self.pickup_times = torch.zeros(self.batch_size, self.num_nodes, device=device)
        self.service_map = torch.full((self.batch_size, self.num_nodes), -1, dtype=torch.long, device=device)
        
        # [REMOVED] Không cần truck_finished nữa vì Env sẽ tự thu xe về
        # self.truck_finished = ... 

        self.step_penalty = -0.1
        self.unfinished_penalty = 100.0

    def get_current_state(self):
        return self.truck_state, self.drone_state

    def get_mask(self):
        # 1. Customer Mask (Bỏ Depot khỏi mask node để Agent không chọn về 0)
        mask_customers = (~self.visited).float()
        mask_customers[:, 0] = 0 
        
        # Nếu hết khách (tất cả mask_customers[:, 1:] == 0), ta vẫn để mask=0
        # Hàm step sẽ tự detect điều này để kết thúc.
        
        num_veh = self.num_trucks + self.num_drones
        mask_vehicles = torch.zeros(self.batch_size, num_veh, device=self.device)
        
        drone_cap = self.sys_config.drone_capacity_kg
        drone_max_j = self.sys_config.drone_max_energy
        
        # --- CHECK TRUCK ---
        # Truck luôn active nếu còn khách. 
        # Nếu hết khách, nó sẽ bị mask=0, nhưng lúc đó step đã detect done rồi.
        has_cust = mask_customers.sum(dim=1) > 0
        for k in range(self.num_trucks):
            mask_vehicles[:, k] = has_cust.float()

        # --- CHECK DRONE ---
        for d in range(self.num_drones):
            idx = self.num_trucks + d
            curr_loc = self.drone_state[:, 0, d].long()
            curr_e = self.drone_state[:, 2, d]
            curr_load = self.drone_state[:, 3, d]
            
            can_serve_any = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            
            # Check đi Khách
            for c in range(1, self.num_nodes):
                node_valid = (mask_customers[:, c] == 1) & (~self.truck_only[:, c])
                
                if node_valid.any():
                    new_load = curr_load + self.demands_kg[:, c]
                    cap_valid = (new_load <= drone_cap)
                    
                    dist_next = self._dist_batch(curr_loc, c)      
                    dist_next_home = self._dist_batch(c, 0)        
                    
                    e_next = self._calc_energy_one_leg(dist_next, curr_load)
                    e_reserve = self._calc_energy_one_leg(dist_next_home, new_load)
                    
                    total_req = e_next + e_reserve
                    e_valid = (curr_e * drone_max_j) >= total_req
                    
                    can_serve_any = can_serve_any | (node_valid & cap_valid & e_valid)
            
            # Drone cũng có thể về Depot để sạc NẾU cần thiết (Multi-trip)
            # Nhưng để đơn giản theo yêu cầu của bạn: "Tự về khi xong việc"
            # Ta chỉ cho phép drone về Depot để sạc nếu nó KHÔNG CÒN đủ pin đi khách nào khác
            # HOẶC nếu bạn muốn cho phép sạc giữa chừng, thì giữ logic check home.
            # Ở đây tôi cho phép về sạc (Node 0) nếu đang không ở 0.
            
            dist_to_home = self._dist_batch(curr_loc, 0)
            e_return = self._calc_energy_one_leg(dist_to_home, curr_load)
            can_go_home = (curr_loc != 0) & ((curr_e * drone_max_j) >= e_return)
            
            # Mask vehicle
            mask_vehicles[:, idx] = (can_serve_any | can_go_home).float()

        # Mở Node 0 (Depot) CHỈ KHI cần thiết (ví dụ Drone về sạc)
        # Truck không bao giờ cần chọn Node 0 chủ động.
        mask_customers[:, 0] = 1 
        
        return mask_customers, mask_vehicles

    def step(self, vehicle_action, node_action):
        is_drone = vehicle_action >= self.num_trucks
        is_truck = ~is_drone
        truck_idx = vehicle_action
        drone_idx = vehicle_action - self.num_trucks
        
        # 1. Update State
        if is_truck.any():
            self._update_trucks(is_truck, truck_idx, node_action)
        if is_drone.any():
            self._update_drones(is_drone, drone_idx, node_action)
            
        # 2. Update Visited
        batch_idx = torch.arange(self.batch_size, device=self.device)
        not_depot = (node_action != 0)
        self.visited[batch_idx[not_depot], node_action[not_depot]] = True
        self.service_map[batch_idx[not_depot], node_action[not_depot]] = vehicle_action[not_depot]
        
        # 3. Check Completion (Chỉ cần check hết khách)
        all_customers_served = self.visited[:, 1:].all(dim=1)
        
        reward = torch.full((self.batch_size,), self.step_penalty, device=self.device)
        
        # [AUTO RETURN LOGIC]
        if all_customers_served.any():
            # Tự động đưa xe về và tính reward
            final_reward = self._calculate_terminal_reward_with_auto_return(all_customers_served)
            reward[all_customers_served] = final_reward[all_customers_served]
            
        # Done chỉ khi hết khách (Env tự lo phần còn lại)
        done = all_customers_served
            
        return reward, done

    def _calculate_terminal_reward_with_auto_return(self, done_mask):
        """
        Tính toán giai đoạn 'Về Depot' cho tất cả phương tiện và trả về Reward.
        Logic:
        - Nếu xe đang ở Depot: Thời gian/Pin không đổi.
        - Nếu xe đang ở Khách: Cộng thêm thời gian/Pin di chuyển về Depot.
        """
        # Lấy subset các batch đã xong
        batch_idx = torch.arange(self.batch_size, device=self.device)[done_mask]
        
        # --- 1. TÍNH CHO TRUCK ---
        # Clone thời gian hiện tại
        final_truck_times = self.truck_state[batch_idx, 1, :].clone() # (Done_Batch, Num_Trucks)
        
        for k in range(self.num_trucks):
            curr_loc = self.truck_state[batch_idx, 0, k].long()
            curr_time = self.truck_state[batch_idx, 1, k]
            
            # Tính khoảng cách về 0
            # Nếu curr_loc == 0 -> dist = 0 -> time = 0 (Thỏa mãn yêu cầu)
            dist_to_home = self._dist_batch_indices(curr_loc, torch.zeros_like(curr_loc), batch_idx)
            
            speed = self.sys_config.get_truck_speed_batch(curr_time)
            time_return = dist_to_home / speed
            
            # Cộng thêm vào tổng thời gian
            final_truck_times[:, k] += time_return

        # --- 2. TÍNH CHO DRONE ---
        # Tương tự, nếu Drone chưa về kho, bắt buộc bay về
        final_drone_times = self.drone_state[batch_idx, 1, :].clone()
        
        for d in range(self.num_drones):
            curr_loc = self.drone_state[batch_idx, 0, d].long()
            
            dist_to_home = self._dist_batch_indices(curr_loc, torch.zeros_like(curr_loc), batch_idx)
            
            v = self.sys_config.drone_speed
            # Nếu dist > 0 thì tốn thêm thời gian hạ cánh
            is_flying = (dist_to_home > 0).float()
            time_return = (dist_to_home / v) + (self.sys_config.t_landing * is_flying)
            
            final_drone_times[:, d] += time_return
            
            # (Optional) Có thể check pin ở đây, nếu không đủ về thì phạt cực nặng
            
        # --- 3. TÍNH MAKESPAN & REWARD ---
        max_truck = final_truck_times.max(dim=1)[0]
        max_drone = final_drone_times.max(dim=1)[0]
        makespan = torch.max(max_truck, max_drone)
        
        # Cost function
        cost = self.weights[batch_idx, 0] * makespan
        
        # Penalty (vẫn giữ logic phạt nếu bị force stop mà chưa xong, 
        # nhưng ở hàm này gọi khi all_served=True nên penalty=0)
        
        return -cost / 1000.0

    # ... (Các hàm _update_trucks, _update_drones, _dist_batch... giữ nguyên như cũ) ...
    # Copy lại từ phiên bản trước, chỉ cần đảm bảo _update logic đúng
    
    def _update_trucks(self, mask, veh_idx, node_idx):
        b_idx = torch.arange(self.batch_size, device=self.device)[mask]
        t_idx = veh_idx[mask]
        n_idx = node_idx[mask]
        
        curr_loc = self.truck_state[b_idx, 0, t_idx].long()
        curr_time = self.truck_state[b_idx, 1, t_idx]
        
        dist = self._dist_batch_indices(curr_loc, n_idx, b_idx)
        speed = self.sys_config.get_truck_speed_batch(curr_time)
        t_travel = dist / speed
        
        # Nếu về depot (do model chọn sạc/chờ)
        is_return = (n_idx == 0)
        t_service = 60.0
        total_time = t_travel + torch.where(is_return, torch.tensor(0.0, device=self.device), torch.tensor(t_service, device=self.device))
        
        self.truck_state[b_idx, 0, t_idx] = n_idx.float()
        self.truck_state[b_idx, 1, t_idx] += total_time
        
        if (~is_return).any():
            self.pickup_times[b_idx[~is_return], n_idx[~is_return]] = self.truck_state[b_idx[~is_return], 1, t_idx[~is_return]]

    def _update_drones(self, mask, veh_idx, node_idx):
        b_idx = torch.arange(self.batch_size, device=self.device)[mask]
        d_idx = veh_idx[mask]
        n_idx = node_idx[mask]
        
        curr_loc = self.drone_state[b_idx, 0, d_idx].long()
        curr_e_norm = self.drone_state[b_idx, 2, d_idx]
        curr_load = self.drone_state[b_idx, 3, d_idx]
        
        is_return = (n_idx == 0)
        dist = self._dist_batch_indices(curr_loc, n_idx, b_idx)
        
        e_leg = self._calc_energy_one_leg(dist, curr_load)
        e_norm = e_leg / self.sys_config.drone_max_energy
        
        v = self.sys_config.drone_speed
        t_overhead = self.sys_config.t_takeoff + self.sys_config.t_landing
        t_leg = (dist / v) + t_overhead + (30.0 if not is_return.all() else 0.0)
        
        is_move_valid = (curr_e_norm >= e_norm) & (curr_loc != n_idx)
        
        valid_mask = is_move_valid
        if valid_mask.any():
            vb, vd, vn = b_idx[valid_mask], d_idx[valid_mask], n_idx[valid_mask]
            
            self.drone_state[vb, 1, vd] += t_leg[valid_mask]
            self.drone_state[vb, 2, vd] -= e_norm[valid_mask]
            self.drone_state[vb, 0, vd] = vn.float()
            
            is_ret_sub = (vn == 0)
            if is_ret_sub.any():
                self.drone_state[vb[is_ret_sub], 3, vd[is_ret_sub]] = 0.0 
                self.drone_state[vb[is_ret_sub], 2, vd[is_ret_sub]] = 1.0 
            else:
                new_dem = self.demands_kg[vb[~is_ret_sub], vn[~is_ret_sub]]
                self.drone_state[vb[~is_ret_sub], 3, vd[~is_ret_sub]] += new_dem
                self.pickup_times[vb[~is_ret_sub], vn[~is_ret_sub]] = self.drone_state[vb[~is_ret_sub], 1, vd[~is_ret_sub]]

        invalid_mask = ~is_move_valid
        if invalid_mask.any():
            self.visited[b_idx[invalid_mask], n_idx[invalid_mask]] = False

    def _calc_energy_one_leg(self, dist, payload):
        beta = self.sys_config.drone_params['beta(w/kg)']
        gamma = self.sys_config.drone_params['gama(w)']
        v = self.sys_config.drone_speed
        t_up_down = self.sys_config.t_takeoff + self.sys_config.t_landing
        energy = (beta * payload + gamma) * (dist / v + t_up_down)
        return energy

    def _dist_batch_indices(self, idx1, idx2, batch_subset):
        p1 = self.coords_real[batch_subset, :, idx1]
        p2 = self.coords_real[batch_subset, :, idx2]
        return torch.norm(p1 - p2, dim=1)
    
    def _dist_batch(self, idx1, idx2):
        batch_idx = torch.arange(self.batch_size, device=self.device)
        if isinstance(idx1, torch.Tensor): p1 = self.coords_real[batch_idx, :, idx1]
        else: p1 = self.coords_real[:, :, idx1]
        if isinstance(idx2, torch.Tensor): p2 = self.coords_real[batch_idx, :, idx2]
        else: p2 = self.coords_real[:, :, idx2]
        return torch.norm(p1 - p2, dim=1)
    
    def calculate_forced_reward(self):
        # Tính reward phạt khi dừng sớm (vẫn dùng logic tính makespan + penalty chưa xong)
        # Chỉ gọi khi step limit reached
        
        # 1. Force return (tính thời gian về)
        # Tái sử dụng logic của _calculate_terminal_reward_with_auto_return cho toàn bộ batch
        fake_done_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        reward_base = self._calculate_terminal_reward_with_auto_return(fake_done_mask)
        
        # 2. Cộng thêm phạt cho khách chưa xong
        unserved_count = (~self.visited).sum(dim=1).float()
        penalty = unserved_count * self.unfinished_penalty
        
        return reward_base - penalty