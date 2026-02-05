import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# PHẦN 1: CÁC KHỐI XÂY DỰNG (BUILDING BLOCKS)
# ============================================================================

class PairwiseEmbedding(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(PairwiseEmbedding, self).__init__()
        self.conv2d = nn.Conv2d(static_size + dynamic_size, hidden_size, kernel_size=1)
        
    def forward(self, static, dynamic):
        # static: [B, S, N], dynamic: [B, D, V]
        B, S_Feat, N_Cust = static.size()
        _, D_Feat, N_Veh = dynamic.size()
        
        static_expanded = static.unsqueeze(2).expand(-1, -1, N_Veh, -1)
        dynamic_expanded = dynamic.unsqueeze(3).expand(-1, -1, -1, N_Cust)
        
        combined = torch.cat([static_expanded, dynamic_expanded], dim=1)
        return self.conv2d(combined)

class HierarchicalDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(HierarchicalDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # Vehicle Attention
        self.W_veh = nn.Linear(hidden_size * 2, hidden_size)
        self.v_veh = nn.Parameter(torch.rand(hidden_size))
        
        # Customer Attention
        self.W_cust = nn.Linear(hidden_size * 2, hidden_size)
        self.v_cust = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, pairwise_embeds, decoder_input, last_hh, mask_vehicles=None, mask_pairwise=None, deterministic=False, forced_action=None):
        h_t, c_t = last_hh
        h_t, c_t = self.lstm(decoder_input, (h_t, c_t))
        
        B, H, N_Veh, N_Cust = pairwise_embeds.size()
        
        # --- 1. Chọn Vehicle ---
        veh_repr = pairwise_embeds.mean(dim=3)
        h_t_expanded_v = h_t.unsqueeze(2).expand(-1, -1, N_Veh)
        veh_att_input = torch.cat([veh_repr, h_t_expanded_v], dim=1).transpose(1, 2)
        veh_energy = torch.matmul(torch.tanh(self.W_veh(veh_att_input)), self.v_veh)
        
        if mask_vehicles is not None:
            veh_energy = veh_energy.masked_fill(mask_vehicles == 0, float('-inf'))
        
        veh_probs = F.softmax(veh_energy, dim=1)
        
        if forced_action is not None:
            selected_veh_idx = forced_action[0]
        else:
            if deterministic:
                selected_veh_idx = torch.argmax(veh_probs, dim=1)
            else:
                dist = torch.distributions.Categorical(veh_probs)
                selected_veh_idx = dist.sample()

        # --- 2. Chọn Customer ---
        idx_view = selected_veh_idx.view(B, 1, 1, 1).expand(-1, H, 1, N_Cust)
        selected_veh_cust_embeds = pairwise_embeds.gather(2, idx_view).squeeze(2)
        
        h_t_expanded_c = h_t.unsqueeze(2).expand(-1, -1, N_Cust)
        cust_att_input = torch.cat([selected_veh_cust_embeds, h_t_expanded_c], dim=1).transpose(1, 2)
        cust_energy = torch.matmul(torch.tanh(self.W_cust(cust_att_input)), self.v_cust)
        
        current_mask_cust = None
        if mask_pairwise is not None:
            idx_mask = selected_veh_idx.view(B, 1, 1).expand(-1, 1, N_Cust)
            current_mask_cust = mask_pairwise.gather(1, idx_mask).squeeze(1)
        
        if current_mask_cust is not None:
            cust_energy = cust_energy.masked_fill(current_mask_cust == 0, float('-inf'))
        
        cust_probs = F.softmax(cust_energy, dim=1)
        
        # --- Tính Log Probs (cho PPO) ---
        action_log_prob = None
        entropy = None
        
        if forced_action is not None:
            # Vehicle LogProb
            veh_log_dist = F.log_softmax(veh_energy, dim=1)
            chosen_veh_lp = veh_log_dist.gather(1, selected_veh_idx.unsqueeze(1))
            
            # Node LogProb
            cust_log_dist = F.log_softmax(cust_energy, dim=1)
            selected_node_idx = forced_action[1]
            chosen_node_lp = cust_log_dist.gather(1, selected_node_idx.unsqueeze(1))
            
            action_log_prob = chosen_veh_lp + chosen_node_lp
            
            dist_v = torch.distributions.Categorical(logits=veh_energy)
            dist_n = torch.distributions.Categorical(logits=cust_energy)
            entropy = dist_v.entropy() + dist_n.entropy()

        return veh_probs, cust_probs, selected_veh_idx, (h_t, c_t), action_log_prob, entropy


# ============================================================================
# PHẦN 2: ACTOR & CRITIC
# ============================================================================

class MOPVRP_Actor(nn.Module):
    def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone, hidden_size):
        super(MOPVRP_Actor, self).__init__()
        self.max_dyn_size = max(dynamic_size_truck, dynamic_size_drone)
        self.pairwise_encoder = PairwiseEmbedding(static_size, self.max_dyn_size, hidden_size)
        self.coords_embedding = nn.Linear(2, hidden_size)
        self.decoder = HierarchicalDecoder(hidden_size)
        
        self.x0 = nn.Parameter(torch.zeros(1, 2))
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_size))
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1: nn.init.xavier_uniform_(param)
            if 'bias' in name: nn.init.constant_(param, 0)

    def _pad_vehicles(self, trucks, drones):
        def pad(tensor, target):
            diff = target - tensor.size(1)
            if diff > 0:
                p = torch.zeros(tensor.size(0), diff, tensor.size(2), device=tensor.device)
                return torch.cat([tensor, p], dim=1)
            return tensor
        t = pad(trucks, self.max_dyn_size)
        d = pad(drones, self.max_dyn_size)
        return torch.cat([t, d], dim=2)

    def forward(self, static, dynamic_trucks, dynamic_drones, decoder_input=None, last_hh=None, 
                mask_vehicles=None, mask_pairwise=None, deterministic=False, forced_action=None):
        
        B = static.size(0)
        dynamic_vehicles = self._pad_vehicles(dynamic_trucks, dynamic_drones)
        pairwise_embeds = self.pairwise_encoder(static, dynamic_vehicles)
        
        if decoder_input is None: decoder_input = self.x0.expand(B, -1)
        decoder_input_embed = self.coords_embedding(decoder_input)
        
        if last_hh is None:
            last_hh = (self.h0.expand(B, -1), self.c0.expand(B, -1))
            
        return self.decoder(pairwise_embeds, decoder_input_embed, last_hh, 
                            mask_vehicles, mask_pairwise, deterministic, forced_action)

class Critic(nn.Module):
    def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone, hidden_size):
        super(Critic, self).__init__()
        self.static_conv = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.truck_conv = nn.Conv1d(dynamic_size_truck, hidden_size, kernel_size=1)
        self.drone_conv = nn.Conv1d(dynamic_size_drone, hidden_size, kernel_size=1)
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, static, dynamic_trucks, dynamic_drones):
        s = self.relu(self.static_conv(static)).mean(2)
        t = self.relu(self.truck_conv(dynamic_trucks)).mean(2)
        d = self.relu(self.drone_conv(dynamic_drones)).mean(2)
        x = torch.cat([s, t, d], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)



# ============================================================================
# PHẦN 3: MAIN CLASS
# ============================================================================

class MOPRVP(nn.Module):
    def __init__(self, static_size, dyn_truck_size, dyn_drone_size, hidden_size, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.actor = MOPVRP_Actor(static_size, dyn_truck_size, dyn_drone_size, hidden_size).to(device)
        self.critic = Critic(static_size, dyn_truck_size, dyn_drone_size, hidden_size).to(device)
        self.num_trucks = None 

    def _unpack_obs(self, obs):
        graph = obs['graph_ctx'].transpose(1, 2)
        trucks = obs['trucks_ctx'].transpose(1, 2)
        drones = obs['drones_ctx'].transpose(1, 2)
        self.num_trucks = trucks.size(2)
        
        mask_trk = obs['mask_trk']
        mask_dr = obs['mask_dr']
        mask_pairwise = torch.cat([mask_trk, mask_dr], dim=1)
        mask_vehicles = mask_pairwise.any(dim=2).float()
        
        return graph, trucks, drones, mask_vehicles, mask_pairwise

    def _pack_hidden(self, hidden_tuple):
        """Gộp (h, c) thành Tensor [B, 2*H]"""
        if hidden_tuple is None: return None
        return torch.cat(hidden_tuple, dim=-1)

    def _unpack_hidden(self, hidden_tensor):
        """Tách Tensor [B, 2*H] thành (h, c)"""
        if hidden_tensor is None: return None
        # Split đôi
        h, c = hidden_tensor.chunk(2, dim=-1)
        return (h.contiguous(), c.contiguous())

    # --- 1. Init Hidden (Trả về Tensor) ---
    def init_hidden(self, batch_size, device):
        h0 = self.actor.h0.expand(batch_size, -1).to(device)
        c0 = self.actor.c0.expand(batch_size, -1).to(device)
        return torch.cat([h0, c0], dim=-1) # [B, 2H]

    # --- 2. Act (Nhận Tensor -> Trả về Tensor) ---
    @torch.no_grad()
    def act(self, obs, h_prev=None, deterministic=False):
        graph, trucks, drones, mask_veh, mask_pair = self._unpack_obs(obs)
        
        # Unpack h_prev từ Tensor -> Tuple
        last_hh = self._unpack_hidden(h_prev)
        
        veh_probs, cust_probs, flat_veh_idx, h_next_tuple, _, _ = self.actor(
            graph, trucks, drones, 
            last_hh=last_hh, 
            mask_vehicles=mask_veh, 
            mask_pairwise=mask_pair, 
            deterministic=deterministic, 

        )
        
        if deterministic:
            node_idx = torch.argmax(cust_probs, dim=1)
        else:
            dist = torch.distributions.Categorical(cust_probs)
            node_idx = dist.sample()
            
        value = self.critic(graph, trucks, drones)
        
        is_drone = (flat_veh_idx >= self.num_trucks)
        veh_type = is_drone.long()
        veh_inst = torch.where(is_drone, flat_veh_idx - self.num_trucks, flat_veh_idx)
        
        action_tuple = (veh_type, veh_inst, node_idx)
        log_prob = torch.zeros(graph.size(0), 1, device=self.device)
        
        # Pack h_next từ Tuple -> Tensor để trả về cho train.py
        h_next_tensor = self._pack_hidden(h_next_tuple)
        
        return action_tuple, log_prob, value.unsqueeze(1), {}, h_next_tensor

    # --- 3. Evaluate Actions (Nhận Tensor) ---
    def evaluate_actions(self, flat_obs, actions, h_prev=None):
        graph, trucks, drones, mask_veh, mask_pair = self._unpack_obs(flat_obs)
        
        # Unpack h_prev
        last_hh = self._unpack_hidden(h_prev)
        
        veh_type = actions[:, 0]
        veh_inst = actions[:, 1]
        node_idx = actions[:, 2]
        
        flat_veh_idx = torch.where(veh_type == 0, veh_inst, veh_inst + self.num_trucks)
        forced_action = (flat_veh_idx, node_idx)
        
        _, _, _, _, log_prob, entropy = self.actor(
            graph, trucks, drones, 
            last_hh=last_hh, 
            mask_vehicles=mask_veh, 
            mask_pairwise=mask_pair, 
            forced_action=forced_action
        )
        
        value = self.critic(graph, trucks, drones)
        
        return log_prob, entropy.unsqueeze(1), value.unsqueeze(1)