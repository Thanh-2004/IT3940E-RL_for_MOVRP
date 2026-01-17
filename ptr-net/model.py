# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from dataloader import MOPVRPGenerator, get_rl_dataloader

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # from dataloader import MOPVRPGenerator, get_rl_dataloader

# class Encoder(nn.Module):
#     """Encodes static & dynamic features using 1D Convolution."""
#     def __init__(self, input_size, hidden_size):
#         super(Encoder, self).__init__()
#         self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
    
#     def forward(self, x):
#         return self.conv(x)

# class MultiAgentDecoder(nn.Module):
#     """Decoder for multi-agent vehicle routing."""
#     def __init__(self, hidden_size, num_layers=1, dropout=0.2):
#         super(MultiAgentDecoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
#                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
#         self.v_veh = nn.Parameter(torch.zeros((1, 1, hidden_size)))
#         self.W_veh = nn.Parameter(torch.zeros((1, hidden_size, hidden_size * 3)))
        
#         self.v_node = nn.Parameter(torch.zeros((1, 1, hidden_size)))
#         self.W_node = nn.Parameter(torch.zeros((1, hidden_size, hidden_size * 3)))
        
#         # Init weights
#         nn.init.xavier_uniform_(self.v_veh)
#         nn.init.xavier_uniform_(self.W_veh)
#         nn.init.xavier_uniform_(self.v_node)
#         nn.init.xavier_uniform_(self.W_node)
        
#         self.drop_rnn = nn.Dropout(p=dropout)
#         if num_layers == 1:
#             self.drop_hh = nn.Dropout(p=dropout)
    
#     def forward(self, customer_embeds, vehicle_embeds, decoder_hidden, last_hh):
#         batch_size = customer_embeds.size(0)
        
#         # Update LSTM
#         rnn_out, last_hh = self.lstm(decoder_hidden.transpose(2, 1), last_hh)
#         rnn_out = rnn_out.squeeze(1)
#         rnn_out = self.drop_rnn(rnn_out)
        
#         if self.num_layers == 1:
#             h_n, c_n = last_hh
#             h_n = self.drop_hh(h_n)
#             last_hh = (h_n, c_n)
        
#         # --- Attention Mechanism ---
        
#         # 1. Global Context
#         C_node = customer_embeds.mean(dim=2, keepdim=True) 
#         C_veh = vehicle_embeds.mean(dim=2, keepdim=True)   
        
#         # 2. Vehicle Selection Attention
#         h_expanded = rnn_out.unsqueeze(2).expand_as(vehicle_embeds)
#         C_node_expanded = C_node.expand_as(vehicle_embeds)
        
#         veh_input = torch.cat([C_node_expanded, h_expanded, vehicle_embeds], dim=1) 
        

#         v_veh = self.v_veh.expand(batch_size, -1, -1)
#         W_veh = self.W_veh.expand(batch_size, -1, -1)
        
#         veh_energy = torch.bmm(v_veh, torch.tanh(torch.bmm(W_veh, veh_input)))
#         veh_probs = veh_energy.squeeze(1)
        
#         # 3. Customer Selection Attention
#         h_expanded_node = rnn_out.unsqueeze(2).expand_as(customer_embeds)
#         C_veh_expanded = C_veh.expand_as(customer_embeds)
        
#         node_input = torch.cat([C_veh_expanded, h_expanded_node, customer_embeds], dim=1)
        
#         v_node = self.v_node.expand(batch_size, -1, -1)
#         W_node = self.W_node.expand(batch_size, -1, -1)
        
#         node_energy = torch.bmm(v_node, torch.tanh(torch.bmm(W_node, node_input)))
#         node_probs = node_energy.squeeze(1)
        
#         return veh_probs, node_probs, last_hh

# class MOPVRP_Actor(nn.Module):
#     def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone, 
#                  hidden_size, num_layers=1, dropout=0.2):
#         super(MOPVRP_Actor, self).__init__()
        
#         # Encoders
#         self.static_encoder = Encoder(static_size, hidden_size)
#         self.truck_encoder = Encoder(dynamic_size_truck, hidden_size)
#         self.drone_encoder = Encoder(dynamic_size_drone, hidden_size)
        
#         # Decoder input is 2D (x, y) coordinates of the last visited node
#         self.decoder = Encoder(2, hidden_size) 
#         self.pointer = MultiAgentDecoder(hidden_size, num_layers, dropout)
        
#         # Learnable initial placeholder for decoder input
#         self.x0 = nn.Parameter(torch.zeros(1, 2, 1)) 
    
#     def forward(self, static, dynamic_trucks, dynamic_drones, 
#                 decoder_input=None, last_hh=None, mask_customers=None, mask_vehicles=None):
        
#         batch_size = static.size(0)
        
#         # Prepare Decoder Input (First step uses x0)
#         if decoder_input is None:
#             decoder_input = self.x0.expand(batch_size, -1, -1)
        
#         # Prepare Masks
#         if mask_customers is None:
#             mask_customers = torch.ones(batch_size, static.size(2), device=static.device)
#         if mask_vehicles is None:
#             num_veh = dynamic_trucks.size(2) + dynamic_drones.size(2)
#             mask_vehicles = torch.ones(batch_size, num_veh, device=static.device)
        
#         # --- 1. Encoding ---
#         customer_hidden = self.static_encoder(static)      # (B, 128, N)
#         truck_hidden = self.truck_encoder(dynamic_trucks)  # (B, 128, T)
#         drone_hidden = self.drone_encoder(dynamic_drones)  # (B, 128, D)
        
#         # Combine vehicles
#         vehicle_hidden = torch.cat([truck_hidden, drone_hidden], dim=2) # (B, 128, T+D)
        
#         # --- 2. Decoding Step ---
#         decoder_hidden = self.decoder(decoder_input)
        
#         veh_logits, node_logits, last_hh = self.pointer(
#             customer_hidden, vehicle_hidden, decoder_hidden, last_hh
#         )
        
#         # --- 3. Masking & Softmax ---
#         # Masking: Set logits of invalid actions to -inf
#         # mask = 1 (valid), 0 (invalid)
#         veh_logits = veh_logits.masked_fill(mask_vehicles == 0, float('-inf'))
#         node_logits = node_logits.masked_fill(mask_customers == 0, float('-inf'))
        
#         veh_probs = F.softmax(veh_logits, dim=1)
#         node_probs = F.softmax(node_logits, dim=1)
        
#         return veh_probs, node_probs, last_hh

# class Critic(nn.Module):
#     def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone, hidden_size):
#         super(Critic, self).__init__()
#         self.static_conv = nn.Conv1d(static_size, hidden_size, kernel_size=1)
#         self.truck_conv = nn.Conv1d(dynamic_size_truck, hidden_size, kernel_size=1)
#         self.drone_conv = nn.Conv1d(dynamic_size_drone, hidden_size, kernel_size=1)
#         self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
        
#     # def forward(self, static, dynamic_trucks, dynamic_drones):
#     #     static_embed = self.static_conv(static)
#     #     truck_embed = self.truck_conv(dynamic_trucks)
#     #     drone_embed = self.drone_conv(dynamic_drones)
#     #     combined = torch.cat([static_embed.mean(2), truck_embed.mean(2), drone_embed.mean(2)], dim=1)
#     #     x = self.relu(self.fc1(combined))
#     #     x = self.dropout(x)
#     #     x = self.relu(self.fc2(x))
#     #     x = self.dropout(x)
#     #     return self.fc3(x).squeeze(-1)

#     def forward(self, static, dynamic_trucks, dynamic_drones):
#         # 1. Thêm ReLU cho phần Embedding để trích xuất tính chất phi tuyến
#         static_embed = self.relu(self.static_conv(static))
#         truck_embed = self.relu(self.truck_conv(dynamic_trucks))
#         drone_embed = self.relu(self.drone_conv(dynamic_drones))
        
#         # 2. Global Average Pooling (Giữ nguyên logic của bạn)
#         combined = torch.cat([
#             static_embed.mean(2), 
#             truck_embed.mean(2), 
#             drone_embed.mean(2)
#         ], dim=1)
        
#         # 3. Các lớp Fully Connected
#         x = self.relu(self.fc1(combined))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
        
#         # 4. Lớp Output: Tuyệt đối không có Activation
#         # Squeeze dim=1 để đảm bảo luôn giữ lại dim của Batch
#         return self.fc3(x).squeeze(1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseEmbedding(nn.Module):
    """
    Tạo vector embedding cho từng cặp (Vehicle, Customer).
    Input: Static (Customer) & Dynamic (Vehicle)
    Output: Tensor [Batch, Hidden, Num_Vehicles, Num_Customers]
    """
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(PairwiseEmbedding, self).__init__()
        # Input size = feature tĩnh + feature động
        self.conv2d = nn.Conv2d(static_size + dynamic_size, hidden_size, kernel_size=1)
        
    def forward(self, static, dynamic):
        """
        static: [Batch, Static_Feat, Num_Customers]
        dynamic: [Batch, Dyn_Feat, Num_Vehicles]
        """
        B, S_Feat, N_Cust = static.size()
        _, D_Feat, N_Veh = dynamic.size()
        
        # 1. Broadcasting để khớp kích thước
        # Static: [B, S_Feat, 1, N_Cust] -> Lặp lại cho mọi Vehicle
        static_expanded = static.unsqueeze(2).expand(-1, -1, N_Veh, -1)
        
        # Dynamic: [B, D_Feat, N_Veh, 1] -> Lặp lại cho mọi Customer
        dynamic_expanded = dynamic.unsqueeze(3).expand(-1, -1, -1, N_Cust)
        
        # 2. Concatenate: [B, S+D, N_Veh, N_Cust]
        combined = torch.cat([static_expanded, dynamic_expanded], dim=1)
        
        # 3. Embedding (Conv2d kernel 1 tương đương Linear cho từng cặp)
        # Output: [B, Hidden, N_Veh, N_Cust]
        pairwise_embeds = self.conv2d(combined)
        return pairwise_embeds

class HierarchicalDecoder(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(HierarchicalDecoder, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM để nhớ ngữ cảnh quá khứ (History)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # --- Attention cho bước 1: Chọn Vehicle ---
        # Query: LSTM State + Global Context
        # Key: Vehicle Representation (Aggregated from customers)
        self.W_veh = nn.Linear(hidden_size * 2, hidden_size) # Project Context
        self.v_veh = nn.Parameter(torch.rand(hidden_size))
        
        # --- Attention cho bước 2: Chọn Customer ---
        # Query: LSTM State + Selected Vehicle Info
        # Key: Pairwise Embedding của (Selected Vehicle, Customers)
        self.W_cust = nn.Linear(hidden_size * 2, hidden_size)
        self.v_cust = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, pairwise_embeds, decoder_input, last_hh, mask_veh=None, mask_cust=None, deterministic=False):
        """
        pairwise_embeds: [B, H, N_Veh, N_Cust]
        decoder_input: [B, H] (Embedding của node vừa ghé thăm)
        """
        h_t, c_t = last_hh
        h_t, c_t = self.lstm(decoder_input, (h_t, c_t)) # Update LSTM
        
        B, H, N_Veh, N_Cust = pairwise_embeds.size()
        
        # =========================================================
        # BƯỚC 1: CHỌN VEHICLE (Vehicle Selection)
        # =========================================================
        
        # 1. Tạo Vector đại diện cho từng Vehicle
        # Bằng cách: Gộp (Mean Pooling) tất cả Customer tương ứng với Vehicle đó
        # Shape: [B, H, N_Veh, N_Cust] -> [B, H, N_Veh]
        veh_repr = pairwise_embeds.mean(dim=3) 
        
        # 2. Tính điểm (Attention Score) cho từng Vehicle
        # Context gồm: LSTM output (h_t) mở rộng
        # Score = v^T * tanh(W_veh * [veh_repr; h_t])
        
        h_t_expanded_v = h_t.unsqueeze(2).expand(-1, -1, N_Veh) # [B, H, V]
        
        # Gộp Vehicle Rep và LSTM Context (theo chiều feature dim 1)
        # Input cho attention: [B, 2*H, V] -> transpose -> [B, V, 2*H]
        veh_att_input = torch.cat([veh_repr, h_t_expanded_v], dim=1).transpose(1, 2)
        
        # Tính Energy: [B, V, H] -> [B, V]
        veh_energy = torch.matmul(torch.tanh(self.W_veh(veh_att_input)), self.v_veh)
        
        # Masking & Softmax
        if mask_veh is not None:
            veh_energy = veh_energy.masked_fill(mask_veh == 0, float('-inf'))
        veh_probs = F.softmax(veh_energy, dim=1)
        
        # 3. Chọn Vehicle (Sampling hoặc Greedy)
        if deterministic:
            selected_veh_idx = torch.argmax(veh_probs, dim=1) # [B]
        else:
            dist = torch.distributions.Categorical(veh_probs)
            selected_veh_idx = dist.sample() # [B]

        # =========================================================
        # BƯỚC 2: CHỌN CUSTOMER (Customer Selection)
        # =========================================================
        
        # 1. Lấy vector cặp của (Vehicle ĐƯỢC CHỌN, Tất cả Customers)
        # Chúng ta cần lấy lát cắt (slice) tương ứng với selected_veh_idx
        
        # Tạo index để gather: [B, H, 1, N_Cust]
        idx_view = selected_veh_idx.view(B, 1, 1, 1).expand(-1, H, 1, N_Cust)
        # idx_view = pairwise_embeds[: : , : : , selected_veh_idx, : : ]
        
        # Gather: Lấy ra [B, H, 1, N_Cust] -> squeeze -> [B, H, N_Cust]
        # Đây là vector đặc trưng của việc "Vehicle X đi đến từng Customer"
        selected_veh_cust_embeds = pairwise_embeds.gather(2, idx_view).squeeze(2)
        print("Embedding: ", selected_veh_cust_embeds)
        
        # 2. Tính điểm cho từng Customer
        # Context: LSTM output (h_t)
        h_t_expanded_c = h_t.unsqueeze(2).expand(-1, -1, N_Cust) # [B, H, N]
        
        # Input: [Pairwise(V_selected, C); h_t]
        cust_att_input = torch.cat([selected_veh_cust_embeds, h_t_expanded_c], dim=1).transpose(1, 2)
        
        # Tính Energy: [B, N]
        cust_energy = torch.matmul(torch.tanh(self.W_cust(cust_att_input)), self.v_cust)
        
        # Masking & Softmax
        if mask_cust is not None:
            cust_energy = cust_energy.masked_fill(mask_cust == 0, float('-inf'))
        cust_probs = F.softmax(cust_energy, dim=1)
        
        # Trả về cả index xe đã chọn để bên ngoài biết
        return veh_probs, cust_probs, selected_veh_idx, (h_t, c_t)


class MOPVRP_Actor(nn.Module):
    def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone, 
                 hidden_size, dropout=0.1):
        super(MOPVRP_Actor, self).__init__()
        
        # Tự động tính kích thước Dynamic lớn nhất để Padding
        self.max_dyn_size = max(dynamic_size_truck, dynamic_size_drone)
        
        # Encoder tạo ma trận cặp
        self.pairwise_encoder = PairwiseEmbedding(static_size, self.max_dyn_size, hidden_size)
        
        # Embed tọa độ (x,y) của node trước đó làm input cho LSTM
        self.coords_embedding = nn.Linear(2, hidden_size)
        
        # Decoder chính
        self.decoder = HierarchicalDecoder(hidden_size)
        
        # Learnable initial state
        self.x0 = nn.Parameter(torch.zeros(1, 2))
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_size))
        
        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo cơ bản"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)
    
    def _pad_and_combine_vehicles(self, trucks, drones):
        """
        Hàm helper: Padding feature và gộp Truck + Drone thành 1 tensor
        Trucks: [B, F_T, N_T]
        Drones: [B, F_D, N_D]
        Output: [B, Max_F, N_T + N_D]
        """
        # Pad Truck
        diff_t = self.max_dyn_size - trucks.size(1)
        if diff_t > 0:
            pad_t = torch.zeros(trucks.size(0), diff_t, trucks.size(2), device=trucks.device)
            trucks = torch.cat([trucks, pad_t], dim=1)
            
        # Pad Drone
        diff_d = self.max_dyn_size - drones.size(1)
        if diff_d > 0:
            pad_d = torch.zeros(drones.size(0), diff_d, drones.size(2), device=drones.device)
            drones = torch.cat([drones, pad_d], dim=1)
            
        # Gộp lại
        return torch.cat([trucks, drones], dim=2)

    def forward(self, static, dynamic_trucks, dynamic_drones, 
                decoder_input=None, last_hh=None, mask_customers=None, mask_vehicles=None, deterministic=False):
        
        batch_size = static.size(0)
        
        # 1. Xử lý Input: Padding & Combine
        dynamic_vehicles = self._pad_and_combine_vehicles(dynamic_trucks, dynamic_drones)
        
        # 2. Tạo Pairwise Embedding [B, H, V, N]
        pairwise_embeds = self.pairwise_encoder(static, dynamic_vehicles)
        
        # 3. Chuẩn bị LSTM Input
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1)
        
        decoder_input_embed = self.coords_embedding(decoder_input)
        
        if last_hh is None:
            last_hh = (self.h0.expand(batch_size, -1), self.c0.expand(batch_size, -1))
            
        # 4. Giải mã Hierarchical
        # Lưu ý: Hàm này trả thêm selected_veh_idx vì nó được chọn nội bộ
        veh_probs, node_probs, selected_veh_idx, last_hh = self.decoder(
            pairwise_embeds, 
            decoder_input_embed, 
            last_hh, 
            mask_vehicles, 
            mask_customers,
            deterministic
        )
        
        # Trả về dạng (Veh_Probs, Node_Probs, Last_HH) như cũ
        # Nhưng LƯU Ý: Trong vòng lặp training PPO, bạn nên sử dụng `selected_veh_idx` 
        # được trả về từ model này thay vì sample lại bên ngoài (để đồng bộ).
        # Tuy nhiên, để khớp API cũ, ta trả về các biến chính.
        
        # Hack nhẹ: Gắn selected_veh_idx vào tuple trả về hoặc xử lý ở PPOTrainer
        # Ở đây tôi trả về thêm 1 biến thứ 4, bạn chỉ cần sửa dòng gọi hàm trong PPOTrainer là:
        # veh_probs, node_probs, last_hh, internal_veh_idx = model(...)
        
        return veh_probs, node_probs, selected_veh_idx, last_hh

    # ======================================================================
    # NEW METHOD ADDED: PERTURB WEIGHTS (Chỉ dùng khi Test/Debug)
    # ======================================================================
    def perturb_weights(self, noise_scale=1.0):
        """
        Thêm nhiễu mạnh để phá vỡ thế kẹt (Local Optima) cho kiến trúc Hierarchical.
        """
        print(f"⚡ [Hierarchical_Actor] Adding STRONG noise (scale={noise_scale})...")
        with torch.no_grad():
            # 1. Nhiễu Encoder (Pairwise Conv2d)
            # Thay thế hoàn toàn trọng số bằng phân phối Uniform rộng (Reset mạnh)
            if hasattr(self.pairwise_encoder, 'conv2d'):
                self.pairwise_encoder.conv2d.weight.data.uniform_(-noise_scale, noise_scale)
                if self.pairwise_encoder.conv2d.bias is not None:
                     self.pairwise_encoder.conv2d.bias.data.uniform_(-noise_scale, noise_scale)

            # 2. Nhiễu Decoder (Hierarchical Steps)
            # Với các lớp Linear (W), ta cộng thêm nhiễu (Additive Noise) thay vì thay thế
            # để giữ lại một phần kiến thức đã học nhưng làm rung chuyển nó.
            
            # --- Nhánh chọn Vehicle ---
            self.decoder.W_veh.weight.data += torch.randn_like(self.decoder.W_veh.weight.data) * noise_scale
            self.decoder.v_veh.data.normal_(0, noise_scale * 2) # Vector v reset mạnh
            
            # --- Nhánh chọn Customer ---
            self.decoder.W_cust.weight.data += torch.randn_like(self.decoder.W_cust.weight.data) * noise_scale
            self.decoder.v_cust.data.normal_(0, noise_scale * 2) # Vector v reset mạnh
            
        print("✓ Hierarchical Weights perturbed successfully.")

    def _init_weights_high_variance(self):
        """
        Khởi tạo trọng số với phương sai lớn để phá vỡ tính đối xứng ban đầu.
        Giúp model không bị tình trạng chọn tất cả các xe/khách với xác suất ngang nhau (50/50).
        """
        scale_factor = 2.0  # Std lớn để tạo logit lớn -> Softmax nhọn (Sharp)
        
        with torch.no_grad():
            # 1. Các ma trận chiếu (Linear Projection - W)
            # Giữ Xavier để đảm bảo luồng gradient ổn định qua tanh()
            nn.init.xavier_uniform_(self.decoder.W_veh.weight)
            nn.init.xavier_uniform_(self.decoder.W_cust.weight)
            
            # Nếu có bias thì đưa về 0
            if self.decoder.W_veh.bias is not None: nn.init.zeros_(self.decoder.W_veh.bias)
            if self.decoder.W_cust.bias is not None: nn.init.zeros_(self.decoder.W_cust.bias)

            # 2. Các vector năng lượng (Scoring Vectors - v)
            # Dùng Normal distribution với độ lệch chuẩn LỚN
            # Điều này khiến điểm Energy ban đầu dao động mạnh, giúp mô hình
            # "dám" đưa ra quyết định dứt khoát ngay từ đầu thay vì ngập ngừng.
            nn.init.normal_(self.decoder.v_veh, mean=0.0, std=scale_factor)
            nn.init.normal_(self.decoder.v_cust, mean=0.0, std=scale_factor)
            
            # 3. Pairwise Encoder
            # Khởi tạo Kaiming cho Conv2d (tốt cho ReLU/Non-linearity sau đó)
            nn.init.kaiming_normal_(self.pairwise_encoder.conv2d.weight, mode='fan_out', nonlinearity='relu')
        
        print(f"⚡ Weights initialized with High Variance (std={scale_factor}) to force random bias.")

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
        self.dropout = nn.Dropout(0.1)
        
    # def forward(self, static, dynamic_trucks, dynamic_drones):
    #     static_embed = self.static_conv(static)
    #     truck_embed = self.truck_conv(dynamic_trucks)
    #     drone_embed = self.drone_conv(dynamic_drones)
    #     combined = torch.cat([static_embed.mean(2), truck_embed.mean(2), drone_embed.mean(2)], dim=1)
    #     x = self.relu(self.fc1(combined))
    #     x = self.dropout(x)
    #     x = self.relu(self.fc2(x))
    #     x = self.dropout(x)
    #     return self.fc3(x).squeeze(-1)

    def forward(self, static, dynamic_trucks, dynamic_drones):
        # 1. Thêm ReLU cho phần Embedding để trích xuất tính chất phi tuyến
        static_embed = self.relu(self.static_conv(static))
        truck_embed = self.relu(self.truck_conv(dynamic_trucks))
        drone_embed = self.relu(self.drone_conv(dynamic_drones))
        
        # 2. Global Average Pooling (Giữ nguyên logic của bạn)
        combined = torch.cat([
            static_embed.mean(2), 
            truck_embed.mean(2), 
            drone_embed.mean(2)
        ], dim=1)
        
        # 3. Các lớp Fully Connected
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 4. Lớp Output: Tuyệt đối không có Activation
        # Squeeze dim=1 để đảm bảo luôn giữ lại dim của Batch
        return self.fc3(x).squeeze(1)


# def check_model_compatibility():
#     print("\n🚀 STARTING COMPATIBILITY CHECK...")
    
#     # 1. Setup
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     BATCH_SIZE = 4
#     HIDDEN_SIZE = 128
    
#     # Kích thước đặc trưng theo DataLoader
#     STATIC_SIZE = 4       # x, y, demand, type
#     DYN_TRUCK_SIZE = 2    # loc, time
#     DYN_DRONE_SIZE = 4    # loc, time, energy, payload
    
#     # 2. Init Model
#     print(f"🔹 Initializing Model on {DEVICE}...")
#     model = MOPVRP_Actor(
#         static_size=STATIC_SIZE,
#         dynamic_size_truck=DYN_TRUCK_SIZE,
#         dynamic_size_drone=DYN_DRONE_SIZE,
#         hidden_size=HIDDEN_SIZE
#     ).to(DEVICE)



#     # =================================================================
#     # QUAN TRỌNG: GỌI HÀM LÀM NHIỄU Ở ĐÂY
#     # =================================================================
#     try:
#         checkpoint_path = "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net/checkpoints/checkpoint_epoch_497.pth"  # Đường dẫn checkpoint nếu có
    
#         checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
#         model.load_state_dict(checkpoint["actor_state_dict"])
#         # Gọi hàm perturb_weights với noise lớn để thấy rõ sự khác biệt
#         model.perturb_weights(noise_scale=1.0)
#         model._init_weights_high_variance()
#     except:
#         print("⚠️ Warning: Model chưa có hàm perturb_weights. Hãy cập nhật class MOPVRP_Actor trước.")
#     # =================================================================
    
#     # 3. Init DataLoader
#     print("🔹 Initializing DataLoader...")
#     # Giả sử bạn đã định nghĩa class MOPVRPGenerator ở trên
#     dataloader = get_rl_dataloader(batch_size=BATCH_SIZE, device=DEVICE)
#     data_iter = iter(dataloader)
    
#     # 4. Run Test
#     try:
#         # Lấy 1 batch
#         print("🔹 Fetching Batch data...")
#         static, dyn_trucks, dyn_drones, mask_cust, mask_veh, scale, weights = next(data_iter)
        
#         batch_size, _, num_nodes = static.shape
#         num_trucks = dyn_trucks.shape[2]
#         num_drones = dyn_drones.shape[2]
        
#         print(f"   Input Shapes:")
#         print(f"   - Static: {static.shape}")
#         print(f"   - Trucks: {dyn_trucks.shape}")
#         print(f"   - Drones: {dyn_drones.shape}")

#         # --- CAN THIỆP THỦ CÔNG ĐỂ TEST (Nuclear Option) ---
#         print("\n☢️  MANUALLY HACKING WEIGHTS TO FORCE SKEW...")
#         with torch.no_grad():
#             # 1. Ép xe đầu tiên (Index 0) có điểm số cực cao
#             # model.pointer.v_veh shape: (1, 1, hidden)
#             # Ta cộng một số rất lớn vào phần tử đầu tiên của vector v
#             model.pointer.v_veh.data.fill_(10.0) # Tăng độ lớn vector v lên
            
#             # Ép bias của xe đầu tiên trong lớp Linear W_veh (nếu có)
#             # Nhưng ở đây W_veh không có bias, ta hack vào input trucks
#             # Thay vào đó, ta hack trực tiếp vào decoder output của xe
            
#             # Cách hiệu quả nhất: Hack vào lớp Conv1d của Truck Encoder
#             # Làm cho đặc trưng của Truck 0 cực kỳ khác biệt so với các xe khác
#             # Truck Encoder weights: (hidden, input_size, 1)
#             model.truck_encoder.conv.weight.data.normal_(0, 5.0) 
#             model.drone_encoder.conv.weight.data.normal_(0, 2.0) # Drone nhỏ xíu
            
#             # Hack vào v_node để làm lệch Node Probs
#             model.pointer.v_node.data.normal_(0, 5.0)
            
#         print("✅ Weights hacked successfully.")
#         # ---------------------------------------------------

        
#         # Forward Pass
#         print("🔹 Running Forward Pass...")
#         veh_probs, node_probs, last_hh = model(
#             static, dyn_trucks, dyn_drones, 
#             decoder_input=None, 
#             last_hh=None, 
#             mask_customers=mask_cust, 
#             mask_vehicles=mask_veh
#         )
        
#         print("✅ Forward Pass Successful!")
#         print(f"   Output Shapes:")
#         print(f"   - Vehicle Probs: {veh_probs.shape} (Expected: [{batch_size}, {num_trucks + num_drones}])")
#         print(f"   - Node Probs:    {node_probs.shape} (Expected: [{batch_size}, {num_nodes}])")

#         print(f"   Output Explicit Probability:")
#         print(f"   - Vehicle Probs: {veh_probs.detach().cpu().numpy()}")
#         print(f"   - Node Probs:    {node_probs.detach().cpu().numpy()}")
        
#         # Kiểm tra tổng xác suất = 1
#         print(f"   - Sum Vehicle Probs: {veh_probs.sum(dim=1).detach().cpu().numpy()}")
#         print(f"   - Sum Node Probs:    {node_probs.sum(dim=1).detach().cpu().numpy()}")
        
#     except Exception as e:
#         print(f"\n❌ FAILED! Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     check_model_compatibility()

import torch
import torch.nn.functional as F

def check_model_compatibility():
    print("\n🚀 STARTING HIERARCHICAL MODEL COMPATIBILITY CHECK...")
    
    # 1. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    HIDDEN_SIZE = 128
    
    # Kích thước đặc trưng giả định (Mô phỏng dữ liệu thật của bạn)
    STATIC_SIZE = 4       # x, y, demand, type
    DYN_TRUCK_SIZE = 2    # loc, load (Ít feature hơn)
    DYN_DRONE_SIZE = 4    # loc, energy, payload, time (Nhiều feature hơn)
    
    # Kích thước Dynamic đầu vào cho Model phải là MAX của 2 loại xe
    # Vì chúng ta sẽ padding thằng nhỏ lên bằng thằng lớn
    MAX_DYN_SIZE = max(DYN_TRUCK_SIZE, DYN_DRONE_SIZE)

    # 2. Init Hierarchical Model
    print(f"🔹 Initializing MOPVRP_HierarchicalActor on {DEVICE}...")
    # Lưu ý: Class mới chỉ cần 3 tham số này
    model = MOPVRP_Actor(
        static_size=STATIC_SIZE,
        dynamic_size_truck=DYN_TRUCK_SIZE, 
        dynamic_size_drone=DYN_DRONE_SIZE,
        hidden_size=HIDDEN_SIZE
    ).to(DEVICE)

    # =================================================================
    # PHẦN 3: KIỂM TRA TÍNH NĂNG NHIỄU (PERTURBATION CHECK)
    # =================================================================
    try:
        # Đường dẫn checkpoint (Giữ nguyên của bạn)
        checkpoint_path = "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net/checkpoints/checkpoint_epoch_497.pth" 
        
        # Thử load (nếu file tồn tại)
        import os
        if os.path.exists(checkpoint_path):
            print(f"🔹 Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            # Lưu ý: Key state_dict có thể khác nếu bạn đổi tên class, cần check kỹ
            # model.load_state_dict(checkpoint["actor_state_dict"], strict=False) 
        else:
            print("⚠️ Checkpoint file not found. Using random weights.")

        # Gọi hàm perturb_weights mới
        print("⚡ Testing perturb_weights()...")
        model.perturb_weights(noise_scale=5.0)
        model._init_weights_high_variance()
        
    except Exception as e:
        print(f"⚠️ Warning during perturbation: {e}")
    # =================================================================
    
    # 4. Tạo Dữ liệu Giả lập (Dummy Data) 
    # (Tôi tạo trực tiếp để bạn chạy được ngay mà không cần Dataloader)
    print("🔹 Generating Dummy Data with different dimensions...")
    
    NUM_NODES = 20
    NUM_TRUCKS = 2
    NUM_DRONES = 3
    
    # Static: [B, 4, N]
    static = torch.rand(BATCH_SIZE, STATIC_SIZE, NUM_NODES).to(DEVICE)
    
    # Dynamic Truck: [B, 2, T] (Ít chiều)
    dyn_trucks_raw = torch.rand(BATCH_SIZE, DYN_TRUCK_SIZE, NUM_TRUCKS).to(DEVICE)
    
    # Dynamic Drone: [B, 4, D] (Nhiều chiều)
    dyn_drones_raw = torch.rand(BATCH_SIZE, DYN_DRONE_SIZE, NUM_DRONES).to(DEVICE)
    
    mask_cust = torch.ones(BATCH_SIZE, NUM_NODES).to(DEVICE)
    mask_veh = torch.ones(BATCH_SIZE, NUM_TRUCKS + NUM_DRONES).to(DEVICE)

    # 3. Init DataLoader
    print("🔹 Initializing DataLoader...")
    # Giả sử bạn đã định nghĩa class MOPVRPGenerator ở trên
    dataloader = get_rl_dataloader(batch_size=BATCH_SIZE, device=DEVICE)
    data_iter = iter(dataloader)

    print("🔹 Fetching Batch data...")
    static, dyn_trucks, dyn_drones, mask_cust, mask_veh, scale, weights = next(data_iter)

    batch_size, _, num_nodes = static.shape
    num_trucks = dyn_trucks.shape[2]
    num_drones = dyn_drones.shape[2]

    # print(f"   Input Shapes:")
    # print(f"   - Static: {static.shape}")
    # print(f"   - Trucks: {dyn_trucks.shape}")
    # print(f"   - Drones: {dyn_drones.shape}")
    

    # =================================================================
    # PHẦN 5: PADDING LOGIC (QUAN TRỌNG)
    # =================================================================
    print(f"🔹 Processing Dynamic Features (Padding)...")
    print(f"   Original Truck Shape: {dyn_trucks.shape}")
    print(f"   Original Drone Shape: {dyn_drones.shape}")
    
    def pad_feature_dim(tensor, target_dim):
        """Hàm padding feature dimension (dim 1) cho bằng target_dim"""
        b, f, n = tensor.size()
        diff = target_dim - f
        if diff > 0:
            # Tạo tensor 0 có kích thước [B, diff, N]
            padding = torch.zeros(b, diff, n, device=tensor.device)
            # Nối vào đuôi feature
            return torch.cat([tensor, padding], dim=1)
        return tensor

    # Pad cả 2 loại xe để đảm bảo cùng số feature = MAX_DYN_SIZE
    dyn_trucks_padded = pad_feature_dim(dyn_trucks, MAX_DYN_SIZE)
    dyn_drones_padded = pad_feature_dim(dyn_drones, MAX_DYN_SIZE)
    
    print(f"   -> Padded Truck Shape: {dyn_trucks_padded.shape}")
    print(f"   -> Padded Drone Shape: {dyn_drones_padded.shape}")

    # =================================================================
    # PHẦN 6: MANUAL HACKING (Update cho Hierarchical Model)
    # =================================================================
    # print("\n☢️  MANUALLY HACKING WEIGHTS (HIERARCHICAL VERSION)...")
    # with torch.no_grad():
    #     # 1. Hack vào Pairwise Embedding (Conv2d)
    #     # Làm cho đặc trưng của cặp (Xe 0, Khách hàng) cực mạnh
    #     # Pairwise Encoder: self.pairwise_encoder.conv2d
    #     print("   -> Hacking Pairwise Conv2d...")
    #     model.pairwise_encoder.conv2d.weight.data.normal_(0, 5.0) 
        
    #     # 2. Hack vào nhánh chọn Vehicle (W_veh, v_veh)
    #     # Ép model cực kỳ thiên vị khi chọn xe
    #     print("   -> Hacking Vehicle Selection Branch...")
    #     model.decoder.v_veh.data.fill_(10.0) # Tăng độ lớn vector chấm điểm xe
    #     model.decoder.W_veh.weight.data.normal_(0, 5.0)

    #     # 3. Hack vào nhánh chọn Customer (W_cust, v_cust)
    #     print("   -> Hacking Customer Selection Branch...")
    #     model.decoder.v_cust.data.fill_(10.0) # Tăng độ lớn vector chấm điểm khách
    #     model.decoder.W_cust.weight.data.normal_(0, 5.0)
        
    # print("✅ Weights hacked successfully.")

    # =================================================================
    # PHẦN 7: FORWARD PASS
    # =================================================================
    try:
        print("\n🔹 Running Forward Pass...")
        # Lấy 1 batch

        
        # Lưu ý: Truyền vào tensor ĐÃ ĐƯỢC PADDING
        veh_probs, node_probs, idx, last_hh = model(
            static, 
            dyn_trucks_padded, 
            dyn_drones_padded, 
            decoder_input=None, 
            last_hh=None, 
            mask_customers=mask_cust, 
            mask_vehicles=mask_veh
        )
        
        print("✅ Forward Pass Successful!")
        print(f"\n📊 OUTPUT ANALYSIS:")
        
        # Check Shape
        expected_veh = NUM_TRUCKS + NUM_DRONES
        print(f"   - Vehicle Probs Shape: {veh_probs.shape} (Expected: [{BATCH_SIZE}, {expected_veh}])")
        print(f"   - Node Probs Shape:    {node_probs.shape} (Expected: [{BATCH_SIZE}, {NUM_NODES}])")
        print(f"   - Selected Index Shape: {idx.shape}")

        # Check Values
        print(f"\n   Example Probs (Batch 0):")
        print(f"   - Vehicle Probs: {veh_probs[0].detach().cpu().numpy().round(3)}")
        print(f"   - Node Probs:    {node_probs[0].detach().cpu().numpy().round(3)}")
        
        # Check Sum = 1
        sum_veh = veh_probs.sum(dim=1).detach().cpu().numpy()
        sum_node = node_probs.sum(dim=1).detach().cpu().numpy()
        print(f"\n   Probability Integrity Check (Should be all 1.0):")
        print(f"   - Sum Veh:  {sum_veh}")
        print(f"   - Sum Node: {sum_node}")
        
    except Exception as e:
        print(f"\n❌ FAILED! Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_compatibility()