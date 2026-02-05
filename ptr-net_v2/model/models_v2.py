import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# SIMPLIFIED PAIRWISE ENCODER
# ============================================================================

class SimplifiedPairwiseEncoder(nn.Module):
    """
    Lightweight pairwise encoder
    No self-attention - just project & combine
    """
    def __init__(self, static_size, dynamic_size, hidden_size):
        super().__init__()
        # Project inputs to hidden_size
        self.static_conv = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dynamic_conv = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)
        
        # Combine projector
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, static, dynamic):
        """
        static: [B, H, N] - already projected
        dynamic: [B, H, V] - already projected
        Returns: [B, H, V, N] pairwise embeddings
        """
        B, H, N = static.size()
        _, _, V = dynamic.size()
        
        # Expand to create pairwise combinations
        s_exp = static.unsqueeze(2).expand(-1, -1, V, -1)  # [B, H, V, N]
        d_exp = dynamic.unsqueeze(3).expand(-1, -1, -1, N)  # [B, H, V, N]
        
        # Concat along feature dimension
        combined = torch.cat([s_exp, d_exp], dim=1)  # [B, 2H, V, N]
        combined = combined.permute(0, 2, 3, 1)  # [B, V, N, 2H]
        
        # Project to hidden_size
        out = self.combine(combined)  # [B, V, N, H]
        out = out.permute(0, 3, 1, 2)  # [B, H, V, N]
        
        return out


# ============================================================================
# SIMPLIFIED MEMORY
# ============================================================================

class SimpleMemory(nn.Module):
    """
    Lightweight memory: GRU + pooling
    No Transformer complexity
    """
    def __init__(self, hidden_size, mem_window=8):
        super().__init__()
        self.mem_window = mem_window
        
        # GRU for memory processing
        self.mem_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Readout projection
        self.mem_readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
    def forward(self, h_t, h_mem=None):
        """
        h_t: [B, H] - current hidden state
        h_mem: [B, W, H] - memory window
        Returns: mem_context [B, H], h_mem_new [B, W, H]
        """
        B, H = h_t.size()
        device = h_t.device
        
        if h_mem is None:
            h_mem = torch.zeros(B, self.mem_window, H, device=device)
        
        # Update memory window (FIFO queue)
        h_mem_new = torch.cat([
            h_mem[:, 1:, :],  # Drop oldest
            h_t.unsqueeze(1)  # Add newest
        ], dim=1)  # [B, W, H]
        
        # Process memory with GRU
        mem_out, _ = self.mem_gru(h_mem_new)  # [B, W, H]
        
        # Pool and project
        mem_pooled = mem_out.mean(dim=1)  # [B, H]
        mem_context = self.mem_readout(mem_pooled)  # [B, H]
        
        return mem_context, h_mem_new


# ============================================================================
# SIMPLIFIED DECODER
# ============================================================================

class SimplifiedDecoder(nn.Module):
    """
    Hierarchical decoder with standard attention
    No multi-head additive - use PyTorch's optimized MHA
    """
    def __init__(self, hidden_size, nhead=4, mem_window=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM cell
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # Memory
        self.memory = SimpleMemory(hidden_size, mem_window)
        
        # Attention layers
        self.veh_attention = nn.MultiheadAttention(
            hidden_size, nhead, batch_first=True
        )
        self.cust_attention = nn.MultiheadAttention(
            hidden_size, nhead, batch_first=True
        )
        
        # Layer norms
        self.ln_veh = nn.LayerNorm(hidden_size)
        self.ln_cust = nn.LayerNorm(hidden_size)
        
    def forward(self, pairwise_embeds, decoder_input, last_hh, h_mem,
                mask_vehicles=None, mask_pairwise=None,
                deterministic=False, forced_action=None):
        """
        pairwise_embeds: [B, H, V, N]
        decoder_input: [B, H]
        last_hh: (h, c) each [B, H]
        h_mem: [B, W, H]
        """
        B, H, V, N = pairwise_embeds.size()
        device = pairwise_embeds.device
        
        # === 1. LSTM STEP ===
        h_t, c_t = self.lstm(decoder_input, last_hh)
        
        # === 2. MEMORY UPDATE ===
        mem_context, h_mem_new = self.memory(h_t, h_mem)
        
        # Enrich h_t with memory
        h_t_enriched = h_t + mem_context  # [B, H]
        
        # === 3. VEHICLE SELECTION ===
        # Pool over customers to get vehicle representations
        veh_repr = pairwise_embeds.mean(dim=3)  # [B, H, V]
        veh_repr = veh_repr.transpose(1, 2)  # [B, V, H]
        
        # Query from enriched hidden state
        q = h_t_enriched.unsqueeze(1)  # [B, 1, H]
        
        # Prepare mask
        veh_key_mask = None
        if mask_vehicles is not None:
            veh_key_mask = (mask_vehicles == 0)  # True = masked out
        
        # Cross-attention
        veh_out, veh_attn_weights = self.veh_attention(
            q, veh_repr, veh_repr,
            key_padding_mask=veh_key_mask
        )
        veh_out = self.ln_veh(veh_out)  # [B, 1, H]
        
        # Extract probabilities from attention weights
        veh_probs = veh_attn_weights.squeeze(1)  # [B, V]
        
        # Apply mask explicitly (safety)
        if mask_vehicles is not None:
            veh_probs = masked_softmax(
                torch.log(veh_probs + 1e-8), 
                mask_vehicles, 
                dim=-1
            )
        
        # Sample vehicle
        if forced_action is not None:
            veh_idx = forced_action[0]
        else:
            if deterministic:
                veh_idx = torch.argmax(veh_probs, dim=1)
            else:
                veh_idx = torch.distributions.Categorical(veh_probs).sample()
        
        # === 4. CUSTOMER SELECTION ===
        # Extract embeddings for selected vehicle
        idx_view = veh_idx.view(B, 1, 1, 1).expand(-1, H, 1, N)
        selected_embeds = pairwise_embeds.gather(2, idx_view).squeeze(2)  # [B, H, N]
        selected_embeds = selected_embeds.transpose(1, 2)  # [B, N, H]
        
        # Extract mask for selected vehicle
        cust_key_mask = None
        if mask_pairwise is not None:
            idx_mask = veh_idx.view(B, 1, 1).expand(-1, 1, N)
            cust_mask = mask_pairwise.gather(1, idx_mask).squeeze(1)  # [B, N]
            cust_key_mask = (cust_mask == 0)
        
        # Cross-attention
        cust_out, cust_attn_weights = self.cust_attention(
            veh_out, selected_embeds, selected_embeds,
            key_padding_mask=cust_key_mask
        )
        cust_out = self.ln_cust(cust_out)  # [B, 1, H]
        
        # Extract probabilities
        cust_probs = cust_attn_weights.squeeze(1)  # [B, N]
        
        # Apply mask
        if mask_pairwise is not None:
            cust_probs = masked_softmax(
                torch.log(cust_probs + 1e-8),
                cust_mask,
                dim=-1
            )
        
        # Sample customer
        if forced_action is not None:
            node_idx = forced_action[1]
        else:
            if deterministic:
                node_idx = torch.argmax(cust_probs, dim=1)
            else:
                node_idx = torch.distributions.Categorical(cust_probs).sample()
        
        # === 5. COMPUTE LOG PROBS & ENTROPY ===
        action_log_prob = None
        entropy = None
        
        if forced_action is not None:
            # Log probabilities
            veh_log = torch.log(veh_probs + 1e-8)
            veh_lp = veh_log.gather(1, veh_idx.unsqueeze(1))
            
            cust_log = torch.log(cust_probs + 1e-8)
            cust_lp = cust_log.gather(1, node_idx.unsqueeze(1))
            
            action_log_prob = veh_lp + cust_lp
            
            # Entropy
            veh_dist = torch.distributions.Categorical(veh_probs)
            cust_dist = torch.distributions.Categorical(cust_probs)
            entropy = veh_dist.entropy().unsqueeze(1) + cust_dist.entropy().unsqueeze(1)
        
        return (veh_probs, cust_probs, veh_idx, (h_t, c_t), 
                h_mem_new, action_log_prob, entropy)


# ============================================================================
# SIMPLIFIED ACTOR
# ============================================================================

class SimplifiedEnhancedActor(nn.Module):
    """
    Main Actor Model
    - Weight Aware conditioning from weights
    - Simplified pairwise encoding
    - Memory-augmented decoder
    """
    def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone,
                 hidden_size, nhead=4, mem_window=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.mem_window = mem_window
        
        # === INPUT PROJECTORS ===
        self.static_proj = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.truck_proj = nn.Conv1d(dynamic_size_truck, hidden_size, kernel_size=1)
        self.drone_proj = nn.Conv1d(dynamic_size_drone, hidden_size, kernel_size=1)
        
        # === WEIGHT ENCODER ===
        self.weight_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # === WEIGHT AWARE GENERATORS ===
        self.wa_static = nn.Linear(hidden_size, 2 * hidden_size)
        self.wa_dynamic = nn.Linear(hidden_size, 2 * hidden_size)
        
        # === PAIRWISE ENCODER ===
        self.pairwise_encoder = SimplifiedPairwiseEncoder(
            hidden_size, hidden_size, hidden_size
        )
        
        # === DECODER ===
        self.decoder = SimplifiedDecoder(hidden_size, nhead, mem_window)
        
        # === COORDINATE EMBEDDING ===
        self.coords_embedding = nn.Linear(2, hidden_size)
        
        # === LEARNABLE INITIAL STATES ===
        self.x0 = nn.Parameter(torch.zeros(1, 2))
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_size))
    
    def forward(self, static, dynamic_trucks, dynamic_drones, weights,
                decoder_input=None, last_hh=None, h_mem=None,
                mask_vehicles=None, mask_pairwise=None,
                deterministic=False, forced_action=None):
        """
        Main forward pass
        
        Inputs:
          static: [B, F_s, N]
          dynamic_trucks: [B, F_t, K]
          dynamic_drones: [B, F_d, D]
          weights: [B, 2]
          
        Returns:
          (veh_probs, cust_probs, veh_idx, hidden, mem, logp, entropy)
        """
        B = static.size(0)
        device = static.device
        
        # === 1. PROJECT INPUTS ===
        s = self.static_proj(static)  # [B, H, N]
        t = self.truck_proj(dynamic_trucks)  # [B, H, K]
        d = self.drone_proj(dynamic_drones)  # [B, H, D]
        
        # === 2. Weight Aware ===
        w_enc = self.weight_encoder(weights)  # [B, H]
        
        gamma_s, beta_s = self.wa_static(w_enc).chunk(2, dim=-1)
        gamma_d, beta_d = self.wa_dynamic(w_enc).chunk(2, dim=-1)
        
        # Modulate features
        s = s * (1 + gamma_s.unsqueeze(2)) + beta_s.unsqueeze(2)
        t = t * (1 + gamma_d.unsqueeze(2)) + beta_d.unsqueeze(2)
        d = d * (1 + gamma_d.unsqueeze(2)) + beta_d.unsqueeze(2)
        
        # === 3. CONCAT VEHICLES ===
        vehicles = torch.cat([t, d], dim=2)  # [B, H, K+D]
        
        # === 4. PAIRWISE ENCODING ===
        pairwise_embeds = self.pairwise_encoder(s, vehicles)  # [B, H, V, N]
        
        # === 5. DECODER SETUP ===
        if decoder_input is None:
            decoder_input = self.x0.expand(B, -1)
        decoder_input_embed = self.coords_embedding(decoder_input)  # [B, H]
        
        if last_hh is None:
            last_hh = (
                self.h0.expand(B, -1),
                self.c0.expand(B, -1)
            )
        
        # === 6. DECODE ===
        results = self.decoder(
            pairwise_embeds, decoder_input_embed, last_hh, h_mem,
            mask_vehicles, mask_pairwise, deterministic, forced_action
        )
        
        return results


# ============================================================================
# IMPROVED DUAL CRITIC
# ============================================================================

class ImprovedDualCritic(nn.Module):
    """
    Dual-head critic with self-attention
    """
    def __init__(self, static_size, dynamic_size_truck, dynamic_size_drone,
                 hidden_size, nhead=4):
        super().__init__()
        
        # === PROJECTORS ===
        self.static_proj = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.truck_proj = nn.Conv1d(dynamic_size_truck, hidden_size, kernel_size=1)
        self.drone_proj = nn.Conv1d(dynamic_size_drone, hidden_size, kernel_size=1)
        
        # === SELF-ATTENTION ===
        self.static_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1
        )
        
        # === WEIGHT ENCODER ===
        self.weight_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU()
        )
        
        # === SHARED PROCESSING ===
        self.shared = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # === DUAL VALUE HEADS ===
        self.v1_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.v2_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, static, dynamic_trucks, dynamic_drones, weights, h_mem=None):
        """
        Compute value estimate
        
        Returns: (value, v1, v2)
        """
        # === 1. PROJECT ===
        s = F.relu(self.static_proj(static))  # [B, H, N]
        t = F.relu(self.truck_proj(dynamic_trucks))  # [B, H, K]
        d = F.relu(self.drone_proj(dynamic_drones))  # [B, H, D]
        
        # === 2. ENCODE STRUCTURE (Key improvement!) ===
        s = s.transpose(1, 2)  # [B, N, H]
        s = self.static_encoder(s)  # Encode global structure
        s_pool = s.mean(dim=1)  # [B, H] - Pool AFTER attention
        
        # === 3. POOL VEHICLES ===
        t_pool = t.mean(dim=2)  # [B, H]
        d_pool = d.mean(dim=2)  # [B, H]
        
        # === 4. ENCODE WEIGHTS ===
        w = self.weight_encoder(weights)  # [B, H]
        
        # === 5. CONCAT ALL ===
        combined = torch.cat([s_pool, t_pool, d_pool, w], dim=1)  # [B, 4H]
        
        # === 6. SHARED PROCESSING ===
        shared = self.shared(combined)  # [B, H]
        
        # === 7. DUAL HEADS ===
        v1 = self.v1_head(shared)  # [B, 1]
        v2 = self.v2_head(shared)  # [B, 1]
        
        # === 8. WEIGHTED COMBINATION ===
        w_norm = _normalize_weights(weights)
        value = w_norm[:, 0:1] * v1 + w_norm[:, 1:2] * v2
        
        return value, v1, v2


# ============================================================================
# MAIN CLASS
# ============================================================================

class MOPVRP(nn.Module):
    def __init__(self, static_size, dyn_truck_size, dyn_drone_size,
                 hidden_size, nhead=4, mem_window=8, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.mem_window = mem_window
        self.num_trucks = None
        
        # === CREATE MODELS ===
        self.actor = SimplifiedEnhancedActor(
            static_size, dyn_truck_size, dyn_drone_size,
            hidden_size, nhead, mem_window
        ).to(device)
        
        self.critic = ImprovedDualCritic(
            static_size, dyn_truck_size, dyn_drone_size,
            hidden_size, nhead
        ).to(device)
    
    # ========================================================================
    # OBSERVATION CONVERSION
    # ========================================================================
    
    def _unpack_obs(self, obs):
        """
        Convert env observation to model format
        
        ENV format:
          graph_ctx: [B, Fg, N+1]
          trucks_ctx: [B, Ft, K]
          drones_ctx: [B, Fd, D]
          mask_trk: [B, K, N+1]
          mask_dr: [B, D, N+1]
          weights: [B, 2]
        
        MODEL format: (same, just extract)
        """
        # Features are already in [B, F, N] format - no transpose needed!
        graph = obs['graph_ctx'].transpose(1, 2) # [B, Fg, N+1]
        trucks = obs['trucks_ctx'].transpose(1, 2)  # [B, Ft, K]
        drones = obs['drones_ctx'].transpose(1, 2)  # [B, Fd, D]
        
        # Track number of trucks for action conversion
        self.num_trucks = trucks.size(2)
        
        # Weights
        weights = obs.get('weights', None)
        if weights is None:
            B = graph.size(0)
            weights = torch.tensor([[0.5, 0.5]], device=graph.device).expand(B, -1)
        
        # Masks
        mask_trk = obs['mask_trk']  # [B, K, N+1]
        mask_dr = obs['mask_dr']    # [B, D, N+1]
        
        # Combine masks
        mask_pairwise = torch.cat([mask_trk, mask_dr], dim=1)  # [B, K+D, N+1]
        
        # Vehicle mask: vehicle is valid if it has at least one feasible node
        mask_vehicles = mask_pairwise.any(dim=2).float()  # [B, K+D]
        
        return graph, trucks, drones, weights, mask_vehicles, mask_pairwise
    
    # ========================================================================
    # HIDDEN STATE MANAGEMENT
    # ========================================================================
    
    def _pack_hidden(self, lstm_tuple, mem_tensor):
        """
        Pack (h, c) + memory into single tensor for buffer
        
        lstm_tuple: (h, c) each [B, H]
        mem_tensor: [B, W, H]
        
        Returns: [B, 2H + W*H] flat tensor
        """
        h, c = lstm_tuple
        
        # Flatten memory
        mem_flat = mem_tensor.reshape(mem_tensor.size(0), -1)  # [B, W*H]
        
        # Concat all
        return torch.cat([h, c, mem_flat], dim=-1)
    
    def _unpack_hidden(self, hidden_tensor):
        """
        Unpack flat tensor into (h, c) + memory
        
        Returns: ((h, c), mem_tensor)
        """
        if hidden_tensor is None:
            return None, None
        
        B = hidden_tensor.size(0)
        H = self.hidden_size
        W = self.mem_window
        
        # Split
        h = hidden_tensor[:, :H]
        c = hidden_tensor[:, H:2*H]
        mem_flat = hidden_tensor[:, 2*H:]
        
        # Reshape memory
        mem_tensor = mem_flat.view(B, W, H)
        
        return (h.contiguous(), c.contiguous()), mem_tensor.contiguous()
    
    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden state
        
        Returns: [B, 2H + W*H] tensor
        """
        if device is None:
            device = self.device
        
        # LSTM states
        h0 = self.actor.h0.expand(batch_size, -1).to(device)
        c0 = self.actor.c0.expand(batch_size, -1).to(device)
        
        # Memory state (zeros)
        mem0 = torch.zeros(batch_size, self.mem_window, self.hidden_size, device=device)
        
        return self._pack_hidden((h0, c0), mem0)
    
    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    
    @torch.no_grad()
    def act(self, obs, h_prev=None, deterministic=False):
        """
        Sample action for rollout
        
        Returns:
          action: (veh_type, veh_inst, node_idx)
          log_prob: [B, 1]
          value: [B, 1]
          info: dict
          h_next: [B, 2H + W*H]
        """
        # 1. Unpack observation
        graph, trucks, drones, weights, mask_veh, mask_pair = self._unpack_obs(obs)
        
        # 2. Unpack hidden state
        last_hh, h_mem = self._unpack_hidden(h_prev)
        
        # 3. Forward actor
        results = self.actor(
            static=graph,
            dynamic_trucks=trucks,
            dynamic_drones=drones,
            weights=weights,
            last_hh=last_hh,
            h_mem=h_mem,
            mask_vehicles=mask_veh,
            mask_pairwise=mask_pair,
            deterministic=deterministic
        )
        
        # Unpack results
        veh_probs, cust_probs, flat_veh_idx, h_next_tuple, h_mem_new, _, _ = results
        
        # 4. Sample node (actor returns vehicle index, we need to sample node)
        if deterministic:
            node_idx = torch.argmax(cust_probs, dim=1)
        else:
            node_idx = torch.distributions.Categorical(cust_probs).sample()
        
        # 5. Forward critic
        value, _, _ = self.critic(graph, trucks, drones, weights, h_mem=h_mem)
        
        # 6. Convert flat vehicle index to hierarchical action
        is_drone = (flat_veh_idx >= self.num_trucks)
        veh_type = is_drone.long()  # 0=truck, 1=drone
        veh_inst = torch.where(is_drone, flat_veh_idx - self.num_trucks, flat_veh_idx)
        
        action_tuple = (veh_type, veh_inst, node_idx)
        
        # 7. Pack hidden state
        h_next_packed = self._pack_hidden(h_next_tuple, h_mem_new)
        
        # Dummy log_prob for inference (not used during rollout)
        log_prob = torch.zeros(graph.size(0), 1, device=self.device)
        
        # Info dict
        info = {
            'veh_probs': veh_probs,
            'cust_probs': cust_probs
        }
        
        return action_tuple, log_prob, value, info, h_next_packed
    
    def evaluate_actions(self, flat_obs, actions, h_prev=None):
        """
        Re-evaluate actions for PPO update
        
        Returns:
          log_prob: [N, 1]
          entropy: [N, 1]
          value: [N, 1]
        """
        # 1. Unpack observation
        graph, trucks, drones, weights, mask_veh, mask_pair = self._unpack_obs(flat_obs)
        
        # 2. Unpack hidden state
        last_hh, h_mem = self._unpack_hidden(h_prev)
        
        # 3. Convert hierarchical actions to flat
        veh_type = actions[:, 0]
        veh_inst = actions[:, 1]
        node_idx = actions[:, 2]
        
        flat_veh_idx = torch.where(
            veh_type == 0,
            veh_inst,
            veh_inst + self.num_trucks
        )
        
        forced_action = (flat_veh_idx, node_idx)
        
        # 4. Forward actor with forced action
        results = self.actor(
            static=graph,
            dynamic_trucks=trucks,
            dynamic_drones=drones,
            weights=weights,
            last_hh=last_hh,
            h_mem=h_mem,
            mask_vehicles=mask_veh,
            mask_pairwise=mask_pair,
            forced_action=forced_action
        )
        
        # Unpack results
        _, _, _, _, _, log_prob, entropy = results
        
        # 5. Forward critic
        value, _, _ = self.critic(graph, trucks, drones, weights, h_mem=h_mem)
        
        return log_prob, entropy, value
    
    @torch.no_grad()
    def get_value(self, obs, h_prev=None):
        """
        Get value estimate only
        """
        graph, trucks, drones, weights, _, _ = self._unpack_obs(obs)
        _, h_mem = self._unpack_hidden(h_prev)
        
        value, _, _ = self.critic(graph, trucks, drones, weights, h_mem=h_mem)
        return value


# ============================================================================
# UTILITIES
# ============================================================================

def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
    """Normalize weights to sum=1"""
    w = torch.clamp(w, min=0)
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return w / denom


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax only over valid entries"""
    if mask is not None:
        # Convert to boolean if needed
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        # Mask out invalid entries
        logits = logits.masked_fill(~mask, -1e9)
    
    return F.softmax(logits, dim=dim)
