import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import os
import json
from datetime import datetime
from tqdm import tqdm 

from config import SystemConfig
from environment import MOPVRPEnvironment
from model import MOPVRP_Actor
from dataloader import get_rl_dataloader
from visualizer import visualize_mopvrp

class RolloutBuffer:
    """Buffer để lưu trữ experience cho PPO (Giữ nguyên)"""
    def __init__(self):
        self.states = []
        self.actions_veh = []
        self.actions_node = []
        self.logprobs_veh = []
        self.logprobs_node = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def clear(self):
        self.states = []
        self.actions_veh = []
        self.actions_node = []
        self.logprobs_veh = []
        self.logprobs_node = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def __len__(self):
        return len(self.rewards)

    def pretty_print(self):
        print("RolloutBuffer Contents:")
        print(f"  States: {len(self.states)}")
        print(f"  Actions (Vehicle): {len(self.actions_veh)}")
        print(f"  Actions (Node): {len(self.actions_node)}")
        print(f"  LogProbs (Vehicle): {len(self.logprobs_veh)}")
        print(f"  LogProbs (Node): {len(self.logprobs_node)}")
        print(f"  Rewards: {len(self.rewards)}")
        print(f"  Dones: {len(self.dones)}")
        print(f"  Values: {len(self.values)}")

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
        
    def forward(self, static, dynamic_trucks, dynamic_drones):
        static_embed = self.static_conv(static)
        truck_embed = self.truck_conv(dynamic_trucks)
        drone_embed = self.drone_conv(dynamic_drones)
        combined = torch.cat([static_embed.mean(2), truck_embed.mean(2), drone_embed.mean(2)], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)

class PPOConfig:
    """Configuration CHỈ DÀNH CHO TRAINING (Hyperparameters)"""
    def __init__(self):
        # Environment settings (Training related)
        self.batch_size = 128
        self.max_steps = 200
        
        # Model Architecture
        self.hidden_size = 128
        self.num_layers = 1
        self.dropout = 0.2
        
        # PPO Hyperparameters
        self.lr_actor = 1e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Training loop
        self.num_epochs = 1000
        self.update_epochs = 4
        self.rollout_steps = 2048
        
        # Logging
        self.log_interval = 1
        self.save_interval = 1
        self.eval_interval = 1
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
    

class PPOTrainer:
    def __init__(self, ppo_config, sys_config): 
        self.config = ppo_config
        self.sys_config = sys_config 
        self.device = ppo_config.device
        
        os.makedirs(ppo_config.checkpoint_dir, exist_ok=True)
        os.makedirs(ppo_config.log_dir, exist_ok=True)
        
        self._init_networks()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ppo_config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=ppo_config.lr_critic)
        
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=0.95)
        
        # Initialize DataLoader
        self.dataloader = get_rl_dataloader(
            batch_size=ppo_config.batch_size, 
            device=self.device
        )
        
        self.env = MOPVRPEnvironment(self.sys_config, self.dataloader, device=self.device)
        
        self.buffer = RolloutBuffer()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{ppo_config.log_dir}/ppo_{timestamp}")
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_reward = float('-inf')
        self.total_steps = 0
        self.num_updates = 0
        
    def _init_networks(self):
        self.actor = MOPVRP_Actor(4, 2, 4, self.config.hidden_size, self.config.num_layers, self.config.dropout).to(self.device)
        self.critic = Critic(4, 2, 4, self.config.hidden_size).to(self.device)
        self._init_weights(self.actor)
        self._init_weights(self.critic)
        
    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    
    def select_action(self, state, last_hh=None, deterministic=False):
        """
        Select action using current policy
        Returns: vehicle_idx, node_idx, logprob_veh, logprob_node, last_hh
        """
        static, dyn_truck, dyn_drone, mask_cust, mask_veh = state
        
        if mask_cust.sum(dim=1).eq(0).any():
            # Clone để không ảnh hưởng đến state gốc
            mask_cust = mask_cust.clone()
            # Tìm các dòng có tổng = 0
            zero_mask_indices = mask_cust.sum(dim=1) == 0
            # Mở Node 0 (Depot) cho các dòng đó
            mask_cust[zero_mask_indices, 0] = 1

        with torch.no_grad():
            # Get probabilities from actor
            veh_probs, node_probs, last_hh = self.actor(
                static, dyn_truck, dyn_drone,
                decoder_input=None,
                last_hh=last_hh,
                mask_customers=mask_cust,
                mask_vehicles=mask_veh
            )
        
        if torch.isnan(node_probs).any() or (node_probs.sum(dim=1) == 0).any():
            # Tạo một phân phối mặc định: 100% về Depot (Node 0)
            fallback_probs = torch.zeros_like(node_probs)
            fallback_probs[:, 0] = 1.0
            
            # Tìm các dòng bị lỗi (NaN hoặc Sum=0)
            invalid_rows = torch.isnan(node_probs).any(dim=1) | (node_probs.sum(dim=1) == 0)
            
            # Gán đè phân phối mặc định vào các dòng lỗi
            node_probs[invalid_rows] = fallback_probs[invalid_rows]

        # Tương tự cho Vehicle Probs (Phòng hờ)
        if torch.isnan(veh_probs).any() or (veh_probs.sum(dim=1) == 0).any():
            fallback_veh = torch.zeros_like(veh_probs)
            fallback_veh[:, 0] = 1.0 # Chọn xe đầu tiên
            invalid_rows_veh = torch.isnan(veh_probs).any(dim=1) | (veh_probs.sum(dim=1) == 0)
            veh_probs[invalid_rows_veh] = fallback_veh[invalid_rows_veh]

        if deterministic:
            # Greedy selection
            veh_idx = torch.argmax(veh_probs, dim=1)
            node_idx = torch.argmax(node_probs, dim=1)
        else:
            # Stochastic sampling
            veh_dist = torch.distributions.Categorical(veh_probs)
            node_dist = torch.distributions.Categorical(node_probs)
            
            veh_idx = veh_dist.sample()
            node_idx = node_dist.sample()
        
        # Calculate log probabilities
        logprob_veh = torch.log(veh_probs.gather(1, veh_idx.unsqueeze(1)) + 1e-10).squeeze(1)
        logprob_node = torch.log(node_probs.gather(1, node_idx.unsqueeze(1)) + 1e-10).squeeze(1)
        
        return veh_idx, node_idx, logprob_veh, logprob_node, last_hh

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns, advantages, gae = [], [], 0
        rewards, values, dones = np.array(rewards), np.array(values), np.array(dones)
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return np.array(returns), np.array(advantages)

    def collect_rollout(self):
        """Collect rollout data với Progress Bar"""
        self.buffer.clear()
        state = self.env.reset()
        last_hh = None
        
        episode_reward = 0
        episode_length = 0
        
        pbar = tqdm(range(self.config.rollout_steps), desc="🔄 Collecting Rollout", leave=False)
        
        for step in pbar:
            # Select action
            veh_idx, node_idx, logprob_veh, logprob_node, last_hh = self.select_action(state, last_hh)
            
            # --- Logic Valid Mask & Fallback ---
            valid_mask = self.env.get_valid_customer_mask(veh_idx)
            invalid_nodes = (valid_mask.gather(1, node_idx.unsqueeze(1)) == 0).squeeze(1)
            if invalid_nodes.any():
                node_idx = torch.where(invalid_nodes, torch.zeros_like(node_idx), node_idx)
            
            static, dyn_truck, dyn_drone, _, _ = state
            with torch.no_grad(): value = self.critic(static, dyn_truck, dyn_drone)
            
            next_state, reward, done, _ = self.env.step(veh_idx, node_idx)
            
            # Store in buffer
            self.buffer.states.append(state)
            self.buffer.actions_veh.append(veh_idx)
            self.buffer.actions_node.append(node_idx)
            self.buffer.logprobs_veh.append(logprob_veh)
            self.buffer.logprobs_node.append(logprob_node)
            self.buffer.rewards.append(reward.cpu().numpy())
            self.buffer.dones.append(done.cpu().numpy())
            self.buffer.values.append(value.cpu().numpy())
            
            # Stats
            step_reward = reward.mean().item()
            episode_reward += step_reward
            episode_length += 1
            self.total_steps += 1
            
            pbar.set_postfix({
                'reward': f"{step_reward:.2f}", 
                'ep_len': f"{episode_length}"
            })
            
            if done.any():
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                state = self.env.reset()
                last_hh = None
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
            
            if len(self.buffer) >= self.config.rollout_steps:
                break
        
        pbar.close()

    def update_policy(self):
        """Update policy với Progress Bar hiển thị Loss"""
        returns, advantages = self.compute_returns_and_advantages(
            self.buffer.rewards, self.buffer.values, self.buffer.dones
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        pbar = tqdm(range(self.config.update_epochs), desc="🧠 Updating Policy", leave=False)
        
        total_loss_log = 0
        
        for epoch_i in pbar:
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            
            for i in range(len(self.buffer)):
                state = self.buffer.states[i]
                old_logprob_veh = self.buffer.logprobs_veh[i]
                old_logprob_node = self.buffer.logprobs_node[i]
                action_veh = self.buffer.actions_veh[i]
                action_node = self.buffer.actions_node[i]
                
                static, dyn_truck, dyn_drone, mask_cust, mask_veh = state
                
                if mask_cust.sum(dim=1).eq(0).any():
                    mask_cust = mask_cust.clone()
                    mask_cust[mask_cust.sum(dim=1) == 0, 0] = 1

                veh_probs, node_probs, _ = self.actor(
                    static, dyn_truck, dyn_drone, None, None, mask_cust, mask_veh
                )
                
                if torch.isnan(node_probs).any() or (node_probs.sum(dim=1) == 0).any():
                    fallback = torch.zeros_like(node_probs); fallback[:, 0] = 1.0
                    inv = torch.isnan(node_probs).any(dim=1) | (node_probs.sum(dim=1) == 0)
                    node_probs[inv] = fallback[inv]

                # =================================================================
                # 🔍 DEBUG PRINT BLOCK (Chỉ in 1 lần mỗi epoch để tránh spam)
                # =================================================================
                if i == 0 and epoch_i == 0: 
                    print("\n" + "="*60)
                    print(f"🔍 DEBUGGING AT UPDATE STEP (Batch size: {len(mb_idx)})")
                    print("="*60)
                    
                    # Lấy mẫu đầu tiên trong batch để soi
                    sample_idx = 0 
                    veh_id = b_act_veh[sample_idx].item()
                    is_drone = veh_id >= self.env.num_trucks
                    
                    # Lấy thông tin Node Truck-Only
                    # Static: [Batch, 4, N] -> Channel 3 là TruckOnly Flag
                    truck_only_flags = b_static[sample_idx, 3, :].cpu().numpy()
                    truck_only_indices = np.where(truck_only_flags == 1)[0]
                    
                    print(f"🔹 Sample 0 | Selected Vehicle ID: {veh_id} [{'DRONE' if is_drone else 'TRUCK'}]")
                    if is_drone:
                        print(f"   🚫 Truck-Only Nodes (Must be Masked): {truck_only_indices}")
                    
                    # So sánh Mask
                    current_mask = refined_mask[sample_idx].cpu().numpy()
                    print(f"   👺 Applied Mask: {current_mask[:].astype(int)}")
                    
                    # Kiểm tra xem Drone có bị chặn đúng không
                    if is_drone:
                        violation = False
                        for t_idx in truck_only_indices:
                            if current_mask[t_idx] == 1:
                                print(f"   ❌ ALARM: Node {t_idx} is Truck-Only BUT Mask is 1 (Valid)!")
                                violation = True
                        if not violation:
                            print(f"   ✅ Masking OK: All Truck-Only nodes are blocked (0).")

                    # So sánh Xác Suất
                    raw_p = node_probs_raw[sample_idx].detach().cpu().numpy()
                    masked_p = node_probs[sample_idx].detach().cpu().numpy()
                    
                    # Lấy Top 5 xác suất cao nhất
                    top_k = 5
                    top_raw_idx = np.argsort(raw_p)[-top_k:][::-1]
                    top_masked_idx = np.argsort(masked_p)[-top_k:][::-1]
                    
                    print(f"\n   📊 Top {top_k} RAW Probs (Before Mask):")
                    for idx in top_raw_idx:
                        print(f"      Node {idx:2d}: {raw_p[idx]:.4f} {'(TruckOnly)' if idx in truck_only_indices else ''}")
                        
                    print(f"   ✅ Top {top_k} FINAL Probs (After Mask):")
                    for idx in top_masked_idx:
                        val = masked_p[idx]
                        status = "blocked" if val == 0 else "valid"
                        print(f"      Node {idx:2d}: {val:.4f} [{status}]")
                    
                    print("="*60 + "\n")
                # =================================================================
                
                # Tính toán Loss 
                new_logprob_veh = torch.log(veh_probs.gather(1, action_veh.unsqueeze(1)) + 1e-10).squeeze(1)
                new_logprob_node = torch.log(node_probs.gather(1, action_node.unsqueeze(1)) + 1e-10).squeeze(1)
                
                ratio = torch.exp(new_logprob_veh - old_logprob_veh) * torch.exp(new_logprob_node - old_logprob_node)
                adv = advantages[i].expand_as(ratio)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                entropy = -((veh_probs * torch.log(veh_probs + 1e-10)).sum(1) + (node_probs * torch.log(node_probs + 1e-10)).sum(1)).mean()
                
                value_pred = self.critic(static, dyn_truck, dyn_drone)
                critic_loss = nn.MSELoss()(value_pred, returns[i].expand_as(value_pred))
                
                loss = actor_loss - self.config.entropy_coef * entropy + self.config.value_loss_coef * critic_loss
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
            
            avg_act_loss = epoch_actor_loss / len(self.buffer)
            avg_cri_loss = epoch_critic_loss / len(self.buffer)
            pbar.set_postfix({'ActLoss': f"{avg_act_loss:.3f}", 'CriLoss': f"{avg_cri_loss:.3f}"})
            
            total_loss_log += (avg_act_loss + avg_cri_loss)
            
        self.num_updates += 1
        self.writer.add_scalar('Loss/Total', total_loss_log / self.config.update_epochs, self.num_updates)


    def train(self):
        print(f"🚀 Starting PPO Training on {self.device}")
        start_time = time.time()
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.collect_rollout()
            self.update_policy()
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            
            if (epoch + 1) % self.config.log_interval == 0:
                avg_r = np.mean(self.episode_rewards) if self.episode_rewards else 0
                print(f"Epoch {epoch+1}: Avg Reward {avg_r:.4f} | Time: {time.time()-epoch_start:.2f}s")
                self.writer.add_scalar('Reward/Average', avg_r, epoch)
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1)

    def evaluate(self, num_episodes=10):
        """Evaluate current policy"""
        self.actor.eval()
        self.critic.eval()
        
        eval_rewards = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                state = self.env.reset()
                last_hh = None
                episode_reward = 0
                done_flag = False
                
                for _ in range(self.config.max_steps):
                    if done_flag:
                        break
                    
                    # Select action (deterministic)
                    veh_idx, node_idx, _, _, last_hh = self.select_action(
                        state, last_hh, deterministic=True
                    )
                    
                    # Validate action
                    static, _, _, _, _ = state
                    valid_mask = self.env.get_valid_customer_mask(veh_idx)
                    invalid = (valid_mask.gather(1, node_idx.unsqueeze(1)) == 0).squeeze(1)
                    if invalid.any():
                        node_idx = torch.where(invalid, torch.zeros_like(node_idx), node_idx)
                    
                    # Step
                    state, reward, done, _ = self.env.step(veh_idx, node_idx)
                    episode_reward += reward.mean().item()
                    
                    if done.any():
                        done_flag = True
                
                eval_rewards.append(episode_reward)
        
        self.actor.train()
        self.critic.train()
        
        return np.mean(eval_rewards)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        # Create a serializable config dict
        config_dict = {
            'batch_size': self.config.batch_size,
            'max_steps': self.config.max_steps,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'lr_actor': self.config.lr_actor,
            'lr_critic': self.config.lr_critic,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'clip_epsilon': self.config.clip_epsilon,
            'entropy_coef': self.config.entropy_coef,
            'value_loss_coef': self.config.value_loss_coef,
            'max_grad_norm': self.config.max_grad_norm,
            'drone_speed': self.sys_config.drone_speed,
            'drone_max_energy': self.sys_config.drone_max_energy,
            't_takeoff': self.sys_config.t_takeoff,
            't_landing': self.sys_config.t_landing,
            'drone_params': self.sys_config.drone_params,
        }
        
        checkpoint = {
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'best_reward': self.best_reward,
            'config': config_dict  # Save as dict instead of object
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.best_reward = checkpoint['best_reward']
        
        print(f"✅ Checkpoint loaded from {path}")
        return checkpoint['epoch']


def main():
    sys_config = SystemConfig('Truck_config.json', 'drone_linear_config.json', drone_type="1")
    
    ppo_config = PPOConfig()
    
    trainer = PPOTrainer(ppo_config, sys_config)
    
    trainer.train()

if __name__ == "__main__":
    main()