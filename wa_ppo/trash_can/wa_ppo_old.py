# """
# Weight Aggregation PPO for Multi-Objective Optimization
# """
# import torch
# import torch.nn as nn
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import BaseCallback
# from tqdm import tqdm
# from typing import Dict
# import gym
# from sb3_contrib import MaskablePPO


# class VRPDFeatureExtractor(BaseFeaturesExtractor):
#     """Custom feature extractor cho VRPD"""
    
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)
        
#         self.customer_encoder = nn.Sequential(
#             nn.Linear(6, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU()
#         )
        
#         self.global_encoder = nn.Sequential(
#             nn.Linear(10, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU()
#         )
        
#         self.attention = nn.MultiheadAttention(
#             embed_dim=128,
#             num_heads=4,
#             batch_first=True
#         )
        
#         self.final_layer = nn.Sequential(
#             nn.Linear(256, features_dim),
#             nn.ReLU()
#         )
        
#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         batch_size = observations.shape[0]
        
#         n_customer_features = observations.shape[1] - 10
#         n_customers = n_customer_features // 6
        
#         customer_features = observations[:, :n_customer_features].reshape(batch_size, n_customers, 6)
#         global_features = observations[:, n_customer_features:]
        
#         customer_encoded = self.customer_encoder(customer_features)
#         attended, _ = self.attention(customer_encoded, customer_encoded, customer_encoded)
#         customer_pooled = torch.mean(attended, dim=1)
        
#         global_encoded = self.global_encoder(global_features)
#         combined = torch.cat([customer_pooled, global_encoded], dim=1)
#         output = self.final_layer(combined)
        
#         return output

# # class MultiObjectiveCallback(BaseCallback):
# #     """Callback để track multi-objective metrics"""
    
# #     def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
# #         super().__init__(verbose)
# #         self.eval_env = eval_env
# #         self.eval_freq = eval_freq
# #         self.best_reward = -np.inf
        
# #     def _on_step(self) -> bool:
# #         if self.n_calls % self.eval_freq == 0:
# #             completion_times = []
# #             waiting_times = []
            
# #             for _ in range(5):
# #                 # obs = self.eval_env.reset()
# #                 obs = self.model._last_obs
# #                 done = False
                
# #                 while not done:
# #                     action, _ = self.model.predict(obs, deterministic=True)
# #                     # obs, reward, done, info = self.eval_env.step(action)
# #                     step_result = self.eval_env.step(action)
# #                     if len(step_result) == 5:
# #                         obs, reward, terminated, truncated, info = step_result
# #                         done = terminated or truncated
# #                     else:
# #                         obs, reward, done, info = step_result

# #                     if terminated or truncated:
# #                         obs, info = self.eval_env.reset()
                
# #                 calc = self.eval_env.calculator
# #                 times = []
# #                 wait = 0
                
# #                 for route in info['truck_routes']:
# #                     if route:
# #                         t, w = calc.calculate_truck_time(route)
# #                         times.append(t)
# #                         wait += w
                
# #                 for route in info['drone_routes']:
# #                     if route:
# #                         t, w, _ = calc.calculate_drone_time(route)
# #                         times.append(t)
# #                         wait += w
                
# #                 completion_times.append(max(times) if times else 0)
# #                 waiting_times.append(wait)
            
# #             avg_completion = np.mean(completion_times)
# #             avg_waiting = np.mean(waiting_times)
            
# #             self.logger.record("eval/completion_time", avg_completion)
# #             self.logger.record("eval/waiting_time", avg_waiting)
# #             self.logger.record("eval/total_objective", avg_completion + avg_waiting)
            
# #             if self.verbose > 0:
# #                 print(f"\nEval @ {self.n_calls}:")
# #                 print(f"  Completion: {avg_completion:.2f}s")
# #                 print(f"  Waiting: {avg_waiting:.2f}s")
        
# #         return True

# class MultiObjectiveCallback(BaseCallback):
#     """Callback để track multi-objective metrics"""
    
#     def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
#         super().__init__(verbose)
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq
#         self.best_reward = -np.inf

#     def _on_step(self) -> bool:
#         # Thực hiện evaluation mỗi eval_freq bước
#         if self.n_calls % self.eval_freq == 0:
#             completion_times = []
#             waiting_times = []

#             for _ in range(5):
#                 # ✅ Reset environment trước mỗi episode eval
#                 obs, info = self.eval_env.reset()
#                 terminated, truncated = False, False

#                 while not (terminated or truncated):
#                     # Predict hành động từ mô hình
#                     action, _ = self.model.predict(obs, deterministic=True)

#                     # Step environment (Gymnasium API: 5 giá trị trả về)
#                     step_result = self.eval_env.step(action)
#                     if len(step_result) == 5:
#                         obs, reward, terminated, truncated, info = step_result
#                     else:
#                         obs, reward, done, info = step_result
#                         terminated, truncated = done, False  # fallback cho env kiểu cũ

#                     # ✅ Reset lại env ngay khi episode kết thúc
#                     if terminated or truncated:
#                         obs, info = self.eval_env.reset()
#                         break  # ra khỏi vòng lặp episode hiện tại

#                 # Sau khi 1 episode eval kết thúc → tính các metric
#                 calc = self.eval_env.calculator
#                 times = []
#                 wait = 0.0

#                 for route in info.get('truck_routes', []):
#                     if route:
#                         t, w = calc.calculate_truck_time(route)
#                         times.append(t)
#                         wait += w

#                 for route in info.get('drone_routes', []):
#                     if route:
#                         t, w, _ = calc.calculate_drone_time(route)
#                         times.append(t)
#                         wait += w

#                 completion_times.append(max(times) if times else 0.0)
#                 waiting_times.append(wait)

#             # ✅ Ghi log trung bình 5 episode
#             avg_completion = np.mean(completion_times)
#             avg_waiting = np.mean(waiting_times)
#             total_obj = avg_completion + avg_waiting

#             self.logger.record("eval/completion_time", avg_completion)
#             self.logger.record("eval/waiting_time", avg_waiting)
#             self.logger.record("eval/total_objective", total_obj)

#             if self.verbose > 0:
#                 print(f"\n[Eval @ step {self.n_calls}]")
#                 print(f"  Completion time: {avg_completion:.2f}")
#                 print(f"  Waiting time:    {avg_waiting:.2f}")
#                 print(f"  Total objective: {total_obj:.2f}")

#         return True


# class WAPPO:
#     """Weight Aggregation PPO wrapper"""
    
#     def __init__(self, env, config_manager, log_dir: str = "./logs/"):
#         self.env = env
#         self.config = config_manager
#         self.log_dir = log_dir
#         self.models = {}
        
#     def create_model(self, weight_id: str, w_completion: float, w_waiting: float):
#         self.config.update_weights(w_completion, w_waiting)
        
#         policy_kwargs = dict(
#             features_extractor_class=VRPDFeatureExtractor,
#             features_extractor_kwargs=dict(features_dim=256),
#             net_arch=dict(
#                 pi=self.config.rl.policy_layers,
#                 vf=self.config.rl.value_layers
#             )
#         )


#         # Dùng MaskablePPO thay cho PPO
#         # model = MaskablePPO(
#         #     "MlpPolicy",
#         #     env=self.env,
#         #     learning_rate=config.rl.learning_rate,
#         #     n_steps=config.rl.n_steps,
#         #     batch_size=config.rl.batch_size,
#         #     n_epochs=config.rl.n_epochs,
#         #     gamma=config.rl.gamma,
#         #     gae_lambda=config.rl.gae_lambda,
#         #     clip_range=config.rl.clip_range,
#         #     ent_coef=config.rl.ent_coef,
#         #     vf_coef=config.rl.vf_coef,
#         #     max_grad_norm=config.rl.max_grad_norm,
#         #     tensorboard_log="./logs/",
#         #     verbose=1
#         # )

        
#         model = MaskablePPO(
#             policy="MlpPolicy",
#             env=self.env,
#             learning_rate=self.config.rl.learning_rate,
#             n_steps=self.config.rl.n_steps,
#             batch_size=self.config.rl.batch_size,
#             n_epochs=self.config.rl.n_epochs,
#             gamma=self.config.rl.gamma,
#             gae_lambda=self.config.rl.gae_lambda,
#             clip_range=self.config.rl.clip_range,
#             ent_coef=self.config.rl.ent_coef,
#             vf_coef=self.config.rl.vf_coef,
#             max_grad_norm=self.config.rl.max_grad_norm,
#             policy_kwargs=policy_kwargs,
#             tensorboard_log=f"{self.log_dir}/{weight_id}",
#             verbose=1
#         )
        
#         self.models[weight_id] = model
#         return model
    
#     def train_model(self, weight_id: str, total_timesteps: int, callback=None):

#         if weight_id not in self.models:
#             raise ValueError(f"Model {weight_id} not found")
        
#         model = self.models[weight_id]
#         progress_callback = ProgressBarCallback(total_timesteps)
#         if callback is not None:
#             from stable_baselines3.common.callbacks import CallbackList
#             callback = CallbackList([callback, progress_callback])
#         else:
#             callback = progress_callback
#         model.learn(
#             total_timesteps=total_timesteps,
#             callback=callback,
#             tb_log_name=f"run_{weight_id}"
#         )
        
#         return model
    
#     def save_model(self, weight_id: str, path: str):
#         if weight_id in self.models:
#             self.models[weight_id].save(path)
    
#     def load_model(self, weight_id: str, path: str):
#         self.models[weight_id] = PPO.load(path, env=self.env)

# class ProgressBarCallback(BaseCallback):
#     """
#     Hiển thị progress bar cho quá trình training PPO.
#     """
#     def __init__(self, total_timesteps: int, verbose=1):
#         super().__init__(verbose)
#         self.total_timesteps = total_timesteps
#         self.pbar = None

#     def _on_training_start(self):
#         self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", ncols=80)

#     def _on_step(self) -> bool:
#         # Mỗi bước SB3 gọi _on_step() một lần
#         self.pbar.update(1)
#         return True

#     def _on_training_end(self):
#         if self.pbar is not None:
#             self.pbar.close()


"""
WA-PPO training helper for ParallelVRPDEnv using sb3-contrib MaskablePPO.
- Gymnasium-compatible
- Proper action masking in both training and evaluation
- Clean callbacks: progress bar + multi-objective eval
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


# ---------------------------
# Progress bar callback
# ---------------------------
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self._start_time = None
        self._last_print = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # In SB3, each call is one env step collected (n_envs=1)
        # Print every ~1s
        now = time.time()
        if now - self._last_print >= 1.0:
            elapsed = now - self._start_time if self._start_time else 0.0
            done = min(self.num_timesteps, self.total_timesteps)
            pct = 100.0 * done / max(self.total_timesteps, 1)
            fps = done / max(elapsed, 1e-6)
            print(f"Training Progress: {done}/{self.total_timesteps} ({pct:.1f}%)  |  fps={fps:.0f}")
            self._last_print = now
        return True


# ---------------------------
# Multi-objective evaluation callback
# ---------------------------
class MultiObjectiveCallback(BaseCallback):
    """
    Pure-Gymnasium eval loop (no VecEnv). Always passes action_masks to model.predict.
    Logs:
      - eval/completion_time
      - eval/waiting_time
      - eval/total_objective
    """
    def __init__(self, eval_env, eval_freq: int = 1000, n_episodes: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env   # base env (already ActionMasker-wrapped)
        self.eval_freq = int(eval_freq)
        self.n_episodes = int(n_episodes)

    def _run_one_episode(self) -> Tuple[float, float]:
        # Reset
        obs, info = self.eval_env.reset()
        done = False
        last_info = info

        # Rollout
        for _ in range(10_000):  # safety cap
            mask = self.eval_env.get_action_mask()
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
            last_info = info
            if terminated or truncated:
                break

        # Aggregate metrics from last_info
        calc = self.eval_env.calculator
        completion_times: List[float] = []
        waiting_total = 0.0

        for route in last_info.get("truck_routes", []):
            if route:
                ct, wt = calc.calculate_truck_time(route)
                completion_times.append(float(ct))
                waiting_total += float(wt)

        for route in last_info.get("drone_routes", []):
            if route:
                ct, wt, _ = calc.calculate_drone_time(route)
                completion_times.append(float(ct))
                waiting_total += float(wt)

        completion = float(max(completion_times)) if completion_times else 0.0
        return completion, waiting_total

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            comps, waits = [], []
            for _ in range(self.n_episodes):
                c, w = self._run_one_episode()
                comps.append(c)
                waits.append(w)

            avg_completion = float(np.mean(comps)) if comps else 0.0
            avg_waiting = float(np.mean(waits)) if waits else 0.0
            self.logger.record("eval/completion_time", avg_completion)
            self.logger.record("eval/waiting_time", avg_waiting)
            self.logger.record("eval/total_objective", avg_completion + avg_waiting)

            if self.verbose:
                print(f"\n[Eval] completion={avg_completion:.2f} | waiting={avg_waiting:.2f} | total={avg_completion+avg_waiting:.2f}")
        return True


# ---------------------------
# Helper: action mask function for wrapper
# ---------------------------
def mask_fn(env) -> np.ndarray:
    # env ở đây là base env đã có get_action_mask()
    return env.get_action_mask()


# ---------------------------
# WA-PPO trainer
# ---------------------------
class WAPPO:
    def __init__(self, env, config_manager, log_dir: Optional[str] = None, seed: int = 0):
        """
        env: base env (Gymnasium). We'll wrap it in ActionMasker + Monitor + DummyVecEnv inside create_model().
        """
        self.base_env = env
        self.config = config_manager
        self.log_dir = log_dir
        self.seed = int(seed)
        self.model: Optional[MaskablePPO] = None

    def create_model(
        self,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        **kwargs,
    ) -> MaskablePPO:
        """
        Wraps env (ActionMasker -> Monitor -> DummyVecEnv) and builds MaskablePPO.
        """
        # Wrap base env for masking
        masked_env = ActionMasker(self.base_env, mask_fn)
        # Monitor
        monitored_env = Monitor(masked_env, filename=None, allow_early_resets=True)
        # Vec
        vec_env = DummyVecEnv([lambda: monitored_env])

        self.model = MaskablePPO(
            policy,
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            seed=self.seed,
            verbose=1,
            tensorboard_log=self.log_dir,
            **kwargs,
        )
        return self.model

    def train_model(
        self,
        total_timesteps: int,
        eval_env,
        eval_freq: int = 1000,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Run .learn() with progress + multi-objective callbacks.
        - eval_env: base Gymnasium env (will be wrapped by ActionMasker for safety here too)
        """
        assert self.model is not None, "Call create_model() first."

        # Ensure eval_env is also wrapped for masking during evaluation inside callback
        if not isinstance(eval_env, ActionMasker):
            eval_env = ActionMasker(eval_env, mask_fn)

        cbs: List[BaseCallback] = []
        if progress_bar:
            cbs.append(ProgressBarCallback(total_timesteps))
        cbs.append(MultiObjectiveCallback(eval_env=eval_env, eval_freq=eval_freq, n_episodes=3, verbose=1))

        self.model.learn(total_timesteps=total_timesteps, callback=cbs)

        return {
            "total_timesteps": total_timesteps,
        }
