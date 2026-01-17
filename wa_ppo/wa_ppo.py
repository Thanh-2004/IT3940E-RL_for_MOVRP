from typing import Any, Dict, Optional, List

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


# ---------- Mask function ----------
def mask_fn(env: gym.Env):
    """Extract action mask from environment"""
    if hasattr(env, "get_action_mask"):
        return env.get_action_mask()
    return [True] * env.action_space.n


# ---------- Feature Extractor ----------
class VRPDFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for ParallelVRPDEnv with multi-objective weights
    
    Observation structure:
    - N * 6: Customer features (x, y, demand_norm, visited, truck_only, dist_depot)
    - Global features: time_norm + vehicle features + 2 weights (w_service, w_wait)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_customers: int = 50,
        customer_dim: int = 6,
    ):
        super().__init__(observation_space, features_dim)
        self.n_customers = n_customers
        self.customer_dim = customer_dim

        total_dim = observation_space.shape[0]
        cust_block = n_customers * customer_dim
        assert cust_block <= total_dim, "n_customers * customer_dim exceeds obs dim"
        self.global_dim = total_dim - cust_block

        # Customer encoder
        self.cust_net = nn.Sequential(
            nn.LayerNorm(cust_block),
            nn.Linear(cust_block, features_dim),
            nn.ReLU(),
        )
        
        # Global encoder (includes time, vehicles, weights)
        self.glob_net = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(self.global_dim, features_dim // 2),
            nn.ReLU(),
        )
        
        # Output fusion
        self.out = nn.Sequential(
            nn.Linear(features_dim + features_dim // 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observation
        
        Args:
            obs: Tensor of shape (batch, obs_dim)
            
        Returns:
            features: Tensor of shape (batch, features_dim)
        """
        # Split observation
        cust = obs[:, : self.n_customers * self.customer_dim]
        glob = obs[:, self.n_customers * self.customer_dim :]
        
        # Encode
        c_emb = self.cust_net(cust)
        g_emb = self.glob_net(glob)
        
        # Fuse
        return self.out(torch.cat([c_emb, g_emb], dim=-1))


# ---------- Callback ----------
class ProgressBarCallback(BaseCallback):
    """Simple progress bar for training"""
    
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_logged = 0

    def _on_step(self) -> bool:
        # Log every 10%
        if self.total_timesteps > 0:
            progress = self.num_timesteps / self.total_timesteps
            if progress - self.last_logged >= 0.1:
                print(f"[TRAIN] {progress:6.1%} ({self.num_timesteps}/{self.total_timesteps})")
                self.last_logged = progress
        return True


class ObjectiveLoggingCallback(BaseCallback):
    """Log multi-objective metrics during training"""
    
    def __init__(self, eval_env: Optional[gym.Env] = None, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.eval_env is None:
            return True
            
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            
            # Run evaluation episode
            obs, info = self.eval_env.reset()
            done = False
            step = 0
            
            while not done and step < 1000:
                mask = info.get("action_mask", None)
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                done = terminated or truncated
                step += 1
            
            # Log objectives
            if self.verbose > 0:
                service_time = info.get("total_service_time", 0.0)
                wait_time = info.get("total_waiting_time", 0.0)
                completion = info.get("completion_rate", 0.0)
                print(f"\n[EVAL @ {self.num_timesteps}] "
                      f"Service={service_time:.1f}, Wait={wait_time:.1f}, "
                      f"Completion={completion*100:.1f}%")
        
        return True


# ---------- Trainer ----------
class WAPPOTrainer:
    """
    Weight-Aware PPO Trainer for Multi-Objective VRPD
    
    Uses MaskablePPO with custom feature extractor that processes
    customer features and multi-objective weights
    """
    
    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        n_customers: Optional[int] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 1,
    ):
        """
        Initialize trainer
        
        Args:
            env: Training environment (should be wrapped with ActionMasker)
            eval_env: Evaluation environment (optional)
            policy_kwargs: Policy network configuration
            n_customers: Number of customers (extracted from env if not provided)
            learning_rate: Learning rate
            n_steps: Steps per rollout
            batch_size: Batch size for training
            n_epochs: Number of optimization epochs per rollout
            gamma: Discount factor
            verbose: Verbosity level
        """
        # Store environments
        self.env = env if isinstance(env, ActionMasker) else ActionMasker(env, mask_fn)
        self.eval_env = eval_env
        if self.eval_env is not None and not isinstance(self.eval_env, ActionMasker):
            self.eval_env = ActionMasker(self.eval_env, mask_fn)

        # Extract n_customers
        if n_customers is None:
            # Try to get from wrapped env
            base_env = env.env if isinstance(env, ActionMasker) else env
            if hasattr(base_env, "N"):
                n_customers = int(base_env.N)
            else:
                raise ValueError("Please provide n_customers or ensure env has attribute N")

        # Default policy configuration
        if policy_kwargs is None:
            policy_kwargs = dict(
                features_extractor_class=VRPDFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=256,
                    n_customers=n_customers,
                    customer_dim=6,  # (x, y, demand_norm, visited, truck_only, dist_depot)
                ),
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            )

        # Create model
        self.model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=verbose,
        )
        
        print(f"✅ WAPPOTrainer initialized for {n_customers} customers")
        print(f"   Policy: {policy_kwargs['features_extractor_class'].__name__}")
        print(f"   Features dim: {policy_kwargs['features_extractor_kwargs']['features_dim']}")

    def learn(self, total_timesteps: int = 200_000, callback: Optional[BaseCallback] = None):
        """
        Train the model
        
        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback (will be combined with progress bar)
        """
        # Create callbacks
        callbacks = [ProgressBarCallback(total_timesteps)]
        
        # Add objective logging if eval_env is provided
        if self.eval_env is not None:
            callbacks.append(ObjectiveLoggingCallback(self.eval_env, eval_freq=10000))
        
        # Add user callback if provided
        if callback is not None:
            callbacks.append(callback)
        
        # Train
        print(f"\n🚀 Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        print(f"✅ Training completed!")

    def predict(self, obs, action_masks=None, deterministic: bool = True):
        """Predict action given observation and mask"""
        return self.model.predict(obs, action_masks=action_masks, deterministic=deterministic)

    def save(self, path: str):
        """Save model to disk"""
        self.model.save(path)
        print(f"💾 Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        self.model = MaskablePPO.load(path)
        print(f"📂 Model loaded from {path}")