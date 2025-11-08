
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# ---------- Attention-based Feature Extractor ----------
class VRPDFeatureExtractor(BaseFeaturesExtractor):
    # Turns a flat observation into (N,6) customer matrix + global vector,
    # then applies MultiheadAttention over customers and concatenates pooled
    # context with global features.
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, n_heads: int = 4):
        super().__init__(observation_space, features_dim)
        self.obs_dim = int(observation_space.shape[0])
        # we infer N from obs_dim: obs = N*6 + (2 + 3)
        # => N = (obs_dim - 5) // 6
        # if (self.obs_dim - 5) % 6 != 0:
        #     raise AssertionError("Observation shape must match N*6 + 5")
        self.N = (self.obs_dim - 5) // 6

        self.cust_in = 6
        self.global_in = 5

        d_model = 128
        self.cust_proj = nn.Linear(self.cust_in, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.post = nn.Sequential(
            nn.Linear(d_model + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        B = obs.shape[0]
        cust = obs[:, : self.N * 6].reshape(B, self.N, 6)
        glob = obs[:, self.N * 6 : self.N * 6 + 5]

        cust = self.cust_proj(cust)                         # (B,N,d)
        attn_out, _ = self.attn(cust, cust, cust)          # (B,N,d)
        pooled = attn_out.mean(dim=1)                      # (B,d)

        g = self.global_mlp(glob)                          # (B,64)
        feat = torch.cat([pooled, g], dim=1)               # (B,d+64)
        return self.post(feat)                              # (B,features_dim)

# ---------- Progress bar (optional) ----------
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, print_every: int = 5_000):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.print_every = print_every
        self._last = 0

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n - self._last >= self.print_every:
            pct = 100.0 * n / max(self.total_timesteps, 1)
            self._last = n
            self.logger.record("progress/percent", pct)
        return True

# ---------- Training wrapper ----------
# def mask_fn(env) -> torch.Tensor:
#     # env must implement get_action_mask()
#     return torch.as_tensor(env.get_action_mask())

def mask_fn(env) -> torch.Tensor:
    """
    Hàm mask_fn dùng cho ActionMasker.
    Khi được gọi, env là ActionMasker, nên ta truy cập env.env.get_action_mask().
    """
    inner_env = getattr(env, "env", env)
    return torch.as_tensor(inner_env.get_action_mask(), dtype=torch.bool)


class WAPPOTrainer:
    def __init__(self, env: gym.Env, eval_env: Optional[gym.Env] = None, policy_kwargs: Optional[Dict[str, Any]] = None):
        self.env = ActionMasker(env, mask_fn)
        self.eval_env = ActionMasker(eval_env, mask_fn) if eval_env is not None else None

        if policy_kwargs is None:
            policy_kwargs = dict(
                features_extractor_class=VRPDFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256, n_heads=4),
                net_arch=dict(pi=[128], vf=[128])
            )

        self.model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=self.env,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=1024,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            verbose=1
        )

    def learn(self, total_timesteps: int = 200_000):
        cbs: List[BaseCallback] = [ProgressBarCallback(total_timesteps)]
        self.model.learn(total_timesteps=total_timesteps, callback=cbs)

    def predict(self, obs,action_masks=None, deterministic: bool = True):
        return self.model.predict(obs, action_masks=action_masks, deterministic=deterministic)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        from sb3_contrib.ppo_mask import MaskablePPO as _M
        self.model = _M.load(path)
