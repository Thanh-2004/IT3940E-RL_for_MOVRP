from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import gymnasium as gym

# === Dummy environment chỉ để kiểm tra mask ===
class MaskTestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.step_count = 0

    def _get_action_mask(self):
        mask = np.ones(self.action_space.n, dtype=bool)
        # Mỗi 5 step lại chặn 1 action
        mask[self.step_count % 5] = False
        print(f"[ENV] step={self.step_count} → valid={np.where(mask)[0]}")
        return mask

    def get_action_mask(self):
        return self._get_action_mask()

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        obs = np.zeros(3, dtype=np.float32)
        info = {}
        print("[ENV] reset called")
        return obs, info

    def step(self, action):
        mask = self._get_action_mask()
        valid = mask[action]
        print(f"[ENV] step {self.step_count}: action={action} valid={valid}")
        self.step_count += 1
        reward = 1.0 if valid else -10.0
        obs = np.zeros(3, dtype=np.float32)
        terminated = self.step_count >= 10
        truncated = False
        info = {"action_mask": mask}
        return obs, reward, terminated, truncated, info


# === Định nghĩa mask function cho wrapper ===
def mask_fn(env):
    mask = env.get_action_mask()
    print(f"[MASK_FN] mask sum={np.sum(mask)}")
    return mask


# === Setup environment đúng thứ tự ===
base_env = MaskTestEnv()
env = ActionMasker(base_env, mask_fn)
env = DummyVecEnv([lambda: env])

# === Huấn luyện ngắn để kiểm tra mask có truyền được không ===
model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=32, ent_coef=0.01)

print("\n=== TEST: Running short learn() ===")
model.learn(total_timesteps=64)

# === Kiểm tra predict() riêng ===
obs, _ = base_env.reset()
mask = base_env.get_action_mask()
print(f"\n[PREDICT TEST] Mask before predict: {mask}")
action, _ = model.predict(obs, action_masks=mask)
print(f"[PREDICT TEST] PPO chose action: {action}")
