# train.py
from __future__ import annotations
import argparse
import json
import os, sys, math, time, random
from pathlib import Path
from typing import Dict, Tuple, List

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))

# Config + Env
from environment.config import build_config_from_files
from environment.env import ParallelVRPDEnvironment                   

# Model
from model.models_v2 import MOPVRP

# ---------------------------
# PPO: small helper utilities
# ---------------------------
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Đảm bảo dữ liệu được ghi ngay lập tức

    def flush(self):
        # Cần thiết cho khả năng tương thích với sys.stdout
        self.terminal.flush()
        self.log.flush()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def to_device_obs(obs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in obs.items()}

def concat_dict_of_lists(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        if not torch.is_tensor(batch[0][k]): 
            continue
        out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out

def _to_1d(x: torch.Tensor) -> torch.Tensor:
    # make sure shape is [B]; for scalars make [1]
    return x.view(1) if x.dim() == 0 else x.view(-1)


@torch.no_grad()
def eval_single_episode(env, policy, device, w1: float, w2: float, max_steps: int = 100000):
    """
    Chạy 1 episode đánh giá và trả về (F1, F2, actions_history).
    """
    env.configure_fixed_weights(w1, w2)
    obs = env.reset()
    obs = _ensure_batch(obs, device)
    done = False
    steps = 0
    last_info = None
    
    # List lưu lịch sử hành động
    actions_history = [] 
    
    h = policy.init_hidden(1, device=device)
    while not done and steps < max_steps:
        # Chạy policy deterministic (chọn action tốt nhất)
        action, _, _, _, h = policy.act(obs, h_prev=h, deterministic=True)
        
        # Chuyển action tensor sang tuple chuẩn (veh_type, veh_idx, node)
        act_tuple = tuple(int(x) for x in action)
        actions_history.append(act_tuple)
        
        step = env.step(act_tuple)
        last_info = step.info
        done = bool(step.done.item()) if torch.is_tensor(step.done) else bool(step.done)
        obs = _ensure_batch(step.obs, device)
        steps += 1
        
    assert last_info is not None, "No step info collected during evaluation."
    return float(last_info["F1"]), float(last_info["F2"]), actions_history


def evaluate_policy_bundle(policy, device, cfg, env_kwargs):
    """
    Create a fresh eval env (so we don't disturb the training env's sampled weights)
    and evaluate 3 settings:
      - F1 best: (w1=1, w2=0)  → key 'f1'
      - F2 best: (w1=0, w2=1)  → key 'f2'
      - Sum best: (w1=0.5, w2=0.5) → key 'sum'
    Returns dict: {'f1': (F1,F2), 'f2': (F1,F2), 'sum': (F1,F2)}
    """
    EvalEnv = ParallelVRPDEnvironment  # same wrapper used in training
    eval_env = EvalEnv(cfg, weight_strategy="fixed", device=device, **env_kwargs)

    res = {}
    res["f1"] = eval_single_episode(eval_env, policy, device, 1.0, 0.0)
    res["f2"] = eval_single_episode(eval_env, policy, device, 0.0, 1.0)
    res["sum"] = eval_single_episode(eval_env, policy, device, 0.5, 0.5)
    return res


@torch.no_grad()
def evaluate_policy_weight_sweep(policy, device, cfg, env_kwargs, step: float = 0.02, max_steps: int = 100000):
    EvalEnv = ParallelVRPDEnvironment
    eval_env = EvalEnv(cfg, weight_strategy="fixed", device=device, **env_kwargs)

    def run_once(w1: float, w2: float):
        # Nhận thêm history từ hàm eval
        F1, F2, history = eval_single_episode(eval_env, policy, device, w1, w2, max_steps=max_steps)
        return {
            "w1": float(w1), 
            "w2": float(w2), 
            "F1": float(F1), 
            "F2": float(F2), 
            "history": history  # Lưu lại lịch sử
        }

    # ... (Phần tạo lưới weights giữ nguyên) ...
    n_steps = int(round(1.0 / step))
    weights = []
    for i in range(n_steps + 1):
        w1 = round(1.0 - i * step, 2)
        if w1 < 0.0: w1 = 0.0
        w2 = round(1.0 - w1, 2)
        weights.append((w1, w2))

    results = []
    for w1, w2 in tqdm(weights, desc="Evaluating Weights"):
        results.append(run_once(w1, w2))

    # ... (Phần tìm argmin giữ nguyên) ...
    def argmin_key(items, key, tie_keys):
        best = None
        for it in items:
            cand = (it[key],) + tuple(it[k] for k in tie_keys)
            if (best is None) or (cand < ((best[key],) + tuple(best[k] for k in tie_keys))):
                best = it
        return best

    best_f1  = argmin_key(results, "F1", ("F2",))
    best_f2  = argmin_key(results, "F2", ("F1",))
    
    sum_val = 0
    for r in results:
        r["_sum"] = r["F1"] + r["F2"]
        sum_val += r["_sum"]
    best_sum = argmin_key(results, "_sum", ("F1","F2"))
    
    # Drop helper key
    for r in results:
        r.pop("_sum", None)

    return {
        "best_f1":  best_f1,
        "best_f2":  best_f2,
        "best_sum": best_sum,
        "avg": sum_val / len(results)
    }


def compute_gae(rewards, values, dones, gamma=1, lam=0.95):
    # rewards/values/dones are lists of tensors (maybe scalars)
    rewards = [_to_1d(r) for r in rewards]
    values  = [_to_1d(v) for v in values]
    dones   = [_to_1d(d).to(dtype=torch.float32) for d in dones]

    T = len(rewards)
    B = rewards[0].shape[0]
    adv = torch.zeros(B, device=rewards[0].device)
    advantages = []
    last_value = values[-1]  # bootstrap V(s_T)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]              # 1 for non-terminal, 0 for terminal
        delta = rewards[t] + gamma * last_value * mask - values[t]
        adv = delta + gamma * lam * adv * mask
        advantages.append(adv)
        last_value = values[t]

    advantages = advantages[::-1]
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns



# ---------------------------
# Rollout collection (on-policy)
# ---------------------------
@torch.no_grad()
def collect_rollout(env: ParallelVRPDEnvironment, policy: nn.Module, horizon: int, device: torch.device):
    obs = env.reset()
    obs = to_device_obs(obs, device)

    obs_buf = []
    actions, logps, values, rewards, dones = [], [], [], [], []
    h_buf = []
    h = policy.init_hidden(1, device=device)

    t = 0
    done = torch.tensor([False], device=device)
    while t < horizon and not bool(done.item()):
        # cache obs (batch B=1)
        obs_batched = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
        obs_batched = _ensure_batch(obs_batched, device=device)
        obs_buf.append(obs_batched)

        # policy action
        h_prev = h
        action, logp, value, _, h = policy.act(obs_batched, h_prev=h_prev)
        h_buf.append(h_prev.detach())

        # step env
        step = env.step(tuple(int(x) for x in action))

        # pack reward/done as [1]
        reward_t = step.reward
        if not torch.is_tensor(reward_t):
            reward_t = torch.as_tensor(reward_t, device=device)
        reward_t = reward_t.view(1)
        rewards.append(reward_t)

        done_t = step.done
        if not torch.is_tensor(done_t):
            done_t = torch.as_tensor(done_t, device=device)
        done_t = done_t.view(1)
        dones.append(done_t)

        # store action/logp/value
        actions.append(torch.as_tensor(action, device=device).unsqueeze(0))  # [1,3]
        logps.append(logp)    # [1]
        values.append(value)  # [1]

        # >>> CRITICAL: update loop guard <<<
        done = done_t  # <--- this makes the while condition work

        # advance obs
        obs = to_device_obs(step.obs, device)
        t += 1
    # print(rewards)

    # bootstrap value
    with torch.no_grad():
        if not bool(done.item()):
            last_v = policy.value_only(obs, h_prev=h)
        else:
            last_v = torch.zeros_like(values[-1])
        values.append(last_v)

    traj = {
        "obs_seq": obs_buf,
        "actions": torch.cat(actions, dim=0),     # [T, 1, 3]
        "logps": torch.cat(logps, dim=0),         # [T, 1]
        "values": torch.cat(values[:-1], dim=0),  # [T, 1]
        "rewards": torch.cat(rewards, dim=0),     # [T, 1]
        "dones": torch.cat(dones, dim=0),         # [T, 1]
        "last_value": values[-1],                 # [1]
        "hxs": torch.stack(h_buf, dim=0),         # [T, 1, W+1, H]
    }
    return traj


def ppo_update(policy, optimizer, traj,
               epochs, clip_ratio, vf_coef, ent_coef, max_grad_norm, device):

    # --- (Giữ nguyên phần chuẩn bị dữ liệu đầu vào) ---
    T = traj["actions"].shape[0]
    B = traj["actions"].shape[1] if traj["actions"].dim() == 3 else 1
    N = T * B

    rewards = [traj["rewards"][t] for t in range(T)]
    values_seq = [traj["values"][t] for t in range(T)] + [traj["last_value"]]
    dones  = [traj["dones"][t]  for t in range(T)]
    adv_list, ret_list = compute_gae(rewards, values_seq, dones, gamma=1, lam=0.95)

    advantages = torch.stack(adv_list, dim=0)
    returns    = torch.stack(ret_list, dim=0)
    with torch.no_grad():
        adv_mean = advantages.mean()
        adv_std  = advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std

    old_logp = traj["logps"].detach()
    actions  = traj["actions"].detach()

    # Flatten time/batch
    advantages = advantages.view(N, 1)
    returns    = returns.view(N, 1)
    old_logp   = old_logp.view(N, 1)
    actions    = actions.view(N, 3)

    # Rebuild flat obs
    flat_obs = {}
    for k in ["graph_ctx", "trucks_ctx", "drones_ctx", "mask_trk", "mask_dr", "weights"]:
        xs = [o[k] for o in traj["obs_seq"]]
        flat_obs[k] = torch.cat(xs, dim=0).to(device)
    hxs = traj["hxs"].to(device)
    h_prev = hxs.view(N, *hxs.shape[2:])

    # --- optimization epochs ---
    # Khởi tạo các biến metric để tránh lỗi reference nếu epochs=0 (dù hiếm)
    approx_kl = torch.tensor(0.0)
    explained_var = 0.0
    clip_frac = torch.tensor(0.0)

    for _ in range(epochs):
        new_logp, entropy, value = policy.evaluate_actions(flat_obs, actions, h_prev=h_prev)

        # Policy loss
        log_ratio = new_logp - old_logp
        log_ratio = torch.nan_to_num(log_ratio, nan=0.0).clamp(-20, 20)
        ratio = torch.exp(log_ratio)

        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surr1   = ratio * advantages
        surr2   = clipped * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Value loss
        value_loss  = F.mse_loss(value, returns)

        # Entropy bonus
        entropy_bonus = entropy.mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

        # --- TÍNH TOÁN CÁC METRIC BỔ SUNG (NEW) ---
        with torch.no_grad():
            # 1. Approx KL Divergence (theo http://joschu.net/blog/kl-approx.html)
            # Công thức này chính xác hơn (new_logp - old_logp).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()

            # 2. Clip Fraction: Tỉ lệ số mẫu bị cắt bởi cơ chế clip của PPO
            clip_frac = (torch.abs(ratio - 1.0) > clip_ratio).float().mean()

            # 3. Explained Variance: 1 - Var(error) / Var(target)
            y_pred = value.view(-1)
            y_true = returns.view(-1)
            var_y = torch.var(y_true)
            if var_y < 1e-8:
                explained_var = float('nan') # Tránh chia cho 0
            else:
                explained_var = 1 - torch.var(y_true - y_pred) / var_y

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

    # --- Cập nhật dictionary stats trả về ---
    stats = {
        "loss": float(loss.item()),
        "pg": float(policy_loss.item()),
        "v": float(value_loss.item()),
        "ent": float(entropy_bonus.item()),
        "adv_mean": float(adv_mean.item()),
        "adv_std": float(adv_std.item()),
        # Metrics mới
        "approx_kl": float(approx_kl.item()),
        "clip_frac": float(clip_frac.item()),
        "explained_var": float(explained_var) if isinstance(explained_var, float) else float(explained_var.item())
    }
    return stats

def dirichlet_for_iter(it, total):
    # Start very F1-heavy, relax over time
    a1 = 12.0 * (1 - it/total) + 2.0      # from ~14 → 2
    a2 = 2.0 * (1 - it/total) + 2.0       # from ~4  → 2
    return (a1, a2)



def _ensure_batch(obs, device):
    keys = ["graph_ctx", "trucks_ctx", "drones_ctx", "mask_trk", "mask_dr", "weights"]
    out = {}
    for k in keys:
        x = obs[k]
        if torch.is_tensor(x):
            x = x.to(device)
            if k == "weights":
                # weights: [2] -> [1,2]; if already [B,2], leave as-is
                if x.dim() == 1:
                    x = x.unsqueeze(0)

            else:
                # others: [V,F] / [K,F] / [D,F] / [K,V] / [D,V] -> add batch
                if x.dim() == 2:
                    x = x.unsqueeze(0)
        out[k] = x
    return out


# ---------------------------
# Main training loop
# ---------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_iters", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.3)
    parser.add_argument("--ent_coef_start", type=float, default=0.02)
    parser.add_argument("--ent_coef_end", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_dir", type=str, default="./results_test")
    parser.add_argument("--eval_step", type=float, default=0.02)

    # model
    parser.add_argument("--d_model", type=int, default=128)
    # parser.add_argument("--nhead", type=int, default=8)
    # parser.add_argument("--depth", type=int, default=1)
    # parser.add_argument("--depth_t", type=int, default=1)
    # parser.add_argument("--depth_d", type=int, default=1)
    # parser.add_argument("--depth_c", type=int, default=1)
    parser.add_argument("--mem_window", type=int, default=64)
    # parser.add_argument("--mem_depth", type=int, default=1)

    # env/config
    parser.add_argument("--truck_json", type=str, default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json")
    parser.add_argument("--drone_json", type=str, default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json")
    parser.add_argument("--customers_txt", type=str, default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/20.10.1.txt")
    parser.add_argument("--max_trips_per_drone", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=2048)
    parser.add_argument("--dirichlet_alpha", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--weight_curriculum", choices=["cycle", "sample", "fixed"], default="cycle")
    parser.add_argument("--fixed_weights", type=float, nargs=2, default=(0.5, 0.5))
    parser.add_argument("--data_dir", type=str, default="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/")
    return parser

def format_move(move):
    """Helper để format text hiển thị hành động"""
    v_type, v_idx, node = move
    v_name = "Truck" if v_type == 0 else "Drone"
    return f"{v_name} {v_idx} -> Node {node}"

def main(argv: List[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stats_file = run_dir / "train_stats.csv"
    # Nếu là bắt đầu mới (it=1), ghi header
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("iter,instance,loss,pg,v,ent,adv,kl,clip,ex_var\n")
    
    # Kích hoạt ghi log vào file train_log.txt bên trong run_dir
    log_file = run_dir / "train_log.txt"
    sys.stdout = Logger(log_file)
    
    print(f"--- Training Started ---")
    print(f"Run Directory: {run_dir}")
    print(f"Log File: {log_file}")

    # ---------- Hyperparams ----------
    seed            = args.seed
    total_iters     = args.total_iters
    horizon         = args.horizon
    lr              = args.lr
    clip_ratio      = args.clip_ratio
    vf_coef         = args.vf_coef
    ent_coef_start  = args.ent_coef_start
    ent_coef_end    = args.ent_coef_end
    max_grad_norm   = args.max_grad_norm
    device          = torch.device(args.device)

    set_seed(seed)

    # ---------- Curriculum ----------
    curriculum_instances = [
    "20.10.1.txt", "20.10.3.txt", "20.20.3.txt", "20.20.2.txt",
    "50.10.1.txt", "50.10.3.txt", "50.10.4.txt", "50.10.2.txt"
    ]

    # curriculum_instances = [
    # "20.10.1.txt", "20.10.3.txt", "20.20.3.txt", "20.20.2.txt",
    # "20.10.1.txt", "20.10.3.txt", "20.20.3.txt", "20.20.2.txt",
    # ]

    iters_per_instance = args.total_iters // len(curriculum_instances)
    current_instance_idx = -1  # Để nhận diện lần đầu tiên load env

    total_weights = 0
    instance_weights = []
    for name in curriculum_instances:
        weight = 2 if name.startswith("50") else 1
        instance_weights.append(weight)
        total_weights += weight

    unit_iter = args.total_iters / total_weights

    # Tính mốc iter kết thúc (milestones) cho từng instance
    milestones = []
    cumulative_iter = 0
    for w in instance_weights:
        cumulative_iter += w * unit_iter
        milestones.append(int(cumulative_iter))

    # ---------- Build config + env ----------
    truck_json_path = args.truck_json
    drone_json_path = args.drone_json
    customers_txt   = args.customers_txt

    cfg, meta = build_config_from_files(truck_json_path, drone_json_path, customers_txt)

    env = ParallelVRPDEnvironment(
        cfg,
        max_drone_trips=args.max_trips_per_drone,
        dirichlet_params=tuple(args.dirichlet_alpha),
        max_episode_steps=args.max_steps,
        device=device,
    )

    # ---- Eval env kwargs to mirror training env (but with fixed weights for eval) ----
    eval_env_kwargs = dict(
        display_actions=False,
        maintain_trip_cycles=True,
        parallel_mode=False,
        enable_idle_action=False,
        max_drone_trips=args.max_trips_per_drone,
        normalize_targets=False,
        max_episode_steps=args.max_steps,
        dirichlet_params=tuple(args.dirichlet_alpha),  # unused when weight_strategy='fixed'
    )

    # ---- Best trackers ----
    best_f1 = math.inf
    best_f2 = math.inf
    best_sum = math.inf
    best_avg = math.inf


    # ---------- Infer Fg/Ft/Fd from first obs ----------
    with torch.no_grad():
        obs0 = env.reset()
        obs0 = _ensure_batch(obs0, device)

        # We assume the last dimension is the feature dimension
        # graph_ctx: [B, V, Fg]
        # trucks_ctx: [B, K, Ft]
        # drones_ctx: [B, D, Fd]
        Fg = obs0["graph_ctx"].shape[-1]
        Ft = obs0["trucks_ctx"].shape[-1]
        Fd = obs0["drones_ctx"].shape[-1]

        # sanity checks for masks: [B, K, V] and [B, D, V]
        B, V = obs0["graph_ctx"].shape[0], obs0["graph_ctx"].shape[1]
        K = obs0["trucks_ctx"].shape[1]
        D = obs0["drones_ctx"].shape[1]
        assert obs0["mask_trk"].shape[:3] == (B, K, V), \
            f"mask_trk shape {obs0['mask_trk'].shape} expected {(B,K,V)} + maybe extra dims"
        assert obs0["mask_dr"].shape[:3] == (B, D, V), \
            f"mask_dr shape {obs0['mask_dr'].shape} expected {(B,D,V)} + maybe extra dims"

    # ---------- Build policy with proper feature sizes ----------

    STATIC_SIZE = 7      
    DYN_TRUCK_SIZE = 5    
    DYN_DRONE_SIZE = 12   

    policy = MOPVRP(
        static_size=STATIC_SIZE,
        dyn_truck_size=DYN_TRUCK_SIZE,
        dyn_drone_size=DYN_DRONE_SIZE,
        hidden_size=args.d_model,  # Ví dụ: 128
        device=device
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ---------- Training ----------
    log_every = args.log_every
    save_every = args.save_every
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for it in tqdm(range(1, total_iters + 1)):
        # 1) Collect rollout

        # alpha = dirichlet_for_iter(it, total_iters)
        # env.dirichlet_alpha = alpha

        new_instance_idx = 0
        for idx, m in enumerate(milestones):
            if it <= m:
                new_instance_idx = idx
                break
        else:
            new_instance_idx = len(curriculum_instances) - 1

        if new_instance_idx != current_instance_idx:
            if current_instance_idx != -1:
                prev_name = curriculum_instances[current_instance_idx].replace(".txt", "")
                milestone_path = ckpt_dir / f"milestone_{prev_name}.pt"
                torch.save({
                    "iter": it,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "instance": prev_name
                }, milestone_path)
                print(f"\n[CURRICULUM] Milestone saved: {milestone_path}")

            current_instance_idx = new_instance_idx
            instance_file = os.path.join(args.data_dir, curriculum_instances[current_instance_idx])
            print(f"\n[CURRICULUM] Starting Level {current_instance_idx + 1}: {curriculum_instances[current_instance_idx]}")

            cfg, meta = build_config_from_files(args.truck_json, args.drone_json, instance_file)
            env = ParallelVRPDEnvironment(cfg, device=device, weight_strategy="sample", **eval_env_kwargs) # Sử dụng kwargs chung

            # 4. QUAN TRỌNG: Reset trackers cho instance mới
            best_f1, best_f2, best_sum, best_avg = math.inf, math.inf, math.inf, math.inf


        # weight curriculum
        if args.weight_curriculum == "cycle":
            if it % 3 == 1:
                env.configure_fixed_weights(1.0, 0.0)
            elif it % 3 == 2:
                env.configure_fixed_weights(0.0, 1.0)
            else:
                env.weight_strategy = "sample"
        elif args.weight_curriculum == "fixed":
            env.configure_fixed_weights(args.fixed_weights[0], args.fixed_weights[1])
        else:
            env.weight_strategy = "sample"

        
        traj = collect_rollout(env, policy, horizon=horizon, device=device)

        # 2) PPO update
        ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * (it / total_iters)
        stats = ppo_update(policy, optimizer, traj,
                           epochs=4,
                           clip_ratio=clip_ratio,
                           vf_coef=vf_coef,
                           ent_coef=ent_coef,
                           max_grad_norm=max_grad_norm,
                           device=device)


        
        if it % log_every == 0:
            print(f"\n[Iter {it:05d}] "
                  f"loss={stats['loss']:.4f} pg={stats['pg']:.4f} v={stats['v']:.4f} "
                  f"ent={stats['ent']:.4f} adv={stats['adv_mean']:.3f} "
                  f"kl={stats['approx_kl']:.4f} clip={stats['clip_frac']:.2f} "
                  f"ex_var={stats['explained_var']:.3f}")

            inst_name = curriculum_instances[current_instance_idx]
            with open(stats_file, "a", encoding="utf-8") as f:
                f.write(f"{it},{inst_name},{stats['loss']:.6f},{stats['pg']:.6f},"
                        f"{stats['v']:.6f},{stats['ent']:.6f},{stats['adv_mean']:.6f},"
                        f"{stats['approx_kl']:.6f},{stats['clip_frac']:.6f},"
                        f"{stats['explained_var']:.6f}\n")

        if it % save_every == 0:

            # ---------- Evaluate & Save Bests ----------
            print("[INFO]: Running Evaluate")
            eval_res = evaluate_policy_weight_sweep(policy, device, cfg, eval_env_kwargs, step=args.eval_step)

            # --- CODE GHI FILE SOLUTION MỚI ---
            solution_file = ckpt_dir / f"solutions_iter_{it:06d}.txt"
            with open(solution_file, "w", encoding="utf-8") as f:
                f.write(f"--- Solutions at Iteration {it} ---\n\n")
                
                # 1. Ghi lời giải Best F1
                b_f1 = eval_res["best_f1"]
                f.write(f"=== BEST F1 SCENARIO ===\n")
                f.write(f"Weights: w1={b_f1['w1']:.2f}, w2={b_f1['w2']:.2f}\n")
                f.write(f"Result: F1={b_f1['F1']:.2f}, F2={b_f1['F2']:.2f}\n")
                f.write("Moves:\n")
                for step_idx, move in enumerate(b_f1['history']):
                    f.write(f"  {step_idx+1}. {format_move(move)}\n")
                f.write("\n")

                # 2. Ghi lời giải Best F2
                b_f2 = eval_res["best_f2"]
                f.write(f"=== BEST F2 SCENARIO ===\n")
                f.write(f"Weights: w1={b_f2['w1']:.2f}, w2={b_f2['w2']:.2f}\n")
                f.write(f"Result: F1={b_f2['F1']:.2f}, F2={b_f2['F2']:.2f}\n")
                f.write("Moves:\n")
                for step_idx, move in enumerate(b_f2['history']):
                    f.write(f"  {step_idx+1}. {format_move(move)}\n")
                f.write("\n")

                # 3. Ghi lời giải Best Sum
                b_sum = eval_res["best_sum"]
                f.write(f"=== BEST SUM SCENARIO ===\n")
                f.write(f"Weights: w1={b_sum['w1']:.2f}, w2={b_sum['w2']:.2f}\n")
                f.write(f"Result: F1={b_sum['F1']:.2f}, F2={b_sum['F2']:.2f}\n")
                f.write("Moves:\n")
                for step_idx, move in enumerate(b_sum['history']):
                    f.write(f"  {step_idx+1}. {format_move(move)}\n")
                f.write("-" * 40 + "\n")
            
            print(f"[INFO]: Saved detailed solutions to {solution_file}")

            (F1_f1, F2_f1) = (eval_res["best_f1"]['F1'], eval_res["best_f1"]['F2'])
            (F1_f2, F2_f2) = (eval_res["best_f2"]['F1'], eval_res['best_f2']['F2'])
            (F1_s, F2_s)   = (eval_res["best_sum"]['F1'], eval_res['best_sum']['F2'])
            avg_sum = eval_res['avg']

            inst_name = curriculum_instances[current_instance_idx].replace(".txt", "")

            def save_ckpt(name, val_score):
                checkpoint = {
                    "iter": it,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(), # Đã thêm optimizer vào đây
                    "instance_idx": current_instance_idx,
                    "val": val_score,
                    "best_f1": best_f1,
                    "best_f2": best_f2,
                    "best_sum": best_sum,
                    "best_avg": best_avg
                }
                torch.save(checkpoint, ckpt_dir / name)

            save_ckpt("last_checkpoint.pt", None)

            if F1_f1 < best_f1:
                best_f1 = F1_f1
                save_ckpt(f"best_f1_{inst_name}.pt", F1_f1)

            if F2_f2 < best_f2:
                best_f2 = F2_f2
                save_ckpt(f"best_f2_{inst_name}.pt", F2_f2)

            current_sum = F1_s + F2_s
            if current_sum < best_sum:
                best_sum = current_sum
                save_ckpt(f"best_sum_{inst_name}.pt", current_sum)

            if avg_sum < best_avg:
                best_avg = avg_sum
                save_ckpt(f"best_mean_{inst_name}.pt", avg_sum)

            print(f"[Eval @ {it}] F1_best={best_f1:.3f}  F2_best={best_f2:.3f}  "
                  f"(F1+F2)_best={best_sum:.3f}  Mean_best={best_avg:.3f}"
                  f"\nNow:    F1(best F1)={F1_f1:.3f}, F2(best F1)={F2_f1:.3f}\n\tF2(best F2)={F2_f2:.3f}, F1(best F2)={F1_f2:.3f}\n\tBest F1+F2={F1_s+F2_s:.3f}\n\tBest Mean: {avg_sum:.3f}")



if __name__ == "__main__":
    main()
