#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import sys
import numpy as np
import random
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.config import build_config_from_files
from environment.env import ParallelVRPDEnvironment          
from model.models_v2 import MOPVRP

# --- Lớp hỗ trợ ghi log song song ra console và file ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _ensure_batch(obs, device):
    keys = ["graph_ctx", "trucks_ctx", "drones_ctx", "mask_trk", "mask_dr", "weights"]
    out = {}
    for k in keys:
        x = obs[k]
        if torch.is_tensor(x):
            x = x.to(device)
            if k == "weights":
                if x.dim() == 1: x = x.unsqueeze(0)
            else:
                if x.dim() == 2: x = x.unsqueeze(0)
        out[k] = x
    return out

@torch.no_grad()
def eval_once(env, policy, device, w1, w2):
    env.configure_fixed_weights(w1, w2)
    obs = env.reset()
    obs = _ensure_batch(obs, device=device)
    done = False
    info_last = None
    actions_history = [] 
    
    h = policy.init_hidden(1, device=device)
    while not done:
        action, _, _, _, h = policy.act(obs, h_prev=h, deterministic=False)
        v_type, v_idx, chosen_node = tuple(int(x) for x in action)
        
        actions_history.append((v_type, v_idx, chosen_node))
        step = env.step((v_type, v_idx, chosen_node))
        info_last = step.info
        done = bool(step.done.item()) if torch.is_tensor(step.done) else bool(step.done)
        obs = _ensure_batch(step.obs, device=device)

    for k in range(env.num_trucks):
        curr_node = int(env.truck_position[k].item())
        if curr_node != 0:
            actions_history.append((0, k, 0))
            step = env.step((0, k, 0))
            info_last = step.info


    for d in range(env.num_drones):
        curr_node_dr = int(env.drone_position[d].item())
        if curr_node_dr != 0:
            actions_history.append((1, d, 0))
            step = env.step((1, d, 0))
            info_last = step.info

    return float(info_last["F1"]), float(info_last["F2"]), actions_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truck_json", required=True)
    ap.add_argument("--drone_json", required=True)
    ap.add_argument("--customers_txt", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid_step", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, default="./simulator_inputs")
    args = ap.parse_args()

    # Tạo thư mục đầu ra trước để lưu log
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Bắt đầu ghi log
    sys.stdout = Logger(out_dir / "evaluation_summary.txt")

    device = torch.device(args.device)
    set_seed(42)

    p = Path(args.ckpt_path)
    ckpt_files = sorted(list(p.glob("**/*.pt")) + list(p.glob("**/*.pth")))
    if not ckpt_files:
        print(f"Error: No .pt files found in {args.ckpt_path}")
        return

    cfg, _ = build_config_from_files(args.truck_json, args.drone_json, args.customers_txt)  
    env = ParallelVRPDEnvironment(cfg, maintain_trip_cycles=True, parallel_mode=False, enable_idle_action=False,
                   max_drone_trips=200, normalize_targets=True,
                   weight_strategy="fixed", max_episode_steps=2048, device=device)
    
    policy = MOPVRP(static_size=7, dyn_truck_size=5, dyn_drone_size=12,
                                  hidden_size=128, device=device).to(device)

    w1_list = np.round(np.arange(0.0, 1.0 + 1e-9, args.grid_step), 2)
    best_results = { w1: {"sum": float('inf'), "f1": 0.0, "f2": 0.0, "history": [], "ckpt": ""} for w1 in w1_list }

    print(f"Evaluating {len(ckpt_files)} checkpoints across {len(w1_list)} weight settings...")

    for ck_path in tqdm(ckpt_files, desc="Scanning"):
        state_dict = torch.load(ck_path, map_location=device)
        policy.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict, strict=False)
        policy.eval()

        for w1 in w1_list:
            w2 = float(np.round(1.0 - w1, 2))
            f1, f2, history = eval_once(env, policy, device, w1, w2)
            curr_sum = f1 + f2
            
            if curr_sum < best_results[w1]["sum"]:
                best_results[w1] = {"sum": curr_sum, "f1": f1, "f2": f2, "history": history, "ckpt": ck_path.stem}

    # In bảng kết quả ra console (đồng thời ghi vào file summary)
    print("\n" + "="*75)
    print(f"{'W1':<6} | {'F1':<12} | {'F2':<12} | {'Sum (F1+F2)':<12} | {'Source Checkpoint'}")
    print("-" * 75)

    for w1, data in best_results.items():
        if not data["history"]: continue
        
        print(f"{w1:<6.2f} | {data['f1']:<12.4f} | {data['f2']:<12.4f} | {data['sum']:<12.4f} | {data['ckpt']}")

        # Tách file theo yêu cầu: chỉ chứa logic weight và checkpoint
        file_name = f"w{w1:.2f}_{data['ckpt']}.txt"
        file_path = out_dir / file_name
        
        with open(file_path, "w", encoding="utf-8") as f:
            for v_type, v_id, node in data["history"]:
                f.write(f"{v_type} {v_id} {node}\n")
    
    print("="*75)
    print(f"Success! Detailed solutions and 'evaluation_summary.txt' saved in: {out_dir}")

if __name__ == "__main__":
    main()