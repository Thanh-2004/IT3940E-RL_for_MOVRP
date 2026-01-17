import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Import các module
from config import SystemConfig
from dataloader import get_rl_dataloader
from env import MOPVRPEnvironment
from model import MOPVRP_Actor
from visualizer import visualize_mopvrp
from file_loader import get_file_dataloader # Nếu bạn muốn chạy từ file

# --- 1. Hàm lưu JSON (Đã cập nhật Metrics) ---
def save_solution_to_json(env, dyn_truck, dyn_drone, filename="solution_metrics.json"):
    data = {}
    
    # Duyệt qua từng batch
    for b in range(env.batch_size):
        batch_key = f"batch_{b}"
        
        # A. TÍNH METRICS
        # 1. Makespan: Max thời gian của tất cả xe
        t_times = dyn_truck[b, 1, :]
        d_times = dyn_drone[b, 1, :]
        makespan = max(t_times.max().item(), d_times.max().item())
        
        # 2. Waiting Time: Lấy trực tiếp từ Environment (đã tích lũy chính xác)
        # env.total_waiting_time là tensor (B,)
        waiting_time = env.total_waiting_time[b].item()
        
        # 3. Objective Value (Hàm mục tiêu theo trọng số)
        # Giả sử w1=0.8, w2=0.2 (hoặc lấy từ env.weights nếu dynamic)
        w1, w2 = 0.8, 0.2
        if env.weights is not None:
            w1 = env.weights[b, 0].item()
            w2 = env.weights[b, 1].item()
        
        objective = w1 * makespan + w2 * waiting_time

        # B. CẤU TRÚC JSON
        data[batch_key] = {
            "metrics": {
                "makespan": round(makespan, 2),
                "waiting_time": round(waiting_time, 2),
                "objective_score": round(objective, 2),
                "weights": {"w1": round(w1, 2), "w2": round(w2, 2)}
            },
            "trucks": [],
            "drones": []
        }
        
        # C. CHI TIẾT XE
        # Trucks
        for t_idx, path in enumerate(env.routes[0][b]['trucks']):
            finish_time = dyn_truck[b, 1, t_idx].item()
            data[batch_key]["trucks"].append({
                "id": f"Truck_{t_idx}",
                "path": path,
                "finish_time": round(finish_time, 2)
            })
            
        # Drones
        for d_idx, path in enumerate(env.routes[0][b]['drones']):
            finish_time = dyn_drone[b, 1, d_idx].item()
            energy = dyn_drone[b, 2, d_idx].item()
            data[batch_key]["drones"].append({
                "id": f"Drone_{d_idx}",
                "path": path,
                "finish_time": round(finish_time, 2),
                "energy_left": f"{energy*100:.1f}%"
            })
            
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Đã lưu kết quả (Makespan & Waiting Time) vào: {filename}")

# --- 2. Hàm chạy kiểm tra ---
def check_full_metrics():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys_config = SystemConfig('Truck_config.json', 'drone_linear_config.json', drone_type="1")

    
    # Load Data (Dùng FileLoader hoặc DataLoader thường)
    # Ở đây dùng loader thường để test nhanh, bạn có thể đổi sang get_file_dataloader
    loader = get_rl_dataloader(batch_size=1, device=DEVICE) 
    
    env = MOPVRPEnvironment(sys_config, loader, device=DEVICE)
    model = MOPVRP_Actor(4, 2, 4, 128).to(DEVICE)
    
    # Reset
    state = env.reset()
    static, dyn_truck, dyn_drone, mask_cust, mask_veh = state
    
    print(f"\n🚀 BẮT ĐẦU MÔ PHỎNG VỚI METRICS")
    done = False
    step = 0
    
    while not done:
        step += 1
        
        # --- Logic Chọn Action (Smart Random) ---
        valid_veh_indices = torch.where(mask_veh[0] == 1)[0]
        candidates = []
        for v_idx in valid_veh_indices:
            v_tensor = torch.tensor([v_idx], device=DEVICE)
            node_mask = env.get_valid_customer_mask(v_tensor)
            
            # Logic check deadlock đơn giản
            if v_idx < env.num_trucks: loc = dyn_truck[0, 0, v_idx].item()
            else: loc = dyn_drone[0, 0, v_idx - env.num_trucks].item()
            
            valid_nodes = torch.where(node_mask[0] == 1)[0]
            # Có node đi được (khác 0) HOẶC (về 0 nếu đang ở ngoài)
            useful = any(n != 0 for n in valid_nodes) or (loc != 0 and 0 in valid_nodes)
            if useful: candidates.append(v_idx.item())

        if not candidates: candidates = valid_veh_indices.tolist() # Fallback
        if not candidates: break # Deadlock
        
        import random
        selected_veh = torch.tensor([random.choice(candidates)], device=DEVICE)
        
        # Chọn Node
        valid_node_mask = env.get_valid_customer_mask(selected_veh)
        valid_node_indices = torch.where(valid_node_mask[0] == 1)[0]
        if len(valid_node_indices) == 0: selected_node = torch.tensor([0], device=DEVICE)
        else: selected_node = torch.tensor([valid_node_indices[torch.randint(0, len(valid_node_indices), (1,)).item()]], device=DEVICE)

        # Step
        next_state, _, done_tensor, _ = env.step(selected_veh, selected_node)
        static, dyn_truck, dyn_drone, mask_cust, mask_veh = next_state
        done = done_tensor.item()
        
    print(f"✅ Kết thúc sau {step} bước.")
    
    # --- 3. LƯU & HIỂN THỊ METRICS ---
    save_solution_to_json(env, dyn_truck, dyn_drone, "final_solution.json")
    
    # In ra màn hình để kiểm tra nhanh
    with open("final_solution.json", 'r') as f:
        res = json.load(f)
        m = res['batch_0']['metrics']
        print(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
        print(f"   ⏱️  Makespan:      {m['makespan']} s")
        print(f"   ⏳  Waiting Time:  {m['waiting_time']} s")
        print(f"   🎯  Objective:     {m['objective_score']}")

    # Vẽ hình
    visualize_mopvrp(
        static[0], 
        env.routes[0][0]['trucks'], 
        env.routes[0][0]['drones'], 
        title=f"MOPVRP | Makespan: {m['makespan']}s | Wait: {m['waiting_time']}s"
    )

if __name__ == "__main__":
    check_full_metrics()