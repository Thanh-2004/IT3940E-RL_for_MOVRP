import torch
import numpy as np
import os
import json
from torch.distributions import Categorical

# Import các module
from config import SystemConfig
from environment import MOPVRPEnvironment
from model import MOPVRP_Actor
from dataloader import get_rl_dataloader
from visualizer import visualize_mopvrp

def ensure_configs():
    if not os.path.exists("Truck_config.json"):
        with open("Truck_config.json", "w") as f:
            json.dump({"V_max (m/s)": 15.0, "T (hour)": {"0-24": 1.0}}, f)
    if not os.path.exists("drone_linear_config.json"):
        with open("drone_linear_config.json", "w") as f:
            json.dump({"1": {
                "takeoffSpeed [m/s]": 5.0, "cruiseSpeed [m/s]": 20.0, 
                "landingSpeed [m/s]": 3.0, "cruiseAlt [m]": 50, 
                "capacity [kg]": 5.0, "batteryPower [Joule]": 1200000, 
                "beta(w/kg)": 20.0, "gama(w)": 150.0
            }}, f)

def main():
    ensure_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 INTEGRATED SIMULATION SYSTEM on {device}")
    
    # 1. Init Data
    print("\n🔹 [1/4] Generating Data...")
    batch_size = 2
    loader = get_rl_dataloader(batch_size, device)
    static, d_trucks, d_drones, _, _, scale, weights = next(iter(loader))
    
    # 2. Init Environment
    print("🔹 [2/4] Initializing Environment...")
    config_paths = {'truck': 'Truck_config.json', 'drone': 'drone_linear_config.json'}
    env = MOPVRPEnvironment(static, d_trucks, d_drones, weights, scale, config_paths, device)

    num_trucks = env.num_trucks
    num_drones = env.num_drones
    num_cust = static.size(2) - 1
    map_km = scale[0].item() / 1000

    print(f"   ► Kịch bản: {num_cust} Khách hàng | Map {map_km:.0f}x{map_km:.0f} km")
    print(f"   ► Đội xe: {num_trucks} Xe tải + {num_drones} Drone")

    # 3. Init Model
    print("🔹 [3/4] Initializing AI Agent...")
    model = MOPVRP_Actor(static_size=4, dynamic_size_truck=2, dynamic_size_drone=4, hidden_size=128).to(device)
    model.eval()
    model.perturb_weights(noise_scale=5.0) # Làm nhiễu để test đa dạng

    # --- TRACKING LỘ TRÌNH (Batch 0) ---
    truck_routes = [[0] for _ in range(num_trucks)] 
    drone_routes = [[0] for _ in range(num_drones)]
    
    # 4. Simulation Loop
    print("\n🔹 [4/4] STARTING SIMULATION...")
    print("=" * 60)
    
    done = False
    step = 0
    decoder_input = None 
    last_hh = None
    
    while not done:
        step += 1
        
        # A. Observe
        mask_cust, mask_veh = env.get_mask()
        curr_trucks, curr_drones = env.get_current_state()
        
        # B. Think
        with torch.no_grad():
            veh_probs, node_probs, last_hh = model(
                static, curr_trucks, curr_drones, 
                decoder_input, last_hh, mask_cust, mask_veh
            )
            # Dùng Sampling để test độ bao phủ không gian
            dist_veh = Categorical(veh_probs)
            dist_node = Categorical(node_probs)
            veh_action = dist_veh.sample()
            node_action = dist_node.sample()
        
        # C. Act
        rewards, dones = env.step(veh_action, node_action)
        
        # D. Update Input
        batch_idx = torch.arange(batch_size, device=device)
        sel_x = static[batch_idx, 0, node_action].unsqueeze(1)
        sel_y = static[batch_idx, 1, node_action].unsqueeze(1)
        decoder_input = torch.stack([sel_x, sel_y], dim=1) 
        
        # --- GHI NHẬN LỘ TRÌNH & LOGGING (Batch 0) ---
        act_veh_0 = veh_action[0].item()
        act_node_0 = node_action[0].item()
        b = 0
        
        # Ghi vào list lộ trình
        # Lưu ý: Agent giờ chỉ chọn Khách, không chủ động chọn 0 (trừ khi drone cần sạc)
        if act_veh_0 < num_trucks:
            if len(truck_routes[act_veh_0]) == 0 or truck_routes[act_veh_0][-1] != act_node_0:
                truck_routes[act_veh_0].append(act_node_0)
        else:
            d_idx = act_veh_0 - num_trucks
            if len(drone_routes[d_idx]) == 0 or drone_routes[d_idx][-1] != act_node_0:
                drone_routes[d_idx].append(act_node_0)
                # Drone vẫn giữ logic star-shaped trong Env: nếu đi khách, nó tự bay về.
                # Nhưng để visual đẹp, ta cứ ghi nhận điểm đến là được.
        
        # Logging
        v_type = "🚛 Truck" if act_veh_0 < num_trucks else "🚁 Drone"
        l_idx = act_veh_0 if act_veh_0 < num_trucks else act_veh_0 - num_trucks
        
        conf_v = veh_probs[0, act_veh_0].item()
        conf_n = node_probs[0, act_node_0].item()
        
        if act_node_0 != 0:
            print(f"📍 Step {step}: {v_type} {l_idx} --> Node {act_node_0}")
            if "Drone" in v_type:
                rem_e = env.drone_state[b, 2, l_idx].item()
                used_j = (1.0 - rem_e) * env.sys_config.drone_max_energy
                print(f"   🔋 Pin: {rem_e*100:.1f}% (Vừa tốn ~{used_j/1000:.1f} kJ)")
            else:
                acc_t = env.truck_state[b, 1, l_idx].item()
                print(f"   ⏱️ Thời gian: {acc_t:.1f}s")
        else:
            print(f"📍 Step {step}: {v_type} {l_idx} --> 🏠 Về Depot (Sạc/Chờ)")

        if dones.all():
            print("\n✅ MISSION ACCOMPLISHED! All customers served.")
            break
        
        if step > num_cust * 2:
            print("\n⚠️ Simulation limit reached.")
            break

    # 5. POST-PROCESSING (VISUALIZATION FIX)
    # ---------------------------------------------------------
    # Vì môi trường tự động tính đường về khi xong việc,
    # ta cần thêm Node 0 vào cuối mỗi lộ trình để biểu đồ khép kín.
    print("\n🔄 Finalizing routes for visualization...")
    
    # Fix Truck Routes
    for route in truck_routes:
        if len(route) > 0 and route[-1] != 0:
            route.append(0) # Vẽ đường về Depot
            
    # Fix Drone Routes
    for route in drone_routes:
        if len(route) > 0 and route[-1] != 0:
            route.append(0) # Drone luôn phải về

    # Stats
    makespan = max(env.truck_state[0, 1].max(), env.drone_state[0, 1].max()).item()
    unserved = (~env.visited[0, 1:]).sum().item()
    
    print("-" * 60)
    print(f"📊 RESULT (Batch 0):")
    print(f"   - Makespan: {makespan/60:.1f} min")
    print(f"   - Unserved: {unserved}")
    
    print("🎨 Visualizing...")
    visualize_mopvrp(
        static[0], 
        truck_routes, 
        drone_routes, 
        title=f"Result: {unserved} Unserved | Makespan: {makespan/60:.1f} min"
    )

if __name__ == "__main__":
    main()