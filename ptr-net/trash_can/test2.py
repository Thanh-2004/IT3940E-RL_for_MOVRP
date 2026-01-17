import torch
import numpy as np
import os
import json
import time
from torch.distributions import Categorical

# --- IMPORT CÁC MODULE ĐÃ TẠO ---
# Đảm bảo bạn đã lưu các file model.py, environment.py, config.py, dataloader.py
try:
    from model import MOPVRP_Actor
    from environment import MOPVRPEnvironment
    from dataloader import get_rl_dataloader
except ImportError as e:
    print("❌ Lỗi Import: Đảm bảo bạn đã có đủ 4 file: model.py, environment.py, config.py, dataloader.py")
    print(f"Chi tiết: {e}")
    exit(1)

# --- HELPER: TẠO CONFIG GIẢ (Nếu chưa có) ---
def ensure_configs():
    if not os.path.exists("Truck_config.json"):
        with open("Truck_config.json", "w") as f:
            json.dump({"V_max (m/s)": 15.0, "T (hour)": {"0-24": 1.0}}, f)
    if not os.path.exists("drone_linear_config.json"):
        with open("drone_linear_config.json", "w") as f:
            json.dump({"1": {
                "takeoffSpeed [m/s]": 5.0, "cruiseSpeed [m/s]": 20.0, "landingSpeed [m/s]": 3.0,
                "cruiseAlt [m]": 50, "capacity [kg]": 5.0,
                "batteryPower [Joule]": 1200000, 
                "beta(w/kg)": 20.0, "gama(w)": 150.0
            }}, f)

# --- MAIN SIMULATION ---
def main():
    ensure_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 BẮT ĐẦU MÔ PHỎNG TÍCH HỢP TRÊN: {device}")
    
    # 1. SETUP DỮ LIỆU & MÔI TRƯỜNG
    # ---------------------------------------------------------
    print("\n🔹 [1/3] Khởi tạo Dữ liệu & Môi trường...")
    batch_size = 2 # Chạy thử 2 kịch bản cùng lúc
    
    # Lấy 1 batch dữ liệu ngẫu nhiên từ DataLoader
    loader = get_rl_dataloader(batch_size, device)
    static, d_trucks, d_drones, _, _, scale, weights = next(iter(loader))
    
    # Config paths
    config_paths = {'truck': 'Truck_config.json', 'drone': 'drone_linear_config.json'}
    
    # Khởi tạo Environment
    env = MOPVRPEnvironment(static, d_trucks, d_drones, weights, scale, config_paths, device)
    
    # Thông tin kịch bản
    num_cust = static.size(2) - 1
    num_trucks = env.num_trucks
    num_drones = env.num_drones
    map_km = scale[0].item() / 1000
    print(f"   ► Kịch bản: {num_cust} Khách hàng | Map {map_km:.0f}x{map_km:.0f} km")
    print(f"   ► Đội xe: {num_trucks} Xe tải + {num_drones} Drone")

    # 2. KHỞI TẠO MÔ HÌNH (MODEL)
    # ---------------------------------------------------------
    print("\n🔹 [2/3] Khởi tạo Mô hình AI (Actor)...")
    # Input sizes: Static=4, Truck=2, Drone=4 (Khớp với dataloader)
    model = MOPVRP_Actor(static_size=4, dynamic_size_truck=2, dynamic_size_drone=4, hidden_size=128).to(device)
    model.eval()
    
    # [QUAN TRỌNG] Làm nhiễu trọng số để mô hình chọn hành động khác nhau
    # Nếu không có bước này, mô hình chưa train sẽ chọn xác suất đều nhau (Uniform)
    model.perturb_weights(noise_scale=5.0)

    # 3. VÒNG LẶP MÔ PHỎNG (INTERACTION LOOP)
    # ---------------------------------------------------------
    print("\n🔹 [3/3] CHẠY MÔ PHỎNG TƯƠNG TÁC...")
    print("=" * 70)
    
    done = False
    step = 0
    decoder_input = None # Bước đầu tiên chưa có input, dùng x0 của model
    last_hh = None       # Hidden state của LSTM
    
    total_reward = 0
    
    while not done:
        step += 1
        
        # --- BƯỚC A: QUAN SÁT (OBSERVE) ---
        # Lấy mask hợp lệ từ môi trường (đã tính toán pin, tải trọng)
        mask_cust, mask_veh = env.get_mask()
        curr_trucks, curr_drones = env.get_current_state()
        
        # --- BƯỚC B: SUY NGHĨ (THINK) ---
        with torch.no_grad():
            # Model tính toán xác suất (Forward Pass)
            veh_probs, node_probs, last_hh = model(
                static, curr_trucks, curr_drones, 
                decoder_input, last_hh, mask_cust, mask_veh
            )
            
            # # Chọn hành động (Greedy - Chọn xác suất cao nhất)
            # # Model đã được làm nhiễu nên sẽ có "chính kiến" riêng
            # veh_action = veh_probs.argmax(dim=1)
            # node_action = node_probs.argmax(dim=1)

            # --- CÁCH 2: SAMPLING (Mới - Chọn theo phân phối) ---
            # Tạo phân phối từ xác suất
            dist_veh = Categorical(veh_probs)
            dist_node = Categorical(node_probs)
            
            # Lấy mẫu (Sample) dựa trên xác suất (xác suất cao -> dễ được chọn hơn)
            veh_action = dist_veh.sample()
            node_action = dist_node.sample()
        
        # --- BƯỚC C: HÀNH ĐỘNG (ACT) ---
        # Gửi hành động vào môi trường để tính toán vật lý
        rewards, dones = env.step(veh_action, node_action)
        
        # --- BƯỚC D: CẬP NHẬT (UPDATE) ---
        # Chuẩn bị input cho bước tiếp theo (Auto-regressive)
        # Input t+1 = Tọa độ (x, y) của Node vừa chọn ở bước t
        batch_idx = torch.arange(batch_size, device=device)
        
        # Lấy tọa độ x, y từ static data dựa trên node_action
        sel_x = static[batch_idx, 0, node_action].unsqueeze(1) # (Batch, 1)
        sel_y = static[batch_idx, 1, node_action].unsqueeze(1) # (Batch, 1)
        
        # Ghép lại thành (Batch, 2, 1) để đưa vào Decoder
        decoder_input = torch.stack([sel_x, sel_y], dim=1) 
        
        # --- LOGGING (HIỂN THỊ KẾT QUẢ) ---
        # Chỉ in thông tin của Batch 0 để dễ nhìn
        b = 0
        v_idx = veh_action[b].item()
        n_idx = node_action[b].item()
        
        # Xác định loại xe
        v_type = "🚛 Truck" if v_idx < num_trucks else "🚁 Drone"
        local_idx = v_idx if v_idx < num_trucks else v_idx - num_trucks
        
        # Lấy độ tự tin của Model
        conf_v = veh_probs[b, v_idx].item()
        conf_n = node_probs[b, n_idx].item()
        
        if n_idx != 0:
            print(f"📍 Step {step}: {v_type} {local_idx} --> Node {n_idx}")
            print(f"   🧠 Model Confidence: Xe={conf_v*100:.1f}%, Khách={conf_n*100:.1f}%")
            
            # In thông số vật lý từ môi trường
            if "Drone" in v_type:
                # Lấy năng lượng còn lại
                rem_e = env.drone_state[b, 2, local_idx].item()
                # Tính lượng đã dùng (Joule)
                used_j = (1.0 - rem_e) * env.sys_config.drone_max_energy
                print(f"   🔋 Pin: {rem_e*100:.1f}% (Vừa tốn ~{used_j/1000:.1f} kJ)")
            else:
                # Lấy thời gian tích lũy
                acc_t = env.truck_state[b, 1, local_idx].item()
                print(f"   ⏱️ Thời gian: {acc_t:.1f}s")
        else:
            print(f"📍 Step {step}: {v_type} {local_idx} --> Đứng chờ/Về Depot (Node 0)")

        # Kiểm tra điều kiện dừng
        if dones.all():
            print("\n✅ TẤT CẢ KHÁCH HÀNG ĐÃ ĐƯỢC PHỤC VỤ!")
            total_reward = rewards[b].item()
            break
        
        # Safety break (tránh lặp vô hạn nếu model dở)
        if step > num_cust * 2:
            print("\n⚠️ Dừng sớm: Quá giới hạn bước chạy (Model chưa tối ưu nên đi lòng vòng)")
            break
            
    # 4. KẾT QUẢ CUỐI CÙNG
    # ---------------------------------------------------------
    makespan = max(env.truck_state[0, 1].max(), env.drone_state[0, 1].max()).item()
    print("=" * 70)
    print(f"📊 KẾT QUẢ MÔ PHỎNG (Batch 0):")
    print(f"   - Tổng thời gian (Makespan): {makespan/60:.2f} phút")
    print(f"   - Reward (Mục tiêu tối ưu):  {total_reward:.4f}")
    
    # Đếm số khách chưa được phục vụ (để kiểm tra tính đúng đắn)
    unserved = (~env.visited[0, 1:]).sum().item()
    if unserved == 0:
        print("   - Trạng thái: ✅ HOÀN THÀNH 100%")
    else:
        print(f"   - Trạng thái: ❌ CÒN SÓT {unserved} KHÁCH")

if __name__ == "__main__":
    main()