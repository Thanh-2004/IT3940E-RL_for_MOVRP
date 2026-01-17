import torch
import numpy as np
import os
import json
from config import SystemConfig
from dataloader import get_rl_dataloader
from env import MOPVRPEnvironment
from model import MOPVRP_Actor

# --- Setup Config Giả (như cũ) ---
def create_dummy_configs():
    truck_cfg = { "T (hour)": {"0-5": 0.8, "6-24": 1.0}, "V_max (m/s)": 15.0 }
    drone_cfg = { "1": { "batteryPower [Joule]": 500000, "cruiseSpeed [m/s]": 20.0, "capacity [kg]": 5.0, 
                         "beta(w/kg)": 15.0, "gama(w)": 300.0, "cruiseAlt [m]": 50.0, 
                         "takeoffSpeed [m/s]": 5.0, "landingSpeed [m/s]": 3.0 }}
    with open('dummy_truck.json', 'w') as f: json.dump(truck_cfg, f)
    with open('dummy_drone.json', 'w') as f: json.dump(drone_cfg, f)

def check_full_flow():
    create_dummy_configs()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Init
    sys_config = SystemConfig('Truck_config.json', 'drone_linear_config.json', drone_type="1")
    loader = get_rl_dataloader(batch_size=1, device=DEVICE) # Test 1 batch cho dễ nhìn
    env = MOPVRPEnvironment(sys_config, loader, device=DEVICE)
    model = MOPVRP_Actor(4, 2, 4, 128).to(DEVICE)

    # 2. Reset
    state = env.reset()
    static, dyn_truck, dyn_drone, mask_cust, mask_veh = state


    num_trucks = env.num_trucks
    num_drones = env.num_drones
    num_cust = static.size(2) - 1
    map_km = env.scale[0].item() / 1000

    print(f"   ► Kịch bản: {num_cust} Khách hàng | Map {map_km:.0f}x{map_km:.0f} km")
    print(f"   ► Đội xe: {num_trucks} Xe tải + {num_drones} Drone")
    
    
    print(f"\n{'='*40}")
    print(f"🚀 BẮT ĐẦU TEST CHU TRÌNH KÍN (CLOSED LOOP)")
    print(f"Node 0: Depot | Node 1..N: Customer")
    print(f"{'='*40}\n")
    
    done = False
    step = 0
    total_reward = 0
    
    while not done:
        step += 1
        print(f"--- STEP {step} ---")
        
        # A. Lấy Mask Valid từ Environment (Cực kỳ quan trọng)
        # Để đảm bảo ta không chọn node bậy (như Truck chọn Depot khi mới start)
        # Ta cần giả lập việc chọn xe trước. Ở đây test random xe luôn.
        
        # Để đơn giản hóa test: Ta gộp logits của tất cả xe và node
        # Trong thực tế: Model chọn xe -> Mask xe -> Chọn Node -> Mask Node
        
        # 1. Forward Model lấy Logits (chưa mask kỹ)
        veh_probs, node_probs, _ = model(static, dyn_truck, dyn_drone, mask_customers=mask_cust, mask_vehicles=mask_veh)
        
        
        # 2. CHỌN XE THÔNG MINH (Smart Vehicle Selection)
        # Chỉ chọn những xe có khả năng phục vụ khách (hoặc về Depot nếu cần)
        valid_veh_indices = torch.where(mask_veh[0] == 1)[0]
        
        candidates = []
        for v_idx in valid_veh_indices:
            v_tensor = torch.tensor([v_idx], device=DEVICE)
            # Lấy mask node cho xe này
            node_mask = env.get_valid_customer_mask(v_tensor)
            valid_nodes = torch.where(node_mask[0] == 1)[0]
            
            # Logic lọc ứng viên:
            # - Nếu còn khách unvisited: Ưu tiên xe đi được đến Node > 0
            # - Nếu xe chỉ đi được đến Node 0 (Về sạc/nghỉ): 
            #   Chỉ chấp nhận nếu xe đó ĐANG KHÔNG Ở Node 0 (tức là đang ở ngoài cần về).
            #   Nếu đang ở Node 0 mà chỉ đi được đến Node 0 -> Bỏ qua (Đừng chọn nó làm gì)
            
            # Check vị trí hiện tại
            if v_idx < env.num_trucks:
                curr_loc = dyn_truck[0, 0, v_idx].item()
            else:
                curr_loc = dyn_drone[0, 0, v_idx - env.num_trucks].item()
                
            has_useful_move = False
            for n in valid_nodes:
                if n != 0: # Đi khách -> Tốt
                    has_useful_move = True
                    break
                if n == 0 and curr_loc != 0: # Về nhà -> Tốt
                    has_useful_move = True
                    break
            
            if has_useful_move:
                candidates.append(v_idx.item())

        # Nếu không còn candidate nào "có ích", nhưng game chưa Done -> Deadlock?
        # Lúc này mới fallback chọn đại để environment xử lý (có thể chờ)
        if len(candidates) == 0:
             # Fallback: Lấy valid_veh_indices gốc
             candidates = valid_veh_indices.tolist()
             if len(candidates) == 0:
                 print("❌ DEADLOCK THỰC SỰ: Không xe nào hoạt động!")
                 break

        # Chọn ngẫu nhiên từ candidates đã lọc
        import random
        rand_idx = random.choice(candidates)
        selected_veh = torch.tensor([rand_idx], device=DEVICE)
        
        # 3. Chọn NODE (Giữ nguyên logic cũ, nhưng giờ chắc chắn có node ngon)
        valid_node_mask = env.get_valid_customer_mask(selected_veh)
        valid_node_indices = torch.where(valid_node_mask[0] == 1)[0]
        
        # ... (Phần còn lại giữ nguyên)
        
        if len(valid_node_indices) == 0:
            # Trường hợp hiếm: Xe còn pin/active nhưng không đi đâu được (kẹt)
            # Chọn đại node 0 để xem Env xử lý sao (thường là đứng yên hoặc lỗi)
            selected_node = torch.tensor([0], device=DEVICE)
            print(f"   ⚠️ Xe {selected_veh.item()} bị kẹt, thử chọn Node 0...")
        else:
            # Chọn ngẫu nhiên 1 node hợp lệ
            rand_node_idx = torch.randint(0, len(valid_node_indices), (1,)).item()
            selected_node = torch.tensor([valid_node_indices[rand_node_idx]], device=DEVICE)

        # In thông tin hành động
        veh_id = selected_veh.item()
        node_id = selected_node.item()
        v_type = "Truck" if veh_id < env.num_trucks else "Drone"
        
        # Check xem Truck có đang về Depot không
        if v_type == "Truck" and node_id == 0:
            status = "🏠 GOING HOME (Kết thúc chuyến)"
        elif node_id == 0:
            status = "🔋 DRONE RECHARGE (Về sạc)"
        else:
            status = "📦 SERVING CUSTOMER"
            
        print(f"   Action: {v_type} {veh_id} -> Node {node_id} | {status}")

        # B. Step Environment
        next_state, reward, done_tensor, _ = env.step(selected_veh, selected_node)
        
        # C. Update State
        static, dyn_truck, dyn_drone, mask_cust, mask_veh = next_state
        done = done_tensor.item()
        total_reward += reward.item()
        
        # D. In trạng thái thời gian/năng lượng
        if v_type == "Truck":
            t = dyn_truck[0, 1, veh_id].item()
            print(f"     👉 Truck Time: {t:.1f}s")
        else: # Drone
            d_id = veh_id - env.num_trucks
            t = dyn_drone[0, 1, d_id].item()
            e = dyn_drone[0, 2, d_id].item()
            print(f"     👉 Drone Time: {t:.1f}s | Energy: {e:.1%}")

    print(f"\n{'='*40}")
    print("✅ HOÀN THÀNH MÔ PHỎNG")
    print(f"Tổng số bước: {step}")
    
    # In kết quả Makespan
    truck_times = dyn_truck[0, 1, :]
    drone_times = dyn_drone[0, 1, :]
    makespan = max(truck_times.max(), drone_times.max()).item()
    
    print(f"⏱️ Makespan cuối cùng: {makespan:.2f}s")
    print(f"Truck Times: {truck_times.tolist()}")
    print(f"Drone Times: {drone_times.tolist()}")
    
    # Kiểm tra xem Truck có về Depot không
    truck_locs = dyn_truck[0, 0, :]
    print(f"Vị trí cuối cùng của Trucks (Phải là 0): {truck_locs.tolist()}")
    
    if (truck_locs == 0).all():
        print("🎉 SUCCESS: Tất cả xe tải đã về Depot an toàn!")
    else:
        print("❌ FAILURE: Vẫn còn xe tải chưa về Depot.")

    os.remove('dummy_truck.json')
    os.remove('dummy_drone.json')

if __name__ == "__main__":
    check_full_flow()