import torch
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt

# Import các module
from config import SystemConfig
from env import MOPVRPEnvironment
from model import MOPVRP_Actor, Critic
from visualizer import visualize_mopvrp
from file_loader import get_file_dataloader # File loader bạn đã tạo trước đó

# --- HÀM LƯU KẾT QUẢ RA JSON ---
def save_file_solution_json(env, dyn_truck, dyn_drone, filename="solution_from_file.json"):
    data = {}
    
    # Với file input, ta chỉ có 1 batch (index 0)
    b = 0
    batch_key = "file_instance"
    
    # 1. Lấy Metrics
    # Makespan
    t_times = dyn_truck[b, 1, :]
    d_times = dyn_drone[b, 1, :]
    makespan = max(t_times.max().item(), d_times.max().item())
    
    # Waiting Time (Lấy từ biến tích lũy trong Env)
    waiting_time = env.total_waiting_time[b].item()
    
    # Tính Objective (Giả sử w1=0.8, w2=0.2)
    w1, w2 = 0.5, 0.5
    if env.weights is not None:
        w1 = env.weights[b, 0].item()
        w2 = env.weights[b, 1].item()
    
    objective = w1 * makespan + w2 * waiting_time

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
    
    # 2. Lấy chi tiết Truck
    for t_idx, path in enumerate(env.routes[0][b]['trucks']):
        finish_time = dyn_truck[b, 1, t_idx].item()
        data[batch_key]["trucks"].append({
            "id": f"Truck_{t_idx}",
            "path": path,
            "finish_time": round(finish_time, 2)
        })
        
    # 3. Lấy chi tiết Drone
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
    print(f"💾 Đã lưu kết quả chi tiết vào: {filename}")
    return data[batch_key]["metrics"]

# --- HÀM CHẠY CHÍNH ---
def init_model(file_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys_config = SystemConfig('Truck_config.json', 'drone_linear_config.json', drone_type="4")
    loader = get_file_dataloader(file_path, device=DEVICE)

    env = MOPVRPEnvironment(sys_config, loader, device=DEVICE)
    model = MOPVRP_Actor(4, 2, 4, 128).to(DEVICE)
    critic = Critic(4, 2, 4, 128).to(DEVICE)
    
    # 3. Reset Environment
    state = env.reset()

    return env, model, critic, state
    

def load_checkpoint(model, critic, checkpoint_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Load model checkpoint"""
    # SỬA DÒNG NÀY: Thêm weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["actor_state_dict"])
    
    # self.actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    # self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    # self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    # self.best_reward = checkpoint['best_reward']
    
    # Nếu muốn load cả config cũ đè lên config mới (tùy chọn)
    # self.config = checkpoint['config'] 
    
    print(f"✅ Checkpoint loaded from {checkpoint_path}")
    return model, critic



def run_check_with_file(file_path, env, model, critic, state, pretrained):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load Dữ liệu từ File
    print(f"\n📂 Đang đọc dữ liệu từ: {file_path}")
    # Gọi Loader đặc biệt cho file txt
    
    # env = MOPVRPEnvironment(sys_config, loader, device=DEVICE)
    # model = MOPVRP_Actor(4, 2, 4, 128).to(DEVICE)
    
    # # Reset Environment
    # state = env.reset()
    env, model, critic, state = init_model(file_path)

    static, dyn_truck, dyn_drone, mask_cust, mask_veh = state
    


    print(f"🚀 Bắt đầu mô phỏng...")
    print(f"   Nodes: {env.num_nodes} | Scale: {env.scale.item():.1f}m")

    # TEST CRITIC: DỰ ĐOÁN GIÁ TRỊ BAN ĐẦU
    critic.eval()
    predicted_value = 0.0
    with torch.no_grad():
        # Critic nhận vào: static, dynamic_truck, dynamic_drone
        # Output là giá trị dự đoán (Value) của trạng thái hiện tại
        # Lưu ý: Value này thường tương ứng với 'Discounted Returns'
        val = critic(static, dyn_truck, dyn_drone)
        predicted_value = val.item()
        
    print(f"\n🔮 [CRITIC] Dự đoán ban đầu: {predicted_value:.4f}")
    if not pretrained:
        print("   (Lưu ý: Model chưa train thì dự đoán này là ngẫu nhiên)")
    # ====================================================
    
    done = False
    step = 0
    
    # Vòng lặp Simulation
    while not done:
        step += 1
        
        # --- Logic Chọn Xe (Smart Random) ---
        # Chỉ chọn xe nào có mask=1 VÀ có đường đi hợp lệ
        valid_veh_indices = torch.where(mask_veh[0] == 1)[0]
        candidates = []
        
        for v_idx in valid_veh_indices:
            v_tensor = torch.tensor([v_idx], device=DEVICE)
            node_mask = env.get_valid_customer_mask(v_tensor)
            
            # Kiểm tra xem có node nào đi được không
            valid_nodes = torch.where(node_mask[0] == 1)[0]
            
            # Logic chống kẹt:
            # - Xe đi được đến khách (node > 0) -> Tốt
            # - Xe đi được về Depot (node 0) VÀ đang ở ngoài -> Tốt
            if v_idx < env.num_trucks: 
                loc = dyn_truck[0, 0, v_idx].item()
            else: 
                loc = dyn_drone[0, 0, v_idx - env.num_trucks].item()
                
            has_useful_move = False
            for n in valid_nodes:
                if n != 0: 
                    has_useful_move = True
                    break
                if n == 0 and loc != 0: 
                    has_useful_move = True
                    break
            
            if has_useful_move:
                candidates.append(v_idx.item())

        # Nếu không có candidate tốt (hiếm), fallback về random valid
        if not candidates: 
            candidates = valid_veh_indices.tolist()
            
        if not candidates:
            print("❌ DEADLOCK: Không còn xe nào đi được!")
            break
        
        # Chọn ngẫu nhiên xe
        selected_veh = torch.tensor([random.choice(candidates)], device=DEVICE)
        
        # Chọn Node ngẫu nhiên từ mask hợp lệ
        valid_node_mask = env.get_valid_customer_mask(selected_veh)
        valid_node_indices = torch.where(valid_node_mask[0] == 1)[0]
        
        if len(valid_node_indices) == 0:
            selected_node = torch.tensor([0], device=DEVICE)
        else:
            rand_node = valid_node_indices[torch.randint(0, len(valid_node_indices), (1,)).item()]
            selected_node = torch.tensor([rand_node], device=DEVICE)

        # Step Env
        next_state, _, done_tensor, _ = env.step(selected_veh, selected_node)
        
        # Update state
        static, dyn_truck, dyn_drone, mask_cust, mask_veh = next_state
        done = done_tensor.item()
        
        if step % 50 == 0:
            print(f"   Step {step}...")

    print(f"✅ Mô phỏng hoàn tất sau {step} bước.")
    
    # 5. Lưu kết quả & Hiển thị Metrics
    if pretrained == True:
        metrics = save_file_solution_json(env, dyn_truck, dyn_drone, "solution_20_10_1_pretrained.json")
    else:
        metrics = save_file_solution_json(env, dyn_truck, dyn_drone, "solution_20_10_1_random.json")
    
    print(f"\n📊 KẾT QUẢ:")
    print(f"   ⏱️  Makespan:     {metrics['makespan']} s")
    print(f"   ⏳  Waiting Time: {metrics['waiting_time']} s")
    print(f"   🎯  Objective:    {metrics['objective_score']}")
    
    # 6. Vẽ biểu đồ
    print("\n🎨 Đang vẽ biểu đồ...")
    title = f"Result: 20.10.4 | Makespan: {metrics['makespan']}s | Wait: {metrics['waiting_time']}s"
    
    # Lưu ý: static[0] chứa tọa độ Normalized (0-1), Visualizer vẽ theo tỉ lệ này là OK.
    visualize_mopvrp(
        static[0], 
        env.routes[0][0]['trucks'], 
        env.routes[0][0]['drones'], 
        pretrained,
        title=title
    )

def select_action(step, model, state, last_hh=None, deterministic=False):
    """
    Select action using current policy
    Returns: vehicle_idx, node_idx, logprob_veh, logprob_node, last_hh
    """
    static, dyn_truck, dyn_drone, mask_cust, mask_veh = state
    
    if mask_cust.sum(dim=1).eq(0).any():
        # Clone để không ảnh hưởng đến state gốc
        mask_cust = mask_cust.clone()
        # Tìm các dòng có tổng = 0
        zero_mask_indices = mask_cust.sum(dim=1) == 0
        # Mở Node 0 (Depot) cho các dòng đó
        mask_cust[zero_mask_indices, 0] = 1

    with torch.no_grad():
        # Get probabilities from actor
        veh_probs, node_probs, internal_veh_idx, last_hh = model(
            static, dyn_truck, dyn_drone,
            decoder_input=None,
            last_hh=last_hh,
            mask_customers=mask_cust,
            mask_vehicles=mask_veh
        )
    
    if torch.isnan(node_probs).any() or (node_probs.sum(dim=1) == 0).any():
        # Tạo một phân phối mặc định: 100% về Depot (Node 0)
        fallback_probs = torch.zeros_like(node_probs)
        fallback_probs[:, 0] = 1.0
        
        # Tìm các dòng bị lỗi (NaN hoặc Sum=0)
        invalid_rows = torch.isnan(node_probs).any(dim=1) | (node_probs.sum(dim=1) == 0)
        
        # Gán đè phân phối mặc định vào các dòng lỗi
        node_probs[invalid_rows] = fallback_probs[invalid_rows]

    # Tương tự cho Vehicle Probs (Phòng hờ)
    if torch.isnan(veh_probs).any() or (veh_probs.sum(dim=1) == 0).any():
        fallback_veh = torch.zeros_like(veh_probs)
        fallback_veh[:, 0] = 1.0 # Chọn xe đầu tiên
        invalid_rows_veh = torch.isnan(veh_probs).any(dim=1) | (veh_probs.sum(dim=1) == 0)
        veh_probs[invalid_rows_veh] = fallback_veh[invalid_rows_veh]

    veh_idx = internal_veh_idx
    if deterministic:
        # Greedy selection
        # veh_idx = torch.argmax(veh_probs, dim=1)
        
        node_idx = torch.argmax(node_probs, dim=1)
    else:
        # Stochastic sampling
        veh_dist = torch.distributions.Categorical(veh_probs)
        node_dist = torch.distributions.Categorical(node_probs)
        
        veh_idx = veh_dist.sample()
        node_idx = node_dist.sample()
    
    print(f"Step {step}: Chọn Xe {veh_probs} | Node {node_probs}")

    # Calculate log probabilities
    logprob_veh = torch.log(veh_probs.gather(1, veh_idx.unsqueeze(1)) + 1e-10).squeeze(1)
    logprob_node = torch.log(node_probs.gather(1, node_idx.unsqueeze(1)) + 1e-10).squeeze(1)

    if step == 40:
        return
    
    return veh_idx, node_idx, logprob_veh, logprob_node, last_hh


def run_check_with_file(file_path, env, model, critic, state, pretrained):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Unpack state ban đầu
    static, dyn_truck, dyn_drone, mask_cust, mask_veh = state

    print(f"\n🚀 Bắt đầu mô phỏng với file: {os.path.basename(file_path)}")
    print(f"   Nodes: {env.num_nodes} | Scale: {env.scale.item():.1f}m")

    # 1. Chuyển sang chế độ đánh giá (Evaluation Mode)
    model.eval()
    critic.eval()
    
    # 2. Test Critic: Dự đoán giá trị ban đầu
    predicted_value = 0.0
    if pretrained:
        with torch.no_grad():
            # Critic đoán xem từ trạng thái này sẽ tốn bao nhiêu cost (hoặc reward)
            val = critic(static, dyn_truck, dyn_drone)
            predicted_value = val.item()
        print(f"🔮 [CRITIC] Dự đoán Reward tổng: {predicted_value:.4f}")

    done = False
    step = 0
    last_hh = None  # Hidden state ban đầu cho RNN trong Actor
    
    # 3. Vòng lặp mô phỏng (Simulation Loop)
    while not done:
        step += 1
        
        # --- TRƯỜNG HỢP 1: DÙNG MODEL ĐÃ TRAIN ---
        if pretrained:
            with torch.no_grad():
                # Gọi hàm act ta vừa viết ở trên
                # deterministic=True để lấy kết quả tối ưu nhất (Greedy)
                veh_idx, node_idx, logprob_veh, logprob_node, last_hh = select_action(step, model, state, last_hh, deterministic=False)  

            valid_mask = env.get_valid_customer_mask(veh_idx)
            print(f"Valid Mask for selected vehicle {veh_idx.item()}: {valid_mask}")
            invalid_nodes = (valid_mask.gather(1, node_idx.unsqueeze(1)) == 0).squeeze(1) 
            if invalid_nodes.any():
                node_idx = torch.where(invalid_nodes, torch.zeros_like(node_idx), node_idx)
            
            static, dyn_truck, dyn_drone, mask_cust, mask_veh = state
            selected_veh = veh_idx
            selected_node = node_idx
            print(f"Chọn Xe {selected_veh.item()} | Node {selected_node.item()}")
        
        # --- TRƯỜNG HỢP 2: DÙNG RANDOM (MODEL CHƯA TRAIN) ---
        else:
            valid_veh_indices = torch.where(mask_veh[0] == 1)[0]
            candidates = []
            for v_idx in valid_veh_indices:
                v_tensor = torch.tensor([v_idx], device=DEVICE)
                node_mask = env.get_valid_customer_mask(v_tensor)
                
                # Kiểm tra xem có node nào đi được không
                valid_nodes = torch.where(node_mask[0] == 1)[0]
                
                # Logic chống kẹt:
                # - Xe đi được đến khách (node > 0) -> Tốt
                # - Xe đi được về Depot (node 0) VÀ đang ở ngoài -> Tốt
                if v_idx < env.num_trucks: 
                    loc = dyn_truck[0, 0, v_idx].item()
                else: 
                    loc = dyn_drone[0, 0, v_idx - env.num_trucks].item()
                    
                has_useful_move = False
                for n in valid_nodes:
                    if n != 0: 
                        has_useful_move = True
                        break
                    if n == 0 and loc != 0: 
                        has_useful_move = True
                        break
                
                if has_useful_move:
                    candidates.append(v_idx.item())

            # Nếu không có candidate tốt (hiếm), fallback về random valid
            if not candidates: 
                candidates = valid_veh_indices.tolist()
                
            if not candidates:
                print("❌ DEADLOCK: Không còn xe nào đi được!")
                break
            
            # Chọn ngẫu nhiên xe
            selected_veh = torch.tensor([random.choice(candidates)], device=DEVICE)
            
            # Chọn Node ngẫu nhiên từ mask hợp lệ
            valid_node_mask = env.get_valid_customer_mask(selected_veh)
            valid_node_indices = torch.where(valid_node_mask[0] == 1)[0]
            
            if len(valid_node_indices) == 0:
                selected_node = torch.tensor([0], device=DEVICE)
            else:
                rand_node = valid_node_indices[torch.randint(0, len(valid_node_indices), (1,)).item()]
                selected_node = torch.tensor([rand_node], device=DEVICE)

        # --- BƯỚC 4: TƯƠNG TÁC VỚI MÔI TRƯỜNG ---
        next_state, _, done_tensor, _ = env.step(selected_veh, selected_node)
        
        # Cập nhật state cho vòng lặp sau
        static, dyn_truck, dyn_drone, mask_cust, mask_veh = next_state
        state = next_state
        done = done_tensor.item()
        
        # Log tiến độ để đỡ sốt ruột
        # if step % 50 == 0:
        #     print(f"   Step {step}... (Veh: {selected_veh.item()}, Node: {selected_node.item()})")

    print(f"✅ Mô phỏng hoàn tất sau {step} bước.")
    
    # --- BƯỚC 5: LƯU VÀ ĐÁNH GIÁ KẾT QUẢ ---
    out_name = f"solution_{'pretrained' if pretrained else 'random'}.json"
    metrics = save_file_solution_json(env, dyn_truck, dyn_drone, out_name)
    
    # Đánh giá độ lệch của Critic
    objective_score = metrics['objective_score']
    
    # QUAN TRỌNG: Thay số 1000.0 bằng hệ số scale reward thực tế trong Env của bạn
    REWARD_SCALE = 50000.0 
    actual_value = -objective_score / REWARD_SCALE
    
    print(f"\n📊 KẾT QUẢ SO SÁNH:")
    print(f"   🎯 Objective Thực tế:  {objective_score:.2f}")
    if pretrained:
        print(f"   🔮 Critic Dự đoán:     {predicted_value:.4f}")
        print(f"   📉 Sai số (Error):     {abs(predicted_value - actual_value):.4f}")
    
    # Vẽ đồ thị
    title = f"Obj: {metrics['objective_score']:.1f}"
    if pretrained:
        title += f" | Pred: {predicted_value:.2f}"
        
    visualize_mopvrp(static[0], env.routes[0][0]['trucks'], env.routes[0][0]['drones'], pretrained, title=title)



if __name__ == "__main__":
    # Thay tên file txt của bạn vào đây
    input_file = "../data/random_data/20.10.1.txt" 
    checkpoint_path = "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net/runs/01makespan/checkpoints/checkpoint_epoch_49.pth"  # Đường dẫn checkpoint nếu có
    LOAD_CHECKPOINTs = [True]

    for LOAD_CHECKPOINT in LOAD_CHECKPOINTs:
        print(f"\n=================== RUN WITH LOAD_CHECKPOINT={LOAD_CHECKPOINT} ===================")
        env, model, critic, state = init_model(input_file)

        if LOAD_CHECKPOINT:
            model, critic = load_checkpoint(model, critic, checkpoint_path)
        else:
            print("⚠️ Bỏ qua bước load checkpoint, sử dụng weights ngẫu nhiên.")

        
        if os.path.exists(input_file):
            run_check_with_file(input_file, env, model, critic, state, LOAD_CHECKPOINT)
        else:
            print(f"❌ Lỗi: Không tìm thấy file '{input_file}'")
            # Tạo file mẫu để test nếu cần
            print("💡 Gợi ý: Hãy đảm bảo file .txt nằm cùng thư mục.")