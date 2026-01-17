import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

class MOPVRPGenerator(IterableDataset):
    def __init__(self, batch_size=32, device='cpu'):
        super(MOPVRPGenerator, self).__init__()
        self.batch_size = batch_size
        self.device = device
        
        # Cấu hình các kịch bản (Profiles) dựa trên file mẫu của bạn
        # (Số khách, Số Staff, Số Drone, Phạm vi tọa độ)
        self.configs = [
            {'n': 6,   'staff': 1,  'drone': 1, 'scale': 5000.0},
            {'n': 10,  'staff': 2,  'drone': 1, 'scale': 8000.0},
            {'n': 20,  'staff': 2,  'drone': 2, 'scale': 10000.0},
            {'n': 50,  'staff': 4,  'drone': 2, 'scale': 20000.0},
            {'n': 100, 'staff': 4,  'drone': 4, 'scale': 35000.0},
            {'n': 200, 'staff': 10, 'drone': 4, 'scale': 40000.0}
        ]

    def _generate_instance(self, cfg):
        """Sinh 1 batch dữ liệu theo cấu hình cfg"""
        batch_size = self.batch_size
        num_customers = cfg['n']
        num_nodes = num_customers + 1
        num_trucks = cfg['staff']
        num_drones = cfg['drone']
        map_scale = cfg['scale']
        
        # --- 1. Static Data (Bản đồ) ---
        # Shape: (Batch, 4, Num_Nodes)
        # Feature 0, 1: X, Y (Normalized về 0-1 để Model dễ học)
        # Feature 2: Demand
        # Feature 3: Truck Only Flag
        static = torch.zeros(batch_size, 4, num_nodes, device=self.device)
        
        # Tọa độ: Random [0, 1]
        static[:, 0:2, :] = torch.rand(batch_size, 2, num_nodes, device=self.device)
        
        # Depot: Luôn ở trung tâm (0.5, 0.5) hoặc random
        # Để giống file mẫu (depot có thể âm dương), ta cứ để random [0,1] rồi scale sau
        # Nhưng trong logic xe, depot thường là node 0
        
        # Demand: Random nhỏ [0.01, 0.1] như file mẫu
        static[:, 2, 1:] = torch.rand(batch_size, num_customers, device=self.device) * 0.09 + 0.01
        
        # Truck Only: Xác suất 20-30%
        truck_prob = 0.3
        static[:, 3, 1:] = (torch.rand(batch_size, num_customers, device=self.device) < truck_prob).float()
        
        # --- 2. Dynamic Data ---
        # Trucks: [Loc, Time]
        dynamic_trucks = torch.zeros(batch_size, 2, num_trucks, device=self.device)
        
        # Drones: [Loc, Time, Energy, Payload]
        dynamic_drones = torch.zeros(batch_size, 4, num_drones, device=self.device)
        dynamic_drones[:, 2, :] = 1.0 # Full pin (Normalized)
        
        # --- 3. Masks ---
        mask_customers = torch.ones(batch_size, num_nodes, device=self.device)
        mask_customers[:, 0] = 0 # Depot không cần phục vụ
        
        mask_vehicles = torch.ones(batch_size, num_trucks + num_drones, device=self.device)
        
        # Trả về thêm tham số 'scale' để môi trường tính khoảng cách thực tế (km)
        scale_tensor = torch.full((batch_size, 1), map_scale, device=self.device)
        weights = torch.tensor([[0.5, 0.5]] * batch_size, device=self.device)
        
        return static, dynamic_trucks, dynamic_drones, mask_customers, mask_vehicles, scale_tensor, weights

    def __iter__(self):
        """Vòng lặp vô tận sinh dữ liệu cho RL"""
        while True:
            # Bước 1: Chọn ngẫu nhiên 1 kịch bản (Curriculum Learning)
            cfg = np.random.choice(self.configs)
            
            yield self._generate_instance(cfg)

# Hàm tiện ích để tạo DataLoader chuẩn của PyTorch
def get_rl_dataloader(batch_size=32, device='cpu'):
    dataset = MOPVRPGenerator(batch_size=batch_size, device=device)
    # Batch size để None vì dataset tự sinh batch
    return DataLoader(dataset, batch_size=None, batch_sampler=None)


if __name__ == "__main__":

    # 1. Cấu hình
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4  # Để nhỏ cho dễ nhìn log
    
    print(f"🚀 Bắt đầu kiểm tra DataLoader trên thiết bị: {DEVICE}")
    
    # 2. Khởi tạo Loader
    dataloader = get_rl_dataloader(batch_size=BATCH_SIZE, device=DEVICE)
    data_iter = iter(dataloader)
    
    # 3. Chạy thử 5 vòng lặp để xem kích thước thay đổi
    for i in range(1, 6):
        print(f"\n{'='*40}")
        print(f"📡 LẤY BATCH THỨ {i}")
        
        # Lấy dữ liệu từ dataloader
        static, dyn_trucks, dyn_drones, mask_cust, mask_veh, scale, weights = next(data_iter)
        
        # Trích xuất thông tin kích thước
        b_size, _, num_nodes = static.shape
        num_customers = num_nodes - 1
        num_trucks = dyn_trucks.shape[2]
        num_drones = dyn_drones.shape[2]
        
        # In thông tin kiểm tra
        print(f"🔹 Kịch bản (Scenario): {num_customers} Khách hàng")
        print(f"🔹 Đội xe: {num_trucks} Trucks + {num_drones} Drones")
        print(f"🔹 Phạm vi bản đồ thực (Scale): {scale[0].item():.0f} mét")
        
        print(f"\n🔍 Kiểm tra Shape Tensor:")
        print(f"   - Static Input:       {static.shape}  (Mong đợi: [{BATCH_SIZE}, 4, {num_nodes}])")
        print(f"   - Dynamic Trucks:     {dyn_trucks.shape}  (Mong đợi: [{BATCH_SIZE}, 2, {num_trucks}])")
        print(f"   - Dynamic Drones:     {dyn_drones.shape}  (Mong đợi: [{BATCH_SIZE}, 4, {num_drones}])")
        print(f"   - Mask Customers:     {mask_cust.shape}  (Mong đợi: [{BATCH_SIZE}, {num_nodes}])")
        print(f"   - Mask Vehicles:      {mask_veh.shape}  (Mong đợi: [{BATCH_SIZE}, {num_trucks + num_drones}])")
        
        # Kiểm tra tính chuẩn hóa dữ liệu
        max_coord = static[:, 0:2, :].max().item()
        min_coord = static[:, 0:2, :].min().item()
        
        print(f"\n📊 Kiểm tra giá trị:")
        print(f"   - Tọa độ Max: {max_coord:.4f} (Phải <= 1.0)")
        print(f"   - Tọa độ Min: {min_coord:.4f} (Phải >= 0.0)")
        
        if max_coord <= 1.0 and min_coord >= 0.0:
            print("   ✅ Dữ liệu đã được Normalize tốt.")
        else:
            print("   ❌ Cảnh báo: Dữ liệu chưa được Normalize!")

    print(f"\n{'='*40}")
    print("✅ Kiểm tra hoàn tất. DataLoader hoạt động đúng thiết kế RL.")