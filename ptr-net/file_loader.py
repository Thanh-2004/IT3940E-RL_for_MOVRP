import torch
import numpy as np

class FileDataset:
    def __init__(self, file_path, device='cpu'):
        self.device = device
        self.data = self._parse_file(file_path)
        
    def _parse_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            
        # 1. Parse Metadata
        # Giả định format cố định như file mẫu
        num_trucks = int(lines[0].strip().split()[1])
        num_drones = int(lines[1].strip().split()[1])
        flight_limit = float(lines[2].strip().split()[1]) # Có thể dùng update config
        num_customers = int(lines[3].strip().split()[1])
        
        # 2. Parse Nodes Data (Bỏ qua dòng Header thứ 5)
        # Format: X, Y, Demand, TruckOnly, SvcTruck, SvcDrone
        raw_nodes = []
        for line in lines[5:]:
            if not line.strip(): continue
            parts = list(map(float, line.strip().split()))
            raw_nodes.append(parts)
            
        raw_nodes = np.array(raw_nodes) # Shape (N_cust, 6)
        
        # 3. Add Depot (Giả sử Depot tại 0,0 và không có trong list customers)
        # Depot: X=0, Y=0, Dem=0, TruckOnly=0, Svc=0...
        depot_node = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        # Ghép lại: Node 0 là Depot, Node 1..N là Customer
        all_nodes = np.concatenate([depot_node, raw_nodes], axis=0)
        
        # 4. Normalize Coordinates (Min-Max Scaling)
        coords = all_nodes[:, 0:2] # (N+1, 2)
        
        min_xy = np.min(coords, axis=0)
        max_xy = np.max(coords, axis=0)
        diff_xy = max_xy - min_xy
        
        # Scale factor là cạnh lớn nhất của bounding box để giữ tỷ lệ aspect ratio
        scale = float(np.max(diff_xy))
        
        # Nếu scale = 0 (trường hợp chỉ có 1 điểm)
        if scale == 0: scale = 1.0
            
        # Normalize về [0, 1]
        # Công thức: (val - min) / scale
        # Lưu ý: Cần trừ min để đưa về gốc, nhưng để giữ vị trí tương đối
        # ta thường đưa min về 0.
        norm_coords = (coords - min_xy) / scale
        
        # Lưu lại offset để nếu cần visualizer vẽ lại tọa độ gốc
        self.offset = min_xy
        
        # 5. Construct Tensors
        batch_size = 1 # Chạy test 1 file -> Batch = 1
        num_nodes = len(all_nodes)
        
        # --- Static Tensor: [B, 4, N] ---
        # Feature 0,1: X, Y (Normalized)
        # Feature 2: Demand (Giữ nguyên hoặc Normalize nếu demand quá lớn)
        # Trong file demand 0.1 -> OK, giữ nguyên.
        # Feature 3: Truck Only (0 hoặc 1)
        
        static = torch.zeros(batch_size, 4, num_nodes, device=self.device)
        static[0, 0, :] = torch.tensor(norm_coords[:, 0]) # X
        static[0, 1, :] = torch.tensor(norm_coords[:, 1]) # Y
        static[0, 2, :] = torch.tensor(all_nodes[:, 2])   # Demand
        static[0, 3, :] = torch.tensor(all_nodes[:, 3])   # Truck Only
        
        # --- Dynamic Tensors ---
        # Truck: [B, 2, K] -> [Loc, Time]
        dyn_truck = torch.zeros(batch_size, 2, num_trucks, device=self.device)
        
        # Drone: [B, 4, D] -> [Loc, Time, Energy, Payload]
        dyn_drone = torch.zeros(batch_size, 4, num_drones, device=self.device)
        dyn_drone[:, 2, :] = 1.0 # Full Energy
        
        # --- Masks ---
        mask_cust = torch.ones(batch_size, num_nodes, device=self.device)
        mask_cust[:, 0] = 0 # Depot mask = 0
        
        mask_veh = torch.ones(batch_size, num_trucks + num_drones, device=self.device)
        
        # --- Scale Tensor ---
        # Scale này dùng để tính khoảng cách thực tế trong Env
        scale_tensor = torch.tensor([[scale]], device=self.device)
        
        # --- Weights ---
        weights = torch.tensor([[0.5, 0.5]], device=self.device)
        
        return (static, dyn_truck, dyn_drone, mask_cust, mask_veh, scale_tensor, weights)

    def __iter__(self):
        # Giả lập hành vi của DataLoader: Yield dữ liệu vô tận (hoặc 1 lần)
        # Ở đây ta yield 1 lần rồi lặp lại để Env gọi reset() được nhiều lần nếu muốn
        while True:
            # Clone tensor để tránh side-effect khi Env modify state in-place
            data_copy = tuple(t.clone() if isinstance(t, torch.Tensor) else t 
                              for t in self.data)
            yield data_copy

def get_file_dataloader(path, device='cpu'):
    """Hàm wrapper để thay thế get_rl_dataloader"""
    return FileDataset(path, device)