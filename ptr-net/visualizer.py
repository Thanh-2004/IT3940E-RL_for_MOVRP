import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_mopvrp(static_tensor, truck_routes, drone_routes, pretrained, title="MOPVRP Solution"):
    """
    Vẽ biểu đồ lộ trình cho bài toán MOPVRP.
    
    Args:
        static_tensor: Tensor (4, N) chứa [x, y, demand, truck_only]
        truck_routes: List of Lists [[0, 5, 2, 0], [0, 1, 0]] (Lộ trình từng xe tải)
        drone_routes: List of Lists [[0, 8, 0], [0, 4, 0]] (Lộ trình từng drone)
        title: Tiêu đề biểu đồ
    """
    # Chuyển Tensor sang Numpy
    if isinstance(static_tensor, torch.Tensor):
        static = static_tensor.cpu().numpy()
    else:
        static = static_tensor

    coords = static[:2, :] # (2, N)
    truck_only_mask = static[3, :] == 1
    
    plt.figure(figsize=(10, 8))
    
    # 1. Vẽ các Nodes (Khách hàng & Depot)
    # Depot (Node 0)
    plt.scatter(coords[0, 0], coords[1, 0], c='red', marker='s', s=200, label='Depot', zorder=10)
    
    # Khách thường (Flexible)
    flexible_indices = np.where(~truck_only_mask)[0]
    flexible_indices = flexible_indices[flexible_indices != 0] # Bỏ depot
    plt.scatter(coords[0, flexible_indices], coords[1, flexible_indices], 
                c='blue', marker='o', s=50, label='Customer (Flexible)', alpha=0.6)
    
    # Khách Truck-Only
    truck_only_indices = np.where(truck_only_mask)[0]
    plt.scatter(coords[0, truck_only_indices], coords[1, truck_only_indices], 
                c='black', marker='X', s=80, label='Customer (Truck-Only)')

    # Annotate ID cho node
    for i in range(coords.shape[1]):
        plt.text(coords[0, i] + 0.01, coords[1, i] + 0.01, str(i), fontsize=9)

    # 2. Vẽ lộ trình Xe tải (Nét liền)
    colors = plt.cm.get_cmap('tab10', len(truck_routes) + len(drone_routes))
    
    for i, route in enumerate(truck_routes):
        if len(route) < 2: continue # Xe không đi đâu
        
        route_coords = coords[:, route]
        color = colors(i)
        
        plt.plot(route_coords[0], route_coords[1], c=color, linewidth=2, 
                 label=f'Truck {i}', linestyle='-', marker='.')
        
        # Vẽ mũi tên hướng đi
        mid_idx = len(route) // 2
        if mid_idx < len(route) - 1:
            p1 = coords[:, route[mid_idx]]
            p2 = coords[:, route[mid_idx+1]]
            plt.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.5, (p2[1]-p1[1])*0.5, 
                      head_width=0.015, color=color)

    # 3. Vẽ lộ trình Drone (Nét đứt)
    for i, route in enumerate(drone_routes):
        if len(route) < 2: continue
        
        # Drone trips thường là star-shaped: 0 -> Node -> 0
        # Vẽ từng đoạn nhỏ để không bị rối
        color = colors(len(truck_routes) + i)
        
        # Vẽ các chuyến đi (Trips)
        # Giả sử route là [0, 5, 0, 8, 0] -> Vẽ từng cặp (0,5), (5,0), ...
        route_coords = coords[:, route]
        plt.plot(route_coords[0], route_coords[1], c=color, linewidth=1.5, 
                 label=f'Drone {i}', linestyle='--')

    suffix = " (Pretrained Model)" if pretrained else " (Random Model)"
    title = title + suffix

    plt.title(title)
    plt.xlabel("X Coordinate (Normalized)")
    plt.ylabel("Y Coordinate (Normalized)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{suffix}.png", dpi=300)
    plt.show()