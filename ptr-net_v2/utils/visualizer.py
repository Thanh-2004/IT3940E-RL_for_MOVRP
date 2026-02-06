#!/usr/bin/env python3
import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path

def load_customers_simple(path):
    """Load tọa độ và thông tin Only Truck từ file customers.txt"""
    coords = []
    only_truck_nodes = []
    header_re = re.compile(r"^\s*Coordinate\s+X", flags=re.IGNORECASE)
    header_seen = False
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if not header_seen:
                if header_re.search(line): header_seen = True
                continue
            parts = line.split()
            x, y = float(parts[0]), float(parts[1])
            is_only_truck = int(parts[3])
            coords.append((x, y))
            if is_only_truck:
                only_truck_nodes.append(len(coords)) # Index 1-based
    return [(0.0, 0.0)] + coords, only_truck_nodes

def visualize_solution(full_coords, only_truck_nodes, sol_file, output_dir=None):
    """Hàm vẽ đường đi cho một file lời giải cụ thể"""
    truck_routes = {}
    drone_routes = {}
    
    with open(sol_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            nums = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(nums) != 3: continue
            v_type, v_idx, dest = nums
            routes = truck_routes if v_type == 0 else drone_routes
            if v_idx not in routes: routes[v_idx] = [0]
            routes[v_idx].append(dest)

    plt.figure(figsize=(10, 8))
    # Vẽ Nodes
    all_x = [c[0] for c in full_coords]
    all_y = [c[1] for c in full_coords]
    plt.scatter(all_x[1:], all_y[1:], c='lightgrey', s=80, label='Khách hàng', zorder=2)
    
    # Highlight Only Truck Nodes
    ot_x = [full_coords[i][0] for i in only_truck_nodes]
    ot_y = [full_coords[i][1] for i in only_truck_nodes]
    plt.scatter(ot_x, ot_y, c='red', marker='s', s=100, label='Nút chỉ Truck', zorder=3)
    
    # Depot
    plt.scatter([0], [0], c='gold', marker='*', s=250, label='Depot', edgecolors='black', zorder=5)

    # Vẽ Truck Routes (Nét liền)
    for t_id, path in truck_routes.items():
        px = [full_coords[n][0] for n in path]
        py = [full_coords[n][1] for n in path]
        plt.plot(px, py, '-', linewidth=2, alpha=0.8, label=f'Truck {t_id}')

    # Vẽ Drone Routes (Nét đứt)
    for d_id, path in drone_routes.items():
        px = [full_coords[n][0] for n in path]
        py = [full_coords[n][1] for n in path]
        plt.plot(px, py, '--', linewidth=1.2, alpha=0.6, label=f'Drone {d_id}')

    plt.title(f"Visualization: {str(sol_file.name)[:5]}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f"{str(sol_file.stem)[:5]}.png")
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize MOVRP solutions from a directory.")
    parser.add_argument("--customers_txt", required=True, help="Path to customers.txt")
    parser.add_argument("--solutions_dir", required=True, help="Folder containing .txt solution files")
    parser.add_argument("--save_png", action="store_true", help="Save as PNG instead of showing popup")
    
    args = parser.parse_args()

    # Load dữ liệu môi trường (toạ độ)
    full_coords, only_truck_nodes = load_customers_simple(args.customers_txt)
    
    solutions_path = Path(args.solutions_dir)
    output_path = Path("visualizations") if args.save_png else None
    if output_path: output_path.mkdir(exist_ok=True)

    # Quét qua folder như logic file cũ
    solution_files = sorted([f for f in solutions_path.glob("*.txt") if f.name != "evaluation_summary.txt"])

    print(f"Found {len(solution_files)} files. Visualizing...")
    for sol_file in solution_files:
        print(f"Processing: {sol_file.name}")
        visualize_solution(full_coords, only_truck_nodes, sol_file, output_path)

if __name__ == "__main__":
    main()