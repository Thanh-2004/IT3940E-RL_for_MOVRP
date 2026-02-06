import pandas as pd
import csv
import re

def parse_tuple(s):
    # Trích xuất số từ chuỗi định dạng "(F1, F2)"
    vals = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return [float(v) for v in vals]

def calculate_hv_2d(points, ref_point):
    """Tính Hypervolume 2D cho tập điểm (bài toán cực tiểu hóa)"""
    # Lọc điểm nằm trong vùng tham chiếu và sắp xếp theo F1
    valid_points = sorted([p for p in points if p[0] <= ref_point[0] and p[1] <= ref_point[1]], key=lambda x: x[0])
    if not valid_points: return 0.0
    
    # Lọc bỏ các điểm bị thống trị (non-dominated filtering)
    non_dominated = []
    for p in valid_points:
        if not any((other[0] <= p[0] and other[1] <= p[1]) and (other[0] < p[0] or other[1] < p[1]) for other in valid_points):
            non_dominated.append(p)
    
    # Tính diện tích bằng phương pháp quét (sweeping)
    hv = 0.0
    current_f2_limit = ref_point[1]
    for p in non_dominated:
        width = ref_point[0] - p[0]
        height = current_f2_limit - p[1]
        if height > 0:
            hv += width * height
            current_f2_limit = p[1]
    return hv

# Cấu hình đường dẫn và điểm tham chiếu
file_path = '/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net_v2/MOVRP_Final_Pareto.csv'
ref_point = [36000, 360000] # (10*3600, 100*3600)

with open(file_path, 'r', encoding='utf-8') as f:
    reader = list(csv.reader(f))

# Xử lý tiêu đề cột
reader[1].append('Hypervolume')

# Duyệt và xử lý dữ liệu
for r_idx in range(3, len(reader)):
    row = reader[r_idx]
    if not row or not row[0]: continue
    
    unique_points = []
    seen = set()
    
    for i in range(1, 12): # Quét toàn bộ weight từ 0.0 đến 1.0 cho Pareto/HV
        t_str = row[i].strip()
        if t_str and t_str not in seen:
            unique_points.append(parse_tuple(t_str))
            seen.add(t_str)
    
    # Tính Hypervolume
    hv_val = calculate_hv_2d(unique_points, ref_point) / (3600 * 3600)  # Chuyển đổi sang giờ bình thường hóa
    row.append(f"{hv_val:.2f}")

# Lưu kết quả
output_file = 'MOVRP_Final_with_HV.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(reader)