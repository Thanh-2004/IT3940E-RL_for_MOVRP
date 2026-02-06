import pandas as pd
import csv
import re

def parse_tuple(s):
    # Trích xuất các số từ chuỗi định dạng "(F1, F2)"
    vals = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return [float(v) for v in vals]

file_path = 'solution/raw.csv'
with open(file_path, 'r', encoding='utf-8') as f:
    reader = list(csv.reader(f))

# Lấy bộ trọng số từ hàng 2
weights = {}
for i in range(1, 12):
    weights[i] = parse_tuple(reader[2][i])

# Duyệt qua từng hàng dữ liệu (từ hàng index 3)
for r_idx in range(3, len(reader)):
    row = reader[r_idx]
    if not row or len(row) < 12 or not row[0]: continue
    
    scalar_vals_subset = [] # Dành cho Min, Max, Avg (0.1 - 0.9)
    unique_tuples_all = []  # Dành cho Pareto (0.0 - 1.0)
    seen = set()
    
    # Duyệt qua tất cả các cột chứa cặp (F1, F2)
    for i in range(1, 12):
        t_str = row[i].strip()
        if not t_str: continue
        
        # 1. Pareto: Lấy từ tất cả các cột (Trọng số 0.0 đến 1.0)
        if t_str not in seen:
            unique_tuples_all.append(t_str)
            seen.add(t_str)
            
        # 2. Min/Max/Avg: Chỉ lấy từ dải trọng số 0.1 đến 0.9 (Cột index 2 đến 10)
        if 2 <= i <= 10:
            f1, f2 = parse_tuple(t_str)
            w1, w2 = weights[i]
            scalar_vals_subset.append(w1 * f1 + w2 * f2)
            
    if scalar_vals_subset:
        # Cập nhật các giá trị thống kê chỉ dựa trên dải 0.1 - 0.9
        reader[r_idx][12] = f"{min(scalar_vals_subset):.2f}"
        reader[r_idx][13] = f"{max(scalar_vals_subset):.2f}"
        reader[r_idx][14] = f"{sum(scalar_vals_subset)/len(scalar_vals_subset):.2f}"
        
    # Cập nhật cột Pareto lấy từ toàn bộ các cột
    reader[r_idx][15] = " | ".join(unique_tuples_all)

# Lưu file kết quả cuối cùng
output_file = 'MOVRP_Final_Pareto.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(reader)