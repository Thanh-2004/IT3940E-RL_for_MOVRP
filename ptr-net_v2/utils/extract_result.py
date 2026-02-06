import re
import pandas as pd
from pathlib import Path

def parse_summary(file_path, source_label):
    """Trích xuất dữ liệu và gán nhãn nguồn rút gọn (Source 1, 2...)"""
    data = []
    # Regex để bắt các dòng dữ liệu trong bảng
    pattern = re.compile(r'^\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*([\w\.]+)')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                data.append({
                    'w1': float(match.group(1)),
                    'F1': float(match.group(2)),
                    'F2': float(match.group(3)),
                    'F1+F2': float(match.group(4)),
                    'Best_CKPT': match.group(5).strip(),
                    'Source_File': source_label
                })
    return pd.DataFrame(data)
# Chỉ định instance cần so sánh
instance = "50.10.4"

# 1. Liệt kê danh sách các file summary bạn muốn so sánh
summary_files = ['/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net_v2/evaluation_results1/{}/evaluation_summary.txt'.format(instance), 
                 '/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net_v2/evaluation_results2/{}/evaluation_summary.txt'.format(instance)] 

all_dfs = []
for i, f_path in enumerate(summary_files):
    if Path(f_path).exists():
        # Tạo nhãn rút gọn: Source 1, Source 2...
        label = f"Source {i+1}"
        all_dfs.append(parse_summary(f_path, label))

if all_dfs:
    df_combined = pd.concat(all_dfs)
    # Lấy kết quả tốt nhất theo tổng F1+F2 cho mỗi mức w1
    df_best = df_combined.sort_values('F1+F2').groupby('w1').first().reset_index()
    df_best['w2'] = (1.0 - df_best['w1']).round(2)
    
    # --- BẢN 1: ĐƠN VỊ GIỜ (HOURS) ---
    df_hours = df_best[['w1', 'w2', 'F1', 'F2', 'F1+F2', 'Best_CKPT', 'Source_File']]
    df_hours.to_csv(f'solution/best_weights_hours_{instance}.csv', index=False)
    
    # --- BẢN 2: ĐƠN VỊ GIÂY (SECONDS) ---
    df_seconds = df_best.copy()
    df_seconds['F1'] = (df_seconds['F1'] * 3600).round(2)
    df_seconds['F2'] = (df_seconds['F2'] * 3600).round(2)
    df_seconds['F1+F2'] = (df_seconds['F1+F2'] * 3600).round(2)
    
    df_seconds = df_seconds[['w1', 'w2', 'F1', 'F2', 'F1+F2', 'Best_CKPT', 'Source_File']]
    df_seconds.to_csv(f'solution/best_weights_seconds_{instance}.csv', index=False)
    
    print("--- Thống kê kết quả tốt nhất (Đơn vị: Giây) ---")
    print(df_seconds)
    print(f"\nĐã lưu 2 bản: 'best_weights_hours_{instance}.csv' và 'best_weights_seconds_{instance}.csv'")
else:
    print("Không tìm thấy dữ liệu để xử lý.")