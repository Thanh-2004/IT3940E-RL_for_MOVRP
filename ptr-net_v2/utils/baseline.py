import pandas as pd
import re
import matplotlib.pyplot as plt

def parse_pareto(pareto_str):
    """Phân tách chuỗi Pareto thành danh sách các tọa độ (f1, f2)."""
    if pd.isna(pareto_str):
        return []
    # Tìm các mẫu số (x, y) trong chuỗi
    matches = re.findall(r"\(([\d\.]+),\s*([\d\.]+)\)", str(pareto_str))
    return [(float(m[0]), float(m[1])) for m in matches]

# 1. Tải dữ liệu
baseline_df = pd.read_csv('BaseLine.csv')
# MOVRP_Final_Pareto.csv có dòng tiêu đề thực sự ở dòng thứ 2
movrp_df = pd.read_csv('MOVRP_Final_Pareto.csv', skiprows=1)

# Làm sạch dữ liệu MOVRP (loại bỏ dòng không có tên Instance)
movrp_df = movrp_df.dropna(subset=['Instance'])

# 2. Lấy danh sách các Instance chung giữa hai tệp
instances = sorted(list(set(baseline_df['Instance']).intersection(set(movrp_df['Instance']))))

# 3. Thiết lập biểu đồ
n_inst = len(instances)
cols = 2
rows = (n_inst + 1) // cols
plt.figure(figsize=(14, 6 * rows))

for i, inst in enumerate(instances):
    plt.subplot(rows, cols, i + 1)
    
    # Lấy điểm từ Baseline
    b_str = baseline_df[baseline_df['Instance'] == inst]['Pareto'].values[0]
    b_points = parse_pareto(b_str)
    b_points.sort() # Sắp xếp theo F1 để dễ quan sát
    
    # Lấy điểm từ MOVRP Final
    m_str = movrp_df[movrp_df['Instance'] == inst]['Pareto'].values[0]
    m_points = parse_pareto(m_str)
    m_points.sort()

    # Vẽ điểm Baseline
    if b_points:
        bx, by = zip(*b_points)
        plt.scatter(bx, by, color='blue', label='Baseline', marker='o', s=40, alpha=0.6)
    
    # Vẽ điểm MOVRP
    if m_points:
        mx, my = zip(*m_points)
        plt.scatter(mx, my, color='red', label='MOVRP Final', marker='x', s=60, alpha=0.9)

    plt.title(f"Instance: {inst}")
    plt.xlabel("F1 (Makespan)")
    plt.ylabel("F2 (Waiting Time)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

plt.suptitle("So sánh tập Pareto: Baseline vs MOVRP Final", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('Pareto_Comparison.png', bbox_inches='tight')
plt.show()