import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Tải dữ liệu từ file CSV
df = pd.read_csv('/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/ptr-net_v2/results/training_curriculum_v2.1/train_stats.csv')

# 2. Xác định các điểm chuyển đổi Curriculum (nơi instance thay đổi)
transitions = df[df['instance'] != df['instance'].shift()].iloc[1:]

# 3. Cấu hình các thông số cần vẽ
metrics = [
    ('loss', 'Total Loss', 'blue'),
    ('pg', 'Policy Gradient Loss (Actor)', 'green'),
    ('v', 'Value Loss (Critic)', 'red'),
    ('ent', 'Entropy (Exploration)', 'purple'),
    ('kl', 'Approx KL Divergence', 'orange'),
    ('ex_var', 'Explained Variance', 'brown')
]

# 4. Tạo khung biểu đồ (3 hàng x 2 cột)
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axes = axes.flatten()

for i, (col, title, color) in enumerate(metrics):
    ax = axes[i]
    # Vẽ đường thông số chính
    ax.plot(df['iter'], df[col], label=col, color=color, linewidth=1.5)
    
    # Vẽ các đường thẳng đứng đánh dấu mốc Curriculum
    for _, row in transitions.iterrows():
        ax.axvline(x=row['iter'], color='black', linestyle='--', alpha=0.3)
        # Ghi tên instance lên biểu đồ đầu tiên để tránh rối mắt
        if i == 0:
            ax.text(row['iter'], ax.get_ylim()[1], f" {row['instance']}", 
                    rotation=90, verticalalignment='top', fontsize=9, fontweight='bold')
            
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    # Đặt nhãn trục X cho các biểu đồ hàng cuối
    if i >= 4:
        ax.set_xlabel('Iteration', fontsize=12)

# 5. Tinh chỉnh và hiển thị
plt.suptitle(f"WADRL Training Process Visualization - {df['instance'].iloc[-1]}", fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('training_report.png', dpi=300)
plt.show()

# 6. In bảng tổng hợp theo từng Instance ra console
summary = df.groupby('instance').agg({
    'loss': 'mean',
    'ent': 'mean',
    'ex_var': 'mean'
}).reset_index()
print("\n--- Summary by Instance ---")
print(summary)