import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_interactive_with_baseline(folder_path='results'):
    data_list = []

    # --- 1. Dữ liệu Baseline (Cố định) ---
    baseline_points = [
        (2288.85, 18243.1), (2321.0, 18198.8), (2331.07, 17323.0), (2348.04, 16621.2), 
        (2442.63, 16537.2), (2458.85, 16151.0), (2475.3, 15833.5), (2494.84, 15018.6), 
        (2542.71, 14780.8), (2552.49, 14734.1), (2636.24, 14485.7), (2728.2, 14061.9), 
        (2998.16, 13934.2), (3055.08, 13728.9), (3066.92, 13661.4), (3157.39, 13610.2), 
        (3254.45, 13526.5), (3263.7, 13280.9), (3264.22, 13262.5), (3405.85, 13104.4), 
        (3615.77, 13053.2), (3815.36, 13034.0), (3919.51, 13030.2), (3922.72, 13003.0), 
        (3977.06, 12982.9), (4205.4, 12953.4), (4271.5, 12933.2), (4286.26, 12932.7), 
        (4569.97, 12883.0)
    ]

    # Thêm baseline vào danh sách dữ liệu
    for ms, wt in baseline_points:
        data_list.append({
            "Filename": "N/A",
            "Category": "Baseline", # Đặt tên nhóm riêng là Baseline
            "Makespan": ms,
            "Waiting Time": wt
        })

    # --- 2. Dữ liệu từ thư mục 'results' ---
    if os.path.exists(folder_path):
        print(f"Đang đọc dữ liệu từ '{folder_path}'...")
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    if '-' in filename:
                        category = filename.split('-')[0]
                    else:
                        category = "Other"

                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        metrics = content.get('file_instance', {}).get('metrics', {})
                        ms = metrics.get('makespan')
                        wt = metrics.get('waiting_time')
                        
                        if ms is not None and wt is not None:
                            data_list.append({
                                "Filename": filename,
                                "Category": category,
                                "Makespan": ms,
                                "Waiting Time": wt
                            })
                except Exception as e:
                    print(f"Lỗi file {filename}: {e}")
    else:
        print(f"Cảnh báo: Không tìm thấy thư mục '{folder_path}', chỉ vẽ dữ liệu Baseline.")

    # --- 3. Vẽ biểu đồ ---
    if not data_list:
        print("Không có dữ liệu.")
        return

    df = pd.DataFrame(data_list)
    df = df.sort_values(by="Category")

    # Tạo biểu đồ Scatter
    fig = px.scatter(
        df, 
        x="Makespan", 
        y="Waiting Time", 
        color="Category",
        hover_name="Filename",
        title="Biểu đồ so sánh: Baseline vs Kết quả thực nghiệm",
        labels={
            "Makespan": "Makespan",
            "Waiting Time": "Waiting Time"
        },
        template="plotly_white"
    )

    # Tăng kích thước điểm và thêm viền để dễ nhìn
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # Tùy chọn: Nối các điểm Baseline lại thành đường (line) để dễ phân biệt
    # (Nếu bạn chỉ muốn để dạng điểm chấm (scatter) thì xóa đoạn code dưới đây)
    baseline_df = df[df['Category'] == 'Baseline'].sort_values(by="Makespan")
    fig.add_trace(go.Scatter(
        x=baseline_df['Makespan'], 
        y=baseline_df['Waiting Time'], 
        mode='lines', 
        name='Baseline Line',
        line=dict(color='red', width=2, dash='dash'), # Đường nét đứt màu đỏ
        showlegend=False # Ẩn khỏi legend để không bị trùng lặp
    ))

    fig.show()

if __name__ == "__main__":
    plot_interactive_with_baseline('results')