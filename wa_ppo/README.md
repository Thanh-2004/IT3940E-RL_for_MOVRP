# WA-PPO for Parallel VRPD

Deep RL solution cho bài toán Vehicle Routing Problem with Drones sử dụng PPO.

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Chạy Nhanh

### 1. Training
```bash
python train.py --data 6.5.1.txt --total_timesteps 100000
```

### 2. Evaluation
```bash
python3 evaluate.py --model models/model_w_0.5_0.5.zip --data /Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/200.40.4.txt --visualize
```

### 3. Monitor (Optional)
```bash
tensorboard --logdir ./logs/
```

## Tùy Chỉnh

### Training với weights khác
```bash
# Ưu tiên completion time
python train.py --data 6.5.1.txt --weights 0.8,0.2

# Ưu tiên waiting time
python train.py --data 6.5.1.txt --weights 0.2,0.8
```

### Training nhiều weights (Pareto front)
```bash
python train.py --data 6.5.1.txt --train_multiple
```

### Thay đổi hyperparameters
```bash
python train.py --data 6.5.1.txt \
    --lr 5e-4 \
    --n_steps 2048 \
    --batch_size 128 \
    --total_timesteps 500000
```

## Cấu Trúc File

```
├── config.py              # Quản lý cấu hình
├── data_loader.py         # Load dữ liệu
├── route_calculator.py    # Tính toán routes
├── environment.py         # RL environment
├── wa_ppo.py             # WA-PPO model
├── visualizer.py         # Visualization
├── utils.py              # Utilities
├── train.py              # Script training
├── evaluate.py           # Script evaluation
├── example.py            # Ví dụ sử dụng
└── test_system.py        # System tests
```

## Arguments

### train.py
- `--data`: File dữ liệu (required)
- `--weights`: Weights cho objectives (default: 0.5,0.5)
- `--total_timesteps`: Số timesteps training (default: 500000)
- `--lr`: Learning rate (default: 3e-4)
- `--train_multiple`: Train với nhiều weights

### evaluate.py
- `--model`: Path to model (required)
- `--data`: File dữ liệu (required)
- `--n_episodes`: Số episodes evaluate (default: 10)
- `--visualize`: Tạo visualizations

## Kết Quả

Sau khi chạy, kết quả sẽ được lưu trong:
- `./models/`: Trained models
- `./logs/`: TensorBoard logs
- `./results/`: Visualizations và reports

## Examples

### Example 1: Basic training
```python
python example.py --example 1
```

### Example 2: Multi-weight training
```python
python example.py --example 2
```

### Example 3: Evaluation
```python
python example.py --example 3
```

## Test

```bash
python test_system.py
```

## Format Dữ Liệu

File .txt với format:
```
number_staff 1
number_drone 1
droneLimitationFightTime(s) 3600
Customers 6
Coordinate X    Coordinate Y    Demand    OnlyServicedByStaff    ServiceTimeByTruck(s)    ServiceTimeByDrone(s)
1857.81    -723.55    0.05    1    60    30
...
```

## Requirements

- Python 3.8+
- stable-baselines3
- torch
- numpy
- matplotlib
- seaborn
- tensorboard

## Troubleshooting

**Import errors**: 
```bash
pip install -r requirements.txt
```

**Training chậm**: 
```bash
python train.py --total_timesteps 50000  # Giảm timesteps
```

**Memory issues**: 
```bash
python train.py --batch_size 32  # Giảm batch size
```

## License

MIT