#!/bin/bash

# --- Cấu hình các đường dẫn cơ bản ---
TRUCK_JSON="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json"
DRONE_JSON="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json"
DATA_DIR="/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data"
CKPT_PATH="./results/training_curriculum_v2.1/checkpoints"
BASE_OUT_DIR="./evaluation_results"

# --- Danh sách 8 instance ---
INSTANCES=("20.10.1" "20.10.3" "20.20.2" "20.20.3" "50.10.1" "50.10.2" "50.10.3" "50.10.4")

# --- Lựa chọn chế độ chạy ---
echo "Select execution mode:"
echo "1) Run BOTH (test_final.py & run_env.py)"
echo "2) Run ONLY Simulator (run_env.py)"
read -p "Enter choice [1 or 2]: " MODE

for INST in "${INSTANCES[@]}"
do
    echo "------------------------------------------------------------"
    echo "PROCESSING: $INST"
    CUSTOMER_FILE="${DATA_DIR}/${INST}.txt"
    OUT_DIR="${BASE_OUT_DIR}/${INST}"

    # --- BƯỚC 1: test_final.py (Chỉ chạy nếu chọn mode 1) ---
    if [ "$MODE" == "1" ]; then
        echo "[Step 1] Running test_final.py..."
        python utils/test_final.py \
            --truck_json "$TRUCK_JSON" \
            --drone_json "$DRONE_JSON" \
            --customers_txt "$CUSTOMER_FILE" \
            --ckpt_path "$CKPT_PATH" \
            --grid_step 0.1 \
            --out_dir "$OUT_DIR"
        
        if [ $? -ne 0 ]; then
            echo "Error in test_final.py for $INST. Skipping..."
            continue
        fi
    else
        echo "[Skip] Skipping test_final.py as requested."
    fi

    # --- BƯỚC 2: run_env.py (Luôn chạy) ---
    if [ -d "$OUT_DIR" ]; then
        echo "[Step 2] Running run_env.py (Simulator)..."
        python utils/run_env.py \
            --truck_json "$TRUCK_JSON" \
            --drone_json "$DRONE_JSON" \
            --customers_txt "$CUSTOMER_FILE" \
            --solutions_dir "$OUT_DIR/"
    else
        echo "[Error] Directory $OUT_DIR does not exist. Cannot run Simulator."
    fi

    echo -e "Finished instance $INST\n"
done

echo "Done."