python utils/train_cur.py \
    --truck_json "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json" \
    --drone_json "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json" \
    --customers_txt "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/50.10.1.txt" \
    --run_dir "./results/training_curriculum_v2.2" \
    --data_dir "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data" \
    --total_iters 3000 \
    --device cpu

python utils/test_final.py \
    --truck_json "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json" \
    --drone_json "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json" \
    --customers_txt "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/20.10.1.txt" \
    --ckpt_path "./results/training_curriculum_v2.2/checkpoints/" \
    --grid_step 0.1 \
    --out_dir "./evaluation_results/test"


python utils/run_env.py \
    --truck_json "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/Truck_config.json" \
    --drone_json "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/drone_linear_config.json" \
    --customers_txt "/Users/nguyentrithanh/Documents/20251/Project3/IT3940E-RL_for_MOVRP/data/random_data/20.10.1.txt" \
    --solutions_dir "/Users/nguyentrithanh/Downloads/WADRL_PVRP-main/wadrl_v3/evaluation_results/20.10.1/"

  
Order: 

50.10.2,
50.10.4,
50.10.3,
50.10.1,
20.20.3,
20.20.2,
20.10.1,
20.10.3,