#!/usr/bin/env zsh
# ================================
# Run HugSim reconstruction pipeline (ZSH version)
# ================================

# ---- Configuration ----
# seq="311279_frame_100_200"
# seq="311238_part_0_200"
# seq="1002394"
# seq="1006130"
seq="311238_part_0_100_v11"
# seq="1083056"

input_path="/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/${seq}"
output_path="/workspace/Jack/HUGSIM/data/samplepoints_in_data/Reconstruction/311238_part_0_100_v16"
# v11 ground_points3d_v14.ply points_3d_v31.ply meta_data_v31.json ground_params_v12.pkl
# v12  ground_points3d_v3.ply points_3d_v7.ply meta_data_v2.json ground_params_v2.pkl
# v13 ground_points3d_v4.ply points_3d_v8.ply meta_data_v2.json ground_params_v3.pkl
# v14 same as 13 but 60k
# v15 30k grounf_points3d_v6.ply points_3d_v9.ply meta_data_v3.json gorund_params_v5.pkl
# v16  from v11
data_cfg="./configs/waymo.yaml"

# ---- Setup ----
mkdir -p "${output_path}"

echo "=============================================="
echo "Running reconstruction for sequence: ${seq}"
echo "Input path:  ${input_path}"
echo "Output path: ${output_path}"
echo "Config file: ${data_cfg}"
echo "=============================================="

# ---- Stage 1: Ground model training ----
# CUDA_VISIBLE_DEVICES=4 \
python -u train_ground.py \
    --data_cfg "${data_cfg}" \
    --source_path "${input_path}" \
    --model_path "${output_path}"

# ---- Stage 2: Full model training ----
# [[CUDA_VISIBLE_DEVICES=4 \]]
python -u train_custom.py \
    --data_cfg "${data_cfg}" \
    --source_path "${input_path}" \
    --model_path "${output_path}"

# CUDA_VISIBLE_DEVICES=4 \
# python -u train.py \
#     --data_cfg "${data_cfg}" \
#     --source_path "${input_path}" \
#     --model_path "${output_path}"

echo "=============================================="
echo "Reconstruction complete for ${seq}"
echo "Output saved to ${output_path}"
echo "=============================================="
