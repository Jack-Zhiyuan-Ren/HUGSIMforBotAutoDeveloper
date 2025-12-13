#!/usr/bin/env zsh
# ================================
# Run HugSim reconstruction pipeline (ZSH version)
# ================================

# ---- Configuration ----
seq="311238_part_0_100"
input_path="/workspace/HUGSIMforBotAutoDeveloper/data/samplepoints_in_data/waymo_data/${seq}"
output_path="/workspace/HUGSIMforBotAutoDeveloper/data/samplepoints_in_data/Reconstruction/311238_part_0_100"
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


echo "=============================================="
echo "Reconstruction complete for ${seq}"
echo "Output saved to ${output_path}"
echo "=============================================="
