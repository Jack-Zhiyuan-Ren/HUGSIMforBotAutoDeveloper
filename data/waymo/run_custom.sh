#!/bin/bash

cuda=0
export CUDA_VISIBLE_DEVICES=$cuda

# base_dir="/nas/datasets/Waymo_NOTR/static"
# segment="segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord"

base_dir="/workspace/Jack/HUGSIM/raw_data/split_311279_100parts_anchor0"
segment="311279_part_1.tfrecord"

# seg_prefix=$(echo $segment| cut -c 9-15)
# seq_name=${seg_prefix}

# seq_name=$(basename "$segment" .tfrecord)
seq_name="311279_frame_100_200"


out=/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/$seq_name
cameras=(1 2 3)


mkdir -p $out

# load images, camera pose, etc
python waymo/load_custom.py -b ${base_dir} -c ${cameras} -o ${out} -s ${segment}

# generate semantic mask
cd InverseForm
./infer_waymo.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask_custom.py --data_path ${out} --data_type waymo
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype waymo