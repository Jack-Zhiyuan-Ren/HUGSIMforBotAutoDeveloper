#!/bin/bash

cuda=0
export CUDA_VISIBLE_DEVICES=$cuda

## Note: with lastest version of load_custom8 and create_dynamics_mask_custom2
#simpliest version of coordinate transform. For custom files
#new merge_depth_ground_custom and merge_depth_wo_ground_custom with axes make for custom data


# base_dir="/workspace/Jack/HUGSIM/raw_data/split_311238_10parts"
base_dir="/workspace/Jack/HUGSIM/raw_data/split_311238_100parts"
segment="311238_part_0.tfrecord"
seq_name="311238_part_0_100_v12"
#v11 full one with undistorted image and world bounidng box
out=/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/$seq_name
cameras="(1 2 3)"


mkdir -p $out

# load images, camera pose, etc
python waymo/load_custom16.py -b ${base_dir} -c ${cameras} -o ${out} -s ${segment}

# generate semantic mask
cd InverseForm
./infer_waymo.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask_custom6.py --data_path ${out} --data_type waymo
python utils/estimate_depth_custom.py --out ${out}
python utils/merge_depth_wo_ground_custom10.py --out ${out} --total 200000
python utils/merge_depth_ground_custom9.py --out ${out} --total 200000 --datatype waymo