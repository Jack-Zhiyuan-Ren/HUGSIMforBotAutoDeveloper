# Data Preparation

To predict 2D semantic labels, the model weights of InverseForm is required. Please download the checkpoints from [here](https://github.com/Qualcomm-AI-research/InverseForm/tree/main) and place it in `--model_path /nas/users/hyzhou/model_zoo/hrnet48_OCR_HMS_IF_checkpoint.pth` with paths on your machine in `InverseForm/infer_*.sh` scripts.



<summary>Waymo Open Dataset</summary>

Download Waymo NOTR dataset following [the EmerNeRF doc](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md).

Please select the **\$\{segment\}**, replace **\$\{base\_dir\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./waymo/run.sh
```




