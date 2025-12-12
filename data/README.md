# Data Preparation

To predict 2D semantic labels, the model weights of InverseForm is required. Please download the checkpoints from [here](https://github.com/Qualcomm-AI-research/InverseForm/tree/main) and place it in `--model_path /nas/users/hyzhou/model_zoo/hrnet48_OCR_HMS_IF_checkpoint.pth` with paths on your machine in `InverseForm/infer_*.sh` scripts.

Follow the steps below to install `zsh` and `nano` if needed.
``` bash
apt update
apt install zsh
apt install nano -y
```

<summary>Botauto Dataset</summary>

Place `.tfrecord` files in the `raw_data` folder.

Please select the **\$\{segment\}**, replace **\$\{base\_dir\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./waymo/run.sh
```




