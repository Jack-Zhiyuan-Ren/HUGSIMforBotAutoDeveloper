# Data Preparation

To predict 2D semantic labels, the model weights of InverseForm is required. Please download the checkpoints from [here](https://github.com/Qualcomm-AI-research/InverseForm/tree/main) and place it in `--model_path /workspace/HUGSIMforBotAutoDeveloper/data/samplepoints_in_data/hrnet48_OCR_HMS_IF_checkpoint.pth` with paths on your machine in `InverseForm/infer_*.sh` scripts.

Follow the steps below to install `zsh` and `nano` if needed.
``` bash
apt update
apt install zsh
apt install nano -y
```
Then make the following change.
Go to the repository root directory and change to the directory below.
``` bash
.pixi/envs/default/lib/python3.11/site-packages/unidepth/models/backbones/metadinov2/attention.py
```
Find the line that contains
``` bash
XFORMERS_AVAILABLE = XFORMERS_AVAILABLE and torch.cuda.is_available()
```
Change this to
``` bash
XFORMERS_AVAILABLE =False
```

<summary>Botauto Dataset</summary>

Place `.tfrecord` files in the `raw_data` folder.

Go to `data/waymo/run_custom3`. Replace **\$\{base\_dir\}**,**\$\{segement\}**, **\$\{seq\_name\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./waymo/run_custom3.sh
```




