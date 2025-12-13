<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <img src="assets/hugsim.png" alt="Logo" width="300">
  
  <p>
    <a href="https://xdimlab.github.io/HUGSIM/">
      <img src="https://img.shields.io/badge/Project-Page-green?style=for-the-badge" alt="Project Page" height="20">
    </a>
    <a href="https://arxiv.org/abs/2412.01718">
      <img src="https://img.shields.io/badge/arXiv-Paper-red?style=for-the-badge" alt="arXiv Paper" height="20">
    </a>
  </p>
  
  > Hongyu Zhou<sup>1</sup>, Longzhong Lin<sup>1</sup>, Jiabao Wang<sup>1</sup>, Yichong Lu<sup>1</sup>, Dongfeng Bai<sup>2</sup>, Bingbing Liu<sup>2</sup>, Yue Wang<sup>1</sup>, Andreas Geiger<sup>3,4</sup>, Yiyi Liao<sup>1,†</sup> <br>
  > <sup>1</sup> Zhejiang University <sup>2</sup> Huawei <sup>3</sup> University of Tübingen <sup>4</sup> Tübingen AI Center <br>
  > <sup>†</sup> Corresponding Authors

  <img src="assets/teaser.jpg" width="800" style="display: block; margin: 0 auto;">

  <br>

  <p align="left">
    This is the official project repository of the HUGSIM for BotAuto Data.
  </p>
  
</div>

---

# Installation

First, install [pixi](https://pixi.sh/latest/):

``` bash
curl -fsSL https://pixi.sh/install.sh | sh
```

As the repository depends on some packages that can only be installed from source code, and rely on pytorch and cuda to compile, the installation of pixi environment is seperated as **two steps**:

1. Comment the packages below `# install from source code` in `pixi.toml`, then run `pixi install` to install the packages from pypi.
2. Uncomment the packages in the previous step, then run `pixi install` to install these packages from source code.
3. Install apex (required by InverseForm) by running `pixi run install-apex`

Change into the **pixi environment** by using the command `pixi shell`.

Or you can use `pixi run <command>` to run a command in the **pixi environment**.


# Data Preparation

Please refer to [Data Preparation Document](data/README.md)

You can download sample data from [here](https://drive.google.com/file/d/1GXdWYJKJMT-MNb7g8mDm9MHCURBmgkg7/view?usp=drive_link).
The Sample data comes from Scene311238.

# Reconstruction

``` bash
seq=${seq_name}
input_path=${datadir}/${seq}
output_path=${modeldir}/${seq}
mkdir -p ${output_path}
CUDA_VISIBLE_DEVICES=4 \
python -u train_ground.py --data_cfg ./configs/${dataset_name: [kitti360, waymo, nusc, pandaset]}.yaml \
        --source_path ${input_path} --model_path ${output_path}
CUDA_VISIBLE_DEVICES=4 \
python -u train.py --data_cfg ./configs/${dataset_name}.yaml \
        --source_path ${input_path} --model_path ${output_path}
```

# Scene Export

The reconstructed scene folders contain some information that won't be utilized during the simulation. The scenes are expected to be exported as a minimized format to facilitate easier sharing and simulation.
```bash
 python eval_render/export_scene.py --model_path ${recon_scene_path} --output_path ${export_path} --iteration 30000
``` 
We've made some changes in the capturing and reloading code. If you would like to convert scenes from previous version (before commit 1ca821a8) of our code, add `--ver0` in the above command. 

# Vehicles, Scenes and Scenarios

We have released all the 3DRealCar files, 49 scenes and corresponding 345 scenarios available at the [release link](https://huggingface.co/datasets/XDimLab/HUGSIM). 
We are also holding a competition at [RealADSim @ ICCV 2025](https://huggingface.co/spaces/XDimLab/ICCV2025-RealADSim-ClosedLoop), so some scenarios and scenarios are hosted privately. We welcome participants to join!

# Scenarios configuration with GUI

**Note that this GUI is only used for configuration scenarios, rather than simulation. The rendering quality in GUI is not the results during simulation**

First convert the vehicles and scenes to splat and semantic format.

``` bash
python eval_render/convert_vehicles.py --vehicle_path ${PATH_3DRealCar}
python eval_render/convert_scene.py --model_path ${PATH_Scene}
```

Then, you can run the GUI to configure the scenario. 
**nuscenes_camera.yaml** in gui/static/data provides a camera configuration template, you can modify it to fit your needs.

``` bash
cd gui
python app.py --scene ${PATH_Scene} --car_folder ${PATH_3DRealCar/converted}
```

You can configure the scenario with the GUI, and download the yaml file to use in simulation.

Here is a video for the GUI usage demonstration: [GUI Video](https://github.com/hyzhou404/HUGSIM/blob/main/assets/hugsim_gui.mp4)





# Citation


```bibtex
@article{zhou2024hugsim,
  title={HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving},
  author={Zhou, Hongyu and Lin, Longzhong and Wang, Jiabao and Lu, Yichong and Bai, Dongfeng and Liu, Bingbing and Wang, Yue and Geiger, Andreas and Liao, Yiyi},
  journal={arXiv preprint arXiv:2412.01718},
  year={2024}
}
```
