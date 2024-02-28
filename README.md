# ScaleRAFT
code for ScaleRAFT: Cross-Scale Recurrent All-Pairs Field Transforms for 3D Motion Estimation.

(A Robust Method for Extracting 3D Motion from Videos)


## Requirements
The code has been tested with PyTorch 1.11.0 and Cuda 11.3.
```Shell
conda create -n cscv python=3.9
conda activate cscv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Also need to install via pip
```Shell
pip install matplotlib==3.5
pip install opencv-python
pip install tqdm
pip install pypng
pip install scipy
pip install einops
pip install tensorboard
```
## Demo


soapbox:

https://github.com/HanLingsgjk/CSCV/assets/102562963/14c603dd-6690-41e5-9259-025240395da8

parkour:

https://github.com/HanLingsgjk/CSCV/assets/102562963/b722d727-d185-49e3-92bf-c27b02182ff5

motorbike:

https://github.com/HanLingsgjk/CSCV/assets/102562963/ea7be7b6-963d-4f4f-923b-7b1c56a0b0cb


motocross-jump:

https://github.com/HanLingsgjk/CSCV/assets/102562963/6d2f4db7-e6a6-4528-8133-2b4f64420e8e


lady-running:

https://github.com/HanLingsgjk/CSCV/assets/102562963/276dbfe7-1ad7-400c-9507-80da94130689

car-shadow:

https://github.com/HanLingsgjk/CSCV/assets/102562963/c9ab1122-1d0b-4cd8-80bf-4ef9a354ab35

breakdance-flare:


https://github.com/HanLingsgjk/CSCV/assets/102562963/44f7bd1f-6286-4f2b-b117-21272fbad255


Dog:

https://github.com/HanLingsgjk/CSCV/assets/102562963/5f206891-1436-43e5-b5e1-754604eeee70





## Dataset Configuration
To evaluate/train CSCV, you will need to download the required datasets. 
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

We recommend manually specifying the path in `dataset_exp_orin.py` , like in line 477 `def  __init__(self, aug_params=None, split='kitti_test', root='/new_data/datasets/KITTI/',get_depth=0):` , '/new_data/datasets/KITTI/' is where you put the KITTI dataset.

You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder
```Shell
├── datasets
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```
## Pretrained weights for KITTI
Download and place in the root directory
* https://drive.google.com/drive/folders/129lbJWkcMwxispcRVXOvUGF12GuHbhX3?usp=drive_link

## Train on KITTI
```Shell
python train.py --name raft-cscv --stage kitti --validation kitti --restore_ckpt ../CSCV/checkpotins/cscv_kittitest_6.12.pth --gpus 0 --num_steps 60000 --batch_size 2 --lr 0.000125 --image_size 320 960 --wdecay 0.0001 --gamma=0.85
```

## Test on KITTI
Reproduce the results of Table 1 in the paper
```Shell
python dc_flow_eval.py --model=../CSCV/checkpotins/cscv_kitti_42.08.pth --mixed_precision --start=0
```
