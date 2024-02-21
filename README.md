# ScaleRAFT
code for ScaleRAFT
## Requirements
The code has been tested with PyTorch 1.11.0 and Cuda 11.3.
```Shell
conda create -n cscv python=3.9
conda activate cscv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Also need to install via pip
```Shell
pip install matplotlib
pip install opencv-python
...
```
## Demo
![image](https://github.com/HanLingsgjk/CSCV/000050.gif?raw=true)
## Dataset Configuration
To evaluate/train CSCV, you will need to download the required datasets. 
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

We recommend manually specifying the path in `dataset_exp_orin.py`, like `def  __init__(self, aug_params=None, split='kitti_test', root='/new_data/datasets/KITTI/training',get_depth=0):` , because the automatic one often makes mistakes

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
python train.py --name raft-cscv --stage kitti --validation kitti --restore_ckpt ../CSCV/checkpotins/raft-kitti_343_6.08.pth --gpus 0 --num_steps 60000 --batch_size 2 --lr 0.000125 --image_size 320 960 --wdecay 0.0001 --gamma=0.85
```

## Test on KITTI
Reproduce the results of Table 1 in the paper
```Shell
python dc_flow_eval.py --model=../CSCV/checkpotins/raft-kitti_343_160_44.75.pth --mixed_precision --start=0
```
