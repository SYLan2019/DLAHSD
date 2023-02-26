# Prerequisites
DLAHSD: Dynamic label adopted in auxiliary head for SAR detection. An efficient SAR-ship detector based on center point.
Our work improvements and ablation experimental results all conducted on the MMDetection toolkit.
MMDetection works on Linux, Windows. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.5+.

# Installation
**Step 0.** Install VS2019 for C++ compiler.

**Step 1.** Conda environment installmen and activation.

```shell
conda create -n new_env python=3.8
conda activate new_env
```

**Step 2.** install Pytorch and other useful python pakage

```shell
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
-i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

pip install cython matplotlib opencv-python timm -i [http://mirrors.aliyun.com/pypi/simple/](http://mirrors.aliyun.com/pypi/simple/) 
--trusted-host mirrors.aliyun.com
```

**Step 3.** Install MMCV.

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

**Step 4.** Install MMDetection.
```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
# Dataset

SAR Ship Detection Dataset (SSDD): Official Release and Comprehensive Data Analysis
https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view?usp=sharing
https://pan.baidu.com/s/1Lpg28ZvMSgNXq00abHMZ5Q password: 2021

SSDD is organized in coco form and stored in DLAHSD\Official-SSDD-OPEN\BBox_SSDD\coco_style


# Train
DLAHSD training shell
```shell
cd ../DLAHSD
python tools/train.py work_dirs/DLAHSD/DHALSD.py
```
# Test
```shell
python tools/test.py work_dirs/DLAHSD/DHALSD.py work_dirs/DLAHSD/____.pth  --out='   '
```
# Inference
```shell
python tools/inference.py 
```
