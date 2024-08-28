# RetFormer: Embracing Point Cloud Transformer with Retentive Network

This is the official PyTorch implementation of the paper **RetFormer: Embracing Point Cloud Transformer with Retentive Network**, by Gopi Krishna Erabati and Helder Araujo.

G. K. Erabati and H. Araujo, "RetFormer: Embracing Point Cloud Transformer With Retentive Network," in _IEEE Transactions on Intelligent Vehicles_, 2024. (IEEE T-IV, IF=14 (2023)) doi: [10.1109/TIV.2024.3417260](https://doi.org/10.1109/TIV.2024.3417260)

**Contents**
1. [Overview](https://github.com/gopi-erabati/RetFormer#overview)
2. [Results](https://github.com/gopi-erabati/RetFormer#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/RetFormer#requirements-installation-and-usage)
    1. [Prerequistes](https://github.com/gopi-erabati/RetFormer#prerequisites)
    2. [Installation](https://github.com/gopi-erabati/RetFormer#installation)
    3. [Training](https://github.com/gopi-erabati/RetFormer#training)
    4. [Testing](https://github.com/gopi-erabati/RetFormer#testing)
4. [Acknowlegements](https://github.com/gopi-erabati/RetFormer#acknowlegements)
5. [Reference](https://github.com/gopi-erabati/RetFormer#reference)

## Overview
Point Cloud Transformers (PCTs) have gained lot of attention not only on the indoor data but also on the large-scale outdoor 3D point clouds, such as in autonomous driving. However, the vanilla self-attention mechanism in PCTs does not include any explicit prior spatial information about the quantized voxels (or pillars). Recently, Retentive Network has gained attention in the natural language processing (NLP) domain due to its efficient modelling capability and remarkable performance, leveraged by the introduction of explicit decay mechanism which incorporates the distance related spatial prior knowledge into the model. As the NLP tasks are causal and one-dimensional in nature, the explicit decay is designed to be unidirectional and one-dimensional. However, the pillars in the Bird's Eye View (BEV) space are two-dimensional without causal properties. In this work, we propose **RetFormer** model by introducing bidirectional and two-dimensional decay mechanism for pillars in PCT and design the novel Multi-Scale Retentive Self-Attention (MSReSA) module. The introduction of explicit bidirectional and two-dimensional decay incorporates the 2D spatial distance related prior information of pillars into the PCT which significantly improves the modelling capacity of RetFormer. We evaluate our method on large-scale Waymo and KITTI datasets. RetFormer not only achieves significant performance gain over of 2.4 mAP and 0.9 mAP over PCT-based SST and FlatFormer respectively, and 2.7 mAP over sparse convolutional-based CenterPoint on Waymo Open Dataset, but also is efficient with **3.2x** speedup over SST and runs in real-time at ~69 FPS on a RTX 4090 GPU.

## Results

### Predictions on Waymo dataset
![1719764342896-ezgif com-optimize](https://github.com/user-attachments/assets/068992c2-844f-49a4-8d80-1dc00ca6b265)

| Config | Veh. L1 AP/APH | Veh. L2 AP/APH | Ped. L1 AP/APH | Ped. L2 AP/APH | Cyc. L1 AP/APH | Cyc. L2 AP/APH | Latency (ms) |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| [RetFormer-waymo-1f](configs/retformer_waymo_D1_2x_3class.py) | 76.3/75.8 | 68.0/67.6 | 81.5/72.9 | 74.7/66.6 | 71.8/70.4 | 69.1/67.8 | 14.5 |
| [RetFormer-waymo-2f](configs/retformer_waymo_D1_2x_3class_2f.py) | 77.8/77.3 | 69.7/69.3 | 82.5/77.6 | 76.1/71.5 | 74.6/73.6 | 72.3/71.3 | 15.9 | 

We can not distribute the model weights on Waymo dataset due to the [Waymo license terms](https://waymo.com/open/terms).

| Config | Ped. easy | Ped. mod. | Ped. hard | Cyc. easy | Cyc. mod. | Cyc. hard | |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| [RetFormer-kitti](configs/retformer_kitti_3class.py) | 74.0 | 70.3 | 66.0 | 84.5 | 64.8 | 62.0 | [model](https://drive.google.com/file/d/1Ludrfmd4Dsn_5uxmm9cGFE9uW5ZyehLQ/view?usp=sharing) | 

## Requirements, Installation and Usage

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==1.13.1
- [mmcv](https://github.com/open-mmlab/mmcv)==1.7.0
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.28.2
- [mmseg](https://github.com/open-mmlab/mmsegmentation)==0.30.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.0.0.rc6

### Installation

**Clone the repository**
```
git clone https://github.com/gopi-erabati/RetFormer.git
cd RetFormer
```

```
mkvirtualenv retformer

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmcv-full==1.7.0

pip install -r requirements.txt
```

For evaluation on Waymo, please follow the below code to build the binary file `compute_detection_metrics_main` for metrics computation and put it into ```mmdet3d_plugin/core/evaluation/waymo_utils/```.
```
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../RetFormer/mmdet3d_plugin/core/evaluation/waymo_utils/
```

### Data
Follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) to prepare the [Waymo](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) and [KITTI](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html) datasets and symlink the data directories to `data/` folder of this repository.
**Warning:** Please strictly follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Training
#### Waymo dataset 
- Single GPU training
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/retformer_waymo_D1_2x_3class.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/retformer_waymo_D1_2x_3class.py {GPU_NUM} --work-dir {WORK_DIR}`
#### KITTI dataset
- Single GPU training
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/retformer_kitti_3class.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/retformer_kitti_3class.py {GPU_NUM} --work-dir {WORK_DIR}`

### Testing
#### Waymo dataset 
- Single GPU testing
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/retformer_waymo_D1_2x_3class.py /path/to/ckpt --eval waymo`
- Multi GPU training
  `tools/dist_test.sh configs/retformer_waymo_D1_2x_3class.py /path/to/ckpt {GPU_NUM} --eval waymo`
#### KITTI dataset
- Single GPU testing
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/retformer_kitti_3class.py /path/to/ckpt --eval mAP`
- Multi GPU training
  `tools/dist_test.sh configs/retformer_kitti_3class.py /path/to/ckpt {GPU_NUM} --eval mAP`

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [FlatFormer](https://github.com/mit-han-lab/flatformer).

## Reference
```
@ARTICLE{retformerGopi,
  author={Erabati, Gopi Krishna and Araujo, Helder},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={RetFormer: Embracing Point Cloud Transformer With Retentive Network}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Point cloud compression;Transformers;Three-dimensional displays;Task analysis;Feature extraction;Computational modeling;Object detection;Point cloud transformer;retention;LiDAR;3D object detection;autonomous driving},
  doi={10.1109/TIV.2024.3417260}}

```
