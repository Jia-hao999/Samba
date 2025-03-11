# [CVPR 2025] Samba: A Unified Mamba-based Framework for General Salient Object Detection
Jiahao He, Keren Fu, Xiaohong Liu, Qijun Zhao<br />

<!--**Approach**: [[arxiv Paper]](https://arxiv.org/pdf/2311.15011.pdf)-->

## ✈ Overview
We are the first to adapt state space models to SOD tasks, and propose a novel unified framework based on the pure Mamba architecture to flexibly handle general SOD tasks. We propose a saliency-guided Mamba block (SGMB), incorporating a spatial neighboring scanning (SNS) algorithm, to maintain spatial continuity of salient patches, thus enhancing feature representation. We propose a context-aware upsampling (CAU) method to promote hierarchical feature alignment and aggregations by modeling contextual dependencies.
<img src="https://github.com/Jia-hao999/Samba/blob/main/overview.png">

## ✈ Environmental Setups
`PyTorch 1.13.1 + CUDA 11.7`. Please install corresponding PyTorch and CUDA versions.

## ✈ Data Preparation
### 1. RGB SOD
For RGB SOD, we employ the following datasets to train our model: the training set of **DUTS** for `RGB SOD`. 
For testing the RGB SOD task, we use **DUTS**, **ECSSD**, **HKU-IS**, **PASCAL-S**, **DUT-O**. You can directly download these datasets from [[baidu](https://pan.baidu.com/s/18PVmR-Z2wwVtTZEr14aVJg?pwd=m9ht),PIN:m9ht].

### 2. RGB-D SOD
For RGB-D SOD, we employ the following datasets to train our model concurrently: the training sets of **NJU2K**, **NLPR**, **DUT-RGBD** for `RGB-D SOD`. 
For testing the RGB SOD task, we use **NJU2K**, **NLPR**, **DUT-RGBD**, **SIP**, **STERE**. You can directly download these datasets from [[baidu](https://pan.baidu.com/s/18PVmR-Z2wwVtTZEr14aVJg?pwd=m9ht),PIN:m9ht].

### 3. RGB-T SOD
For RGB-T SOD, we employ the training set of **VT5000** to train our model, and the testing of **VT5000**, **VT821**, **VT1000** are utilized for testing. You can directly download these datasets from [[baidu](https://pan.baidu.com/s/18PVmR-Z2wwVtTZEr14aVJg?pwd=m9ht),PIN:m9ht].

### 4. VSOD
For VSOD, we employ the training sets of **DAVIS**, **DAVSOD**, **FBMS** to train our model concurrently, and the testing of **DAVIS**, **DAVSOD**, **FBMS**, **Seg-V2**, **VOS** are utilized for testing. You can directly download these datasets from [[baidu](https://pan.baidu.com/s/18PVmR-Z2wwVtTZEr14aVJg?pwd=m9ht),PIN:m9ht].

### 5. RGB-D VSOD
For RGB-D VSOD, we employ the training sets of **RDVS**, **DVisal**, **Vidsod_100** to train our model individually, and the testing of **RDVS**, **DVisal**, **Vidsod_100** are utilized for testing individually. You can directly download these datasets from [[baidu](https://pan.baidu.com/s/18PVmR-Z2wwVtTZEr14aVJg?pwd=m9ht),PIN:m9ht].

## ✈ Experiments
Run `python train_test_eval.py --Training True --Testing True --Evaluation True` for training, testing, and evaluation which is similar to VST.

## ✈ Citation
If you use Samba in your research or wish to refer our work, please use the following BibTeX entry.
```
@article{he2025samba,
  title={Samba: A Unified Mamba-based Framework for General Salient Object Detection},
  author={Jiahao He, Keren Fu, Xiaohong Liu, Qijun Zhao},
  journal={arXiv preprint arXiv:2311.15011},
  year={2025}
}
```
