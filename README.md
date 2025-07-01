# [CVPR 2025-Highlight] Samba: A Unified Mamba-based Framework for General Salient Object Detection [[PDF]](https://www.kerenfu.top/sources/CVPR2025_Samba.pdf)
Jiahao He, Keren Fu, Xiaohong Liu, Qijun Zhao<br />
<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/intro_1.png" style="width: 80%;"/>

## ✈ Overview
We are the first to adapt state space models to SOD tasks, and propose a novel unified framework based on the pure Mamba architecture to flexibly handle general SOD tasks. We propose a saliency-guided Mamba block (SGMB), incorporating a spatial neighboring scanning (SNS) algorithm, to maintain spatial continuity of salient patches, thus enhancing feature representation. We propose a context-aware upsampling (CAU) method to promote hierarchical feature alignment and aggregations by modeling contextual dependencies.

<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/overview.png">

## ✈ Environmental Setups
`PyTorch 1.13.1 + CUDA 11.7`. Please install corresponding PyTorch and CUDA versions.

VMamba-S backbone weights：[[baidu](https://pan.baidu.com/s/1SaEV237VCzSEn558gEBiXg)，提取码：zsxa]

Full Samba weights：[[baidu](https://pan.baidu.com/s/15787DVEmW59ftztopv-yMg)，提取码：bkvw]

## ✈ Data Preparation
### 1. RGB SOD
For RGB SOD, we employ the following datasets to train our model: the training set of **DUTS** for `RGB SOD`. 
For testing the RGB SOD task, we use **DUTS**, **ECSSD**, **HKU-IS**, **PASCAL-S**, **DUT-O**. [[baidu](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g)，提取码：x7kn]

### 2. RGB-D SOD
For RGB-D SOD, we employ the following datasets to train our model concurrently: the training sets of **NJU2K**, **NLPR**, **DUT-RGBD** for `RGB-D SOD`. 
For testing the RGB SOD task, we use **NJU2K**, **NLPR**, **DUT-RGBD**, **SIP**, **STERE**. [[baidu](https://pan.baidu.com/s/1ibrO3CS7rn7bJUAy8hM9mQ)，提取码：8b9c]

### 3. RGB-T SOD
For RGB-T SOD, we employ the training set of **VT5000** to train our model, and the testing of **VT5000**, **VT821**, **VT1000** are utilized for testing. [[baidu](https://pan.baidu.com/s/1PKW5d_Yr5NFEnq9Q82HitA)，提取码：xhrm]

### 4. VSOD
For VSOD, we employ the training sets of **DAVIS**, **DAVSOD**, **FBMS** to train our model concurrently, and the testing of **DAVIS**, **DAVSOD**, **FBMS**, **Seg-V2**, **VOS** are utilized for testing. [[baidu](https://pan.baidu.com/s/1zQ-vuDnSfRzJ1T_T-hh7sA)，提取码：kcmu]

### 5. RGB-D VSOD
For RGB-D VSOD, we employ the training sets of **RDVS**, **DVisal**, **Vidsod_100** to train our model individually, and the testing of **RDVS**, **DVisal**, **Vidsod_100** are utilized for testing individually. [[baidu](https://pan.baidu.com/s/1VRL3jk7AsQCkL26hwg1rZA)，提取码：q9ty]

## ✈ Prediction
All evaluated saliency maps are put here：[[baidu](https://pan.baidu.com/s/1NA9_ZtA_M4WHugPt92MrSA)，提取码：bdhi]

## ✈ Visual Results
<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/visual_1.png" style="width: 80%;"/>
<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/visual_2.png" style="width: 80%;"/>
<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/visual_3.png" style="width: 80%;"/>
<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/visual_4.png" style="width: 80%;"/>
<img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/visual_5.png" style="width: 80%;"/>

## ✈ Citation
If you use Samba in your research or wish to refer our work, please use the following BibTeX entry.
```
@article{he2025samba,
  title={Samba: A Unified Mamba-based Framework for General Salient Object Detection},
  author={Jiahao He, Keren Fu, Xiaohong Liu, Qijun Zhao},
  year={2025}
}
```
