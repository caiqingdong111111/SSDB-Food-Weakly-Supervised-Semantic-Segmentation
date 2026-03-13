# SSDB-Net: Weakly Supervised Semantic Segmentation for Food Images

Official implementation of the paper:

SSDB-Net: A Single-Step Dual Branch Network for Weakly Supervised Semantic Segmentation of Food Images

Accepted at IEEE International Workshop on Multimedia Signal Processing (MMSP) 2023.

---

## Overview

Weakly supervised semantic segmentation (WSSS) aims to generate pixel-level segmentation masks using only weak annotations such as image-level labels. However, existing WSSS methods often struggle with food images due to high intra-class variability, occlusion, and complex visual structures.

This repository provides the implementation of SSDB-Net, a single-stage dual-branch network designed to improve weakly supervised segmentation performance for food ingredient recognition.

The proposed model integrates:

- A classification branch for semantic supervision
- A segmentation branch for pixel-level mask generation
- Improved feature learning for ingredient localisation

The method improves segmentation performance on food datasets while requiring only image-level annotations.

---

## Dataset

Experiments are conducted using the FoodSeg103 dataset, which contains:

- 104 ingredient categories  
- 7,118 images in total  
- 4,983 training images  
- 2,135 test images  

Please download the dataset from the official source:
https://datasetninja.com/food-seg-103#download

If you use this dataset in your work, please cite the original dataset paper.

---

## Training

To train SSDB-Net:

python training classification branch.py and 
python training semantic branch.py

---

## Inference

To run inference:

python infer.py

The predicted segmentation mask will be saved in the output directory.

---

## Results

SSDB-Net achieves improved performance on the FoodSeg103 benchmark.

Method       mIoU
SEAM         11.49
SSDB-Net     14.79

---

## Paper

If you find this work useful, please cite:

@inproceedings{cai2023ssdb,
  title={SSDB-Net: a single-step dual branch network for weakly supervised semantic segmentation of food images},
  author={Cai, Qingdong and Abhayaratne, Charith},
  booktitle={2023 IEEE 25th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}

---

## Acknowledgements

We thank the authors of the FoodSeg103 dataset for making the dataset publicly available.

Some reference code and implementation ideas were inspired by the following repository:
https://github.com/gyguo/awesome-weakly-supervised-semantic-segmentation

We thank the authors for maintaining this valuable resource for the weakly supervised semantic segmentation community.
---

## License

This project is released under the MIT License.

---

## Support

If you have questions or find any issues, please open an issue in this repository.
