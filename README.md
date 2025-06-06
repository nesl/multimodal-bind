# MMBind: Multimodal Learning with Incomplete Data
This is a repo for SenSys 2025 paper: " MMBind: Unleashing the Potential of Distributed and Heterogeneous Data for Multimodal Learning in IoT ".

# Requirements
The program has been tested in the following environment: 
* Python 3.9.7
* Pytorch 1.8.1
* torchvision 0.9.1
* sklearn 0.24.2
* opencv-python 4.5.5
* numpy 1.20.3

# MMbind Overview
<p align="center" >
	<img src="https://github.com/nesl/multimodal-bind/blob/main/mmbind-overview.png" width="700">
</p>

* Stage 1 of MMbind: 
	* pairing incomplete data with shared modalities;
* Stage 2 of MMbind: 
	*  weighted contrastive learning with heterogeneous data.

# Project Strcuture
* Cross-subject Evaluation:
  * [Evaluation on UTD dataset](https://github.com/nesl/multimodal-bind/blob/main/UTD/UTD-README.md)
  * [Evaluation on MMFI dataset](https://github.com/nesl/multimodal-bind/tree/main/MMFI)
  * [Evaluation on PAMAP2 dataset](https://github.com/nesl/multimodal-bind/blob/main/PAMAP2/PAMAP2-README.md)
  * [Evaluation on SUNRGBD dataset](https://github.com/nesl/multimodal-bind/tree/main/SUN_RGBD_Main)
* Cross-dataset Evaluation:
  * Training on binded MotionSense and Shoaib datasets, testing on RealWorld dataset.

