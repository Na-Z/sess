# SESS: Self-Ensembling Semi-Supervised 3D Object Detection
Created by <a href="https://github.com/Na-Z" target="_blank">Na Zhao</a> from 
<a href="http://www.nus.edu.sg/" target="_blank">National University of Singapore</a>

![teaser](https://github.com/Na-Z/sess/blob/master/teaser.jpg)

## Introduction
This repository contains the PyTorch implementation for our CVPR 2020 Paper 
"SESS: Self-Ensembling Semi-Supervised 3D Object Detection" by Na Zhao, Tat Seng Chua, Gim Hee Lee 
[[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_SESS_Self-Ensembling_Semi-Supervised_3D_Object_Detection_CVPR_2020_paper.pdf) 
| [supp](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Zhao_SESS_Self-Ensembling_Semi-Supervised_CVPR_2020_supplemental.pdf)]

The performance of existing point cloud-based 3D object detection methods heavily relies on large-scale high-quality 3D 
annotations. However, such annotations are often tedious and expensive to collect. Semi-supervised learning is a good 
alternative to mitigate the data annotation issue, but has remained largely unexplored in 3D object detection. Inspired 
by the recent success of self-ensembling technique in semi-supervised image classification task, we propose SESS, a 
self-ensembling semi-supervised 3D object detection framework. Specifically, we design a thorough perturbation scheme 
to enhance generalization of the network on unlabeled and new unseen data. Furthermore, we propose three consistency 
losses to enforce the consistency between two sets of predicted 3D object proposals, to facilitate the learning of 
structure and semantic invariances of objects. Extensive experiments conducted on SUN RGB-D and ScanNet datasets 
demonstrate the effectiveness of SESS in both inductive and transductive semi-supervised 3D object detection. Our SESS 
achieves competitive performance compared to the state-of-the-art fully-supervised method by using only 50% labeled data.

## Setup
- Install `python` --This repo is tested with `python 3.6.8`.
- Install `pytorch` with CUDA -- This repo is tested with `torch 1.1`, `CUDA 9.0`. 
It may wrk with newer versions, but that is not gauranteed.
- Install `tensorflow` (for Tensorboard) -- This repo is tested with `tensorflow 1.14`.
- Compile the CUDA layers for PointNet++, which is used in the backbone network:
    ```
    cd pointnet2
    python setup.py install
    ```
- Install dependencies
    ```
    pip install -r requirements.txt
    ```

## Usage
### Data preparation
For SUNRGB-D, follow the [README](https://github.com/Na-Z/sess/blob/master/sunrgbd/README.md) under `sunrgbd` folder.

For ScanNet, follow the [README](https://github.com/Na-Z/sess/blob/master/scannet/README.md) under `scannet` folder.

### Running experiments
For SUNRGB-D, using the following command to train and evaluate:
    
    python scripts/run_sess_sunrgbd.py

For ScanNet, using the following command to train and evaluate:
    
    python scripts/run_sess_scannet.py

Note that we have included the pretaining phase, training phase, and two evaluation phases 
 (inductive and transductive semi-supervised learning) as four functions in each script. 
You are free to uncomment any function execution line to skip the corresponding phase. 

## Citation
Please cite our paper if it is helpful to your research:

    @inproceedings{zhao2020sess,
      title={SESS: Self-Ensembling Semi-Supervised 3D Object Detection},
      author={Zhao, Na and Chua, Tat-Seng and Lee, Gim Hee},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={11079--11087},
      year={2020}
    }


## Acknowledgements
Our implementation leverages on the source code from the following repositories:
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
- [Mean teachers are better role models](https://github.com/CuriousAI/mean-teacher)
- [Pointnet2/Pointnet++ PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
