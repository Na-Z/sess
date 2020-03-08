# SESS: Self-Ensembling Semi-Supervised 3D Object Detection
Created by <a href="https://github.com/Na-Z" target="_blank">Na Zhao</a> from 
<a href="http://www.nus.edu.sg/" target="_blank">National University of Singapore</a>

![teaser](https://github.com/Na-Z/sess/blob/master/teaser.jpg)

## Introduction
This repository contains the PyTorch implementation for our CVPR 2020 Paper 
"SESS: Self-Ensembling Semi-Supervised 3D Object Detection" by Na Zhao, Tat Seng Chua, Gim Hee Lee (arXiv report 
[here](https://arxiv.org/abs/1912.11803)).

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

**Code will come soon.**