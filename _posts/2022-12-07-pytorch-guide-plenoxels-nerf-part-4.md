---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 4"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We have built our volumetric renderer; it is now time to start calculating loss from training images. We start off with trying out some test images, the source of which will be our renderer itself, as a kind of control set.

### Calculating Loss

### Refactoring the Data Structure

### Incorporating A Single Training Image

### Optimising using RMSProp

### Reconstruction Results

![Plenoxels Flat Surface Training](/assets/images/plenoxels-flat-surface-training.png)
![Plenoxels Flat Surface 5 Epochs 1 Image](/assets/images/plenoxel-flat-surface-1-image-5-epochs.png)

### References

- [Spherical Harmonics](https://patapom.com/blog/SHPortal/#fn:2)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
