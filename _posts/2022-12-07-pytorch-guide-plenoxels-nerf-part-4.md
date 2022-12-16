---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 4"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

This is part of a series of posts breaking down the paper [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131), and providing (hopefully) well-annotated source code to aid in understanding.

- [Part 1]({% post_url 2022-12-04-pytorch-guide-plenoxels-nerf-part-1 %})
- [Part 2]({% post_url 2022-12-05-pytorch-guide-plenoxels-nerf-part-2 %})
- [Part 3]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-3 %})
- [Part 4 (this one)]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-4 %})

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We have built our volumetric renderer; it is now time to start calculating loss from training images. We start off with trying out some test images, the source of which will be our renderer itself, as a kind of control set.

### Calculating Loss

We will add in regularisation later. Let's test the loss on images generated with the exact renderer which generates the test images. The loss should be essentially zero.

### Refactoring the Data Structure
We will also use the original renderer to generate ground truth that we can use to validate that the refactored renderer still generates correct images.

![Plenoxels Data Structure](/assets/images/plenoxels-data-structures.png)

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering-with-loss-interpolating.py' %}
```

### Incorporating A Single Training Image

### Optimising using RMSProp

### A Tale of Debugging

- Using PyTorchViz
- Interpreting the gradient graph
- Replicating the issue
- Whittling it down to the smallest reproducible example

```python
{% include_absolute '/code/pytorch-learn/plenoxels/custom-model-bug.py' %}
```


### Reconstruction Results

these are using just one image, and the reconstruction is correct only from that specific viewpoint

![Plenoxels Flat Surface Training Image](/assets/images/plenoxels-flat-surface-training.png)
![Plenoxels Flat Surface 5 Epochs 1 Image](/assets/images/plenoxel-flat-surface-1-image-5-epochs.png)

![Plenoxels Cube Image](/assets/images/plenoxels-cube-training.png)
![Plenoxels Cube 5 Epochs 1 Image](/assets/images/plenoxels-cube-5-epochs-1-image.png)

![Plenoxels Multicoloured Cube Image](/assets/images/plenoxels-multicoloured-cube-training.png)
![Plenoxels Multicoloured Cube 15 Epochs 1 Image](/assets/images/plenoxels-multicoloured-cube-1-image-15-epochs.png)

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [How to use Pytorch as a general optimizer](https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb)
