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

### Addressing some bugs
- Ray calculations wrong
- RGB calculations using Spherical Harmonics routinely lie outside $$[0,1]$$.
  - The paper does not elaborate a whole lot on how to clamp these values, beyond using a **ReLU** for linearity. However, the original NERF paper uses a sigmoid function, which is what we will use here. It's easy to implement, differentiable, and gives good results.
- Debugging Tips
  - Find the simplest program which can reproduce the issue, preferably in a self-contained code fragment.
  - Track gradient function propagation throughout calculations.
  - Check gradient propagation graph using PyTorchViz.
  - For loops, reduce it to a single degenerate / problematic value and trace.
  - Beware of divide-by-zero errors.
  - Before refactoring complicated rendering logic, have a ground truth version at all times to check your refactored work.

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

![Plenoxels Refactored Cube Image](/assets/images/plenoxels-refactored-cube-training.png)
![Plenoxels Refactored Cube 3 Epochs 1 Image](/assets/images/plenoxels-refactored-cube-reconstruction-1-image-3-epochs.png)

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [How to use Pytorch as a general optimizer](https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb)
