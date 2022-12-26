---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 5"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

This is part of a series of posts breaking down the paper [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131), and providing (hopefully) well-annotated source code to aid in understanding.

- [Part 1]({% post_url 2022-12-04-pytorch-guide-plenoxels-nerf-part-1 %})
- [Part 2]({% post_url 2022-12-05-pytorch-guide-plenoxels-nerf-part-2 %})
- [Part 3]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-3 %})
- [Part 4]({% post_url 2022-12-18-pytorch-guide-plenoxels-nerf-part-4 %})
- [Part 5 (this one)]({% post_url 2022-12-19-pytorch-guide-plenoxels-nerf-part-5 %})

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We have built our volumetric renderer, trained it on a single training image; it is now time to extend the idea to **multiple training images** obtained by cameras around the model. We will also add **Total Variance regularisation** to the loss calculations in this post, as well as work on **pruning voxels** which are less than a specific threshold.

We will show results of the reconstruction using our simple cube.

**The code for this article can be found here: [Volumetric Rendering Code with TV Regularisation](https://github.com/asengupta/avishek.net/blob/master/code/pytorch-learn/plenoxels/volumetric-rendering-with-tv-regularization.py)**

The pictures below compare the reconstruction with the actual ground truth.

![Crude Reconstruction of Cube](/assets/images/reconstruction-cube-scaled.gif)![Original Cube](/assets/images/training-cube-scaled.gif)

### Notes on the Reconstruction
From certain angles, the reconstruction does not look like a cube, or only partially like one. More epochs and training images are obviously needed. The paper mentions running 12800 batches before it converges sufficiently.

### Constructing model parameters differently
We simply set ```requires_grad=True``` for the voxels which should be considered for updates, and set it to ```False``` for the others. See [```modify_grad()```](https://github.com/asengupta/avishek.net/blob/master/code/pytorch-learn/plenoxels/volumetric-rendering-with-tv-regularization.py#L898) for how this done.

### Parallelising Volumetric Calculations using PyTorch

We could have either used [Pathos](https://pypi.org/project/pathos/) or PyTorch's multiprocessing library. Unfortunately, **PyTorch cannot perform distributed gradient propagation** (there is an experimental feature for it, but we haven't tried it), so we ended up using Pathos. It will probably more effective if the volumetric calculations are more intensive; currently, the distributed version takes longer than the serial version.

Look at [```render_parallel()```](https://github.com/asengupta/avishek.net/blob/509acb6dda55c96dca7f4f8bc2da9d88b16c2f3f/code/pytorch-learn/plenoxels/volumetric-rendering-with-tv-regularization.py#L596) to see how Pathos is used.

### Debugging Out of Memory issues
[Memory Profiler](https://pypi.org/project/memory-profiler/) makes it very easy to track incremental per-line memory allocation.

User [ptrblck](https://discuss.pytorch.org/u/ptrblck) says this:

> "If you are seeing an increased memory usage of 10GB in each iteration, you are most likely storing the computation graph accidentally by e.g. appending the loss (without detaching it) to a ```list``` etc."

The moral is: do not store losses as-is: detach them to get the raw numbers, otherwise they will get stored with the entire computational graph.

### Conclusion
We have incorporated Total Variance regularisation in our implementation of the paper, and implemented a proper training schedule using multiple training images. The reconstructions are crude, but they demonstrate how the spherical harmonics can be learned and varied from different angles. We have also incorporated pruning voxels.

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Pathos Documentation](https://pathos.readthedocs.io/en/latest/pathos.html)
- [Memory Profiler](https://pypi.org/project/memory-profiler/)
- [OOM PyTorch](https://discuss.pytorch.org/t/gpu-out-of-memory-after-i-call-the-backward-function/139280)
