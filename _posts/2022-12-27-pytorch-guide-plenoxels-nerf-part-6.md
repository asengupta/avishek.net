---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 6"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

This is part of a series of posts breaking down the paper [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131), and providing (hopefully) well-annotated source code to aid in understanding.

- [Part 1]({% post_url 2022-12-04-pytorch-guide-plenoxels-nerf-part-1 %})
- [Part 2]({% post_url 2022-12-05-pytorch-guide-plenoxels-nerf-part-2 %})
- [Part 3]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-3 %})
- [Part 4]({% post_url 2022-12-18-pytorch-guide-plenoxels-nerf-part-4 %})
- [Part 5]({% post_url 2022-12-19-pytorch-guide-plenoxels-nerf-part-5 %})
- [Part 6 (this one)]({% post_url 2022-12-27-pytorch-guide-plenoxels-nerf-part-6 %})

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). In this sequel, we address some remaining aspects of the paper, before concluding with our study of this paper. We specifically consider the following:

- **Voxel pruning:** This will probably require us to modify our core data structure to be a little more efficient, because it will involve storing all the transmittances of the entire training set per epoch, and then zeroing out candidate voxels.
- **Encouraging voxel sparsity:** Adding more regularisation terms will encourage sparsity of voxels. In the paper, the Cauchy loss is incorporated to speed up computations.
- **Coarse-to-fine resolution scaling:** This will be needed to better resolve the fine structure of our training scenes. At this point, we are working with a very coarse low resolution of $$40 \times 40 \times 40$$. We can get higher resolutions than this, but this will need more work, and more computations.

**The code for this article can be found here: [Volumetric Rendering Code with TV Regularisation](https://github.com/asengupta/avishek.net/blob/master/code/pytorch-learn/plenoxels/volumetric-rendering-with-tv-regularization.py)**

![Cube Reconstruction using correct volumetric rendering formula](/assets/images/cube-reconstruction-correct-rendering-large.gif)![Original Cube](/assets/images/training-cube-scaled.gif)
### Conclusion

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
