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

**Applying to a real world dataset**
For our field testing we pick a model from the Amazon Berkeley Objects dataset. The ABO Dataset is made available under the [**Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)**](https://creativecommons.org/licenses/by-nc/4.0/), and is available [here](https://amazon-berkeley-objects.s3.amazonaws.com/index.html).

We have picked 72 views of a full $$360 \degree$$ fly-around of the object, and run it through our code.

### Notes on the Code
- **Fixing a Transmittance calculation bug:**
- **Refactoring volumetric rendering to use matrices instead of loops:**
- **Configuring empty voxels to be light or dark:**
- **Adding the ```pruned``` attribute:**


![Table Reconstruction - Single View - 15 Epochs](/assets/images/out-table-single-large.gif)

![Ground Truth](/assets/images/plenoxels-table-training-single.png)
![Rough Reconstruction](/assets/images/plenoxels-table-reconstruction-single.png)


### Cauchy Regularisation

The Cauchy Loss Function is introduced in the paper [Robust subspace clustering by Cauchy loss function](https://arxiv.org/abs/1904.12274).

**Reconstruction without Cauchy Regularisation**
![Cube Reconstruction using correct volumetric rendering formula without Cauchy Loss](/assets/images/cube-reconstruction-correct-rendering-large.gif)![Original Cube](/assets/images/training-cube-scaled.gif)

**Reconstruction with Cauchy Regularisation**
![Cube Reconstruction using correct volumetric rendering formula with Cauchy Loss](/assets/images/out-cube-cauchy-large.gif)![Original Cube](/assets/images/training-cube-scaled.gif)


### Conclusion

### References

- [Amazon Berkeley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Robust subspace clustering by Cauchy loss function](https://arxiv.org/abs/1904.12274)
