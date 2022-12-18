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

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We have built our volumetric renderer, trained it on a single training image; it is now time to extend the idea to multiple training images obtained by cameras around the model. We will also add TV regularisation to the loss calculations in this post.

**The code for this article can be found here: [Volumetric Rendering Code](https://github.com/asengupta/avishek.net/blob/master/code/pytorch-learn/plenoxels/volumetric-rendering-with-loss-interpolating.py)**

### Conclusion
We have come quite far in our implementation of the paper. We will progress to training using multiple training images in the sequel and (probably) add TV regularisation.

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [How to use Pytorch as a general optimizer](https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb)
