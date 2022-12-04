---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Building a Volume Renderer"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

The relevant paper is [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We will also use [this explanation](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks) to understand some parts of this paper a little better.

Before we get into the implementations of the paper proper, we will need a game plan. This game plan will include some theoretical background that we will have to go through to implement parts of this paper. The theoretical background will include:

- Camera Model
- Volumetric Rendering Model
- Spherical Harmonics
- Regularisation

In this specific post, however, we will start building out a simple volumetric renderer. On the way, we will also discuss the pinhole camera model, on which most of our rendering will be based on.

### The Pinhole Camera Model and some Linear Algebra

The pinhole camera model has the following characteristics.

- The model exists in the world, and is expressed in the world coordinate system. This is usually our default three-dimensional basis.
- The camera exists somewhere in the world, and has its own coordinate system, the camera coordinate system. This is characterised by the location of the camera, the focal length of the camera, and the three-dimensional basis for the camera.
- The screen of the camera (which is basically where the image is formed) has its own two-dimensional coordinate system.

The challenge is this: we have a point in 3D space expressed in the world coordinate system, let's call it $$X_W$$; we want to know what this point will translate to on the 2D coordinate system of the camera screen/film. At a very high level, given all the information about the camera and the world, we want to know about the camera transform matrix $$P$$.

$$
X_{2D}=PX_W
$$

We need to do the following steps:

- Express $$X_W$$ in the coordinate system of the camera as $$X_C$$. These are the extrinsic parameters of the camera.
- Express $$X_C$$ in the coordinate system of the camera screen, taking into account focal length. These are the intrinsic parameters of the camera.

The camera is characterised first by the camera center $$C$$. The first step is translating the entire world so that the origin is now at the camera. This is simply done by calculating $$X_W-C$$.

The camera is also characterised by its basis, which is essentially three 3D vectors. Now that the camera is at the center, we need to rotate the world so that everything in it is expressed using the camera's coordinate system. How do we achieve this change of basis?

We have discussed change of basis before in a few articles. Specifically see [The Gram-Schmidt Orthogonalisation]({% post_url 2021-05-27-gram-scmidt-orthogonalisation %}) and [Geometry of the Multivariate Gaussian Distribution]({% post_url 2021-08-30-geometry-of-multivariate-gaussian %}).

Specifically, we have an arbitrary basis $$B$$ in $$n$$-dimensional space, and let there be a vector $$v$$ expressed in the world coordinate system. We'd like to be able to represent $$v$$ using $$B$$'s coordinate system. Let's assume that $$v_B$$ is the vector $$v$$ expressed in $$B$$'s coordinate system.

Then, we can write:

$$
B v_B=v \\
\Rightarrow B^{-1}B v_B = B^{-1} v \\
\Rightarrow v_B = B^{-1} v
$$

Thus, multiplying $$B^{-1}$$ with our original world space vector $$v$$ gives us the same vector but expressed in the coordinate space of basis $$B$$.

Thus, the rotation that we need to do is:

$$
X_C=B^{-1} (X_W - C)
$$

A note on convention: the Z-axis of the camera always points in the direction the camera is pointing in: the X- and Y-axes are reserved for mapping the image onto the camera screen.

Now we look at the intrinsic parameters, specifically the focal length and the mapping to the screen (which is where we will finally see the image). The pinhole camera model is represented by the following diagram.

![Voxel Cube](/assets/images/voxel-cube.png)
![Very Basic Volumetric Rendering of Cube](/assets/images/basic-volumetric-rendering-cube.png)

The following code plays around with the pinhole camera model and sets up a very basic (maybe even contrived) volumetric rendering model.

```python
{% include_absolute '/code/pytorch-learn/plenoxels/camera2.py' %}
```

### Building the Voxel Data Structure

We will store opacity and the spherical harmonic coefficients.

### Incorporating the Volumetric Rendering Model

We will take a bit of a pause to understand the optical model involved in volumetric rendering since it is essential to the actual rendering and the subsequent calculation of the loss. This article follows [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE) quite a bit. Solving differential equations is involved, but for the most part, you should be able to skip to the results, if you are not interested in the gory details.

### Calculating Loss

### Incorporating Training Images

### Incorporating Trilinear Interpolation

### Optimising using RMSProp

### References

- [Camera Matrix - Kris Kitani](https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE)
