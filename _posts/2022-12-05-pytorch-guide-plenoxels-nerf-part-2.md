---
title: "Plenoxels and Neural Radiance Fields using PyTorch: Part 2"
author: avishek
usemathjax: true
tags: ["Machine Learning", "PyTorch", "Programming", "Deep Learning", "Neural Radiance Fields", "Machine Vision"]
draft: false
---

This is part of a series of posts breaking down the paper [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131), and providing (hopefully) well-annotated source code to aid in understanding.

- [Part 1]({% post_url 2022-12-04-pytorch-guide-plenoxels-nerf-part-1 %})
- [Part 2 (this one)]({% post_url 2022-12-05-pytorch-guide-plenoxels-nerf-part-2 %})
- [Part 3]({% post_url 2022-12-07-pytorch-guide-plenoxels-nerf-part-3 %})
- [Part 4]({% post_url 2022-12-18-pytorch-guide-plenoxels-nerf-part-4 %})

We continue looking at [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131). We set up a very contrived opacity model in our last post, which gave us an image like this:

![Very Basic Volumetric Rendering of Cube](/assets/images/basic-volumetric-rendering-cube.png)

It looks very chunky; this is due to the finite resolution of the voxel world. More importantly, it does not incorporate a proper volumetric rendering model, which we will discuss in this post.

A point about **performance** and readability is in order here. The implementation that we are building is not going to be very fast, and the code is certainly not as optimised as it should be. However, the focus is on the explainability of the process, and how that translates into code. We may certainly delve into utilising the GPU more effectively at some point, but let us get the basics of the paper sorted first before attempting that.

Another point about cleanliness; much of the intermediate code on the way to a full implementation is going to be somewhat messy, and is presented as-is, without removing commented code, the odd ```print()``` statement and any number of variables which might not end up getting used. I've also always had a Python console open for really quick experiments with the PyTorch API, before lifting an expression from there and planting it in the code proper.

In this article, we will discuss the following:

- Building the Voxel Data Structure: First Pass
- Incorporating the Volumetric Rendering Model
- Incorporating Trilinear Interpolation
- Fixing Volumetric Rendering Artifacts

### Building the Voxel Data Structure: First Pass

In our final model, we'd like to store opacity and the **spherical harmonic coefficients** per voxel. Opacity will be denoted by $$\sigma \in [0,1]$$ and is as straightforward as it sounds. Let's talk quickly about how we should encode colour. Each colour channel (R,G,B) will have its own set of 9 spherical harmonic coefficients; this gives us 27 numbers to store. Add $$\sigma$$ to this, and each voxel is essentially represented by 28 numbers.

For the purposes of our initial experiments, we will not worry about the spherical harmonics, and simply represent colour using RGB triplets ranging from 0.0 to 1.0. We will still have opacity $$\sigma$$. Add the three numbers necessary to represent a voxel's position in 3D space, and we have 3+1+3=7 numbers to completely characterise a voxel.

We will begin by implementing a very simple and naive data structure to represent our world: a $$(i \times j \times k)$$ tensor cube. Indexing this structure using ```world[x,y,z]``` helps us locate a particular voxel in space. Each entry in this world will thus contain a tensor containing 4 numbers, the opacity $$\sigma$$ and the RGB triplet.

In this article, we will not make full use of the RGB triplet; it will mostly be there for demonstration purposes before being replaced by the proper spherical harmonics model going forward.

Addressing a voxel in a position which does not exist in this world will give us $$[0,0,0,0]$$. 

### Volumetric Rendering Model

We will take a bit of a pause to understand the **optical model** involved in volumetric rendering since it is essential to the actual rendering and the subsequent calculation of the loss. This article follows [Optical Model for Volumetric Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE) quite a bit. Solving differential equations is involved, but for the most part, you should be able to skip to the results, if you are not interested in the gory details: the relevant results are $$\eqref{eq:accumulated-transmittance}$$ and $$\eqref{eq:sample-transmittance}$$

![Volumetric Rendering Model with Absorption and Emission](/assets/images/volumetric-rendering-model.png)

Consider a cylinder with cross-sectional area $$E$$. Light flows in through the cylinder from the left. Assume a thin slab of the cylinder of length $$\Delta s$$. Let $$C$$ be the intensity of light (flux per unit area) entering this slab from the left. Let us assume a bunch of particles in this slab cylinder. The density of particles in this slab is $$\rho$$. These particles project circular occlusions of area $$A$$. If this slab is thin enough, the occlusions do not overlap.

The number of particles in this slab is then $$\rho E \Delta s$$. The total area of occlusion of these particles are $$A \rho E \Delta s$$. The total area of this cross section is $$E$$. Thus, the fraction of light occluded is $$A \rho E \Delta s / E = A \rho \Delta s$$. Then, the intensity of light exiting the slab is $$C A \rho \Delta s$$.

Let us also assume that these particles emit some light, with intensity $$c$$ per unit area. Then the total flux emitted is $$c A \rho E \Delta s$$. The intensity (flux per unit area) is thus $$c A \rho E \Delta s / E = c A \rho \Delta s$$

Then, the delta change of intensity is:

$$
\Delta C = c A \rho \Delta s - C A \rho \Delta s \\
\frac{\Delta C}{\Delta s} = c A \rho - C A \rho
$$

In the limit as the slab becomes infinitesmally thin, i.e., $$\Delta s \rightarrow 0$$, we get:

$$
\frac{dC}{ds} = c A \rho - C A \rho
$$

Define $$\sigma = A \rho$$ to get:

$$
\frac{dC}{ds} = \sigma c - \sigma C \\
\begin{equation}
\frac{dC(s)}{ds} = \sigma(s) c(s) - \sigma(s) C(s)
\label{eq:transmittance}
\end{equation}
$$

### Solving the Volumetric Rendering Model

This is the transmittance equation which models emission and absorption. Let's solve it fully, so I can show you the process, even though it's a very ordinary first-order differential equation.

Assume $$C=uv$$. Then, we can write by the Chain Rule of Differentiation:

$$
\frac{dC(s)}{ds} = u\frac{dv}{ds} + v\frac{du}{ds}
$$

Let's get equation $$\eqref{eq:transmittance}$$ into the canonical form $$Y'(x) + P(x)Y = Q(x)$$.

$$
\begin{equation}
\frac{dC(s)}{ds} + \sigma(s)C(s) = \sigma(s)c(s)
\label{eq:transmittance-canonical}
\end{equation}
$$

The integrating factor is thus:

$$
I(t) = e^{\int_0^s P(t) dt} \\
= e^{\int_0^s \sigma(t) dt}
$$

Multiplying both sides of $$\eqref{eq:transmittance-canonical}$$ by the integrating factor, we get:

$$
\frac{dC(s)}{ds}.e^{\int_0^s \sigma(t) dt} + \sigma(s)C(s).e^{\int_0^s \sigma(t) dt} = \sigma(s)c(s).e^{\int_0^s \sigma(t) dt}
$$

We note that for the left side:

$$
\frac{d[C(s).e^{\int_0^s \sigma(t) dt}]}{ds} = \frac{dC(s)}{ds}.e^{\int_0^s \sigma(t) dt} + \sigma(s)C(s).e^{\int_0^s \sigma(t) dt}
$$

So we can essentially write:

$$
\frac{d[C(s).e^{\int_0^s \sigma(t) dt}]}{ds} = \sigma(s)c(s).e^{\int_0^s \sigma(t) dt} \\
\int_0^D \frac{d[C(s).e^{\int_0^s \sigma(t) dt}]}{ds} ds = \int_0^D \sigma(s)c(s).e^{\int_0^s \sigma(t) dt} ds \\
$$

Let's discount the constant factor of integration for the moment, and assume that $$C(0) = 0$$, we then have:

$$
C(D).e^{\int_0^D \sigma(t) dt} = \int_0^D \sigma(s)c(s).e^{\int_0^s \sigma(t) dt} ds \\
C(D) = \int_0^D \sigma(s)c(s).e^{\int_0^s \sigma(t) dt}.e^{-\int_0^D \sigma(t) dt} ds \\
C(D) = \int_0^D \sigma(s)c(s).e^{-\int_s^D \sigma(t) dt} ds \\
C(D) = \int_0^D \sigma(s)c(s).T'(s) ds
$$

Note that $$T'(s)$$ is the transmittance from $$s$$ to the viewer. Let's **flip the direction** around so that the origin is at the viewer's eye. Then the transmittance calculation changes from $$e^{-\int_0^{s'} \sigma'(t) dt}$$ ($$\sigma'(s)$$ is the opacity function as a function of $$s$$ from the viewer's eye). The integration is still from 0 to $$D$$, merely in the other direction, which does not change the result. Then we get the overall colour turns out to be:

$$
C(D) = \int_0^D \sigma'(s)c(s).e^{-\int_0^s \sigma'(t) dt} ds \\
T=e^{-\int_0^s \sigma'(t) dt}
$$

Let's drop the subscript of $$\sigma'$$, so that we get our final result.

$$
\begin{equation}
C(D) = \int_0^D \sigma(s)c(s).T(s) ds \\
\label{eq:volumetric-rendering-integration}
\end{equation}
T(s)=e^{-\int_0^D \sigma(t) dt}
$$

**Note:** I'm not sure if there is a specific mathematical procedure about how this change of coordinate is achieved. [Optical Models for Volumetric Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf) uses a coordinate system which originates from the back of the volume; the previous derivations are based on that. [Local and Global Illumination in the Volume Rendering Integral](https://drops.dagstuhl.de/opus/volltexte/2010/2709/pdf/18.pdf) uses a model which uses the viewing ray starting from the viewer's eye; that's the model that we will start with to derive the discretised version for algorithmic implementation. Hence, the remapping.

**Side Note**

A very nice thing happens if $$c(s)$$ happens to be constant. Let us assume it is $$c$$. Then the above equation becomes:

$$
C(D) = c \int_0^D \sigma(s).e^{-\int_s^D \sigma(t) dt} ds \\
$$

Note that from the **Fundamental Theorem of Calculus** (see [here](https://math.stackexchange.com/questions/1047523/derivative-of-integral-with-x-as-the-lower-limit)), we can write:

$$
\frac{d[e^{-\int_s^D \sigma(t) dt}]}{ds}=e^{-\int_s^D \sigma(t) dt}.\frac{d[-\int_s^D \sigma(t) dt]}{ds} \\
= - e^{-\int_s^D \sigma(t) dt}.\frac{d[\int_s^D \sigma(t) dt]}{ds} \\
= - e^{-\int_s^D \sigma(t) dt}(\sigma(D) \frac{dD}{ds} - \sigma(s) \frac{ds}{ds}) \\
= - e^{-\int_s^D \sigma(t) dt}(\sigma(D).0 - \sigma(s).1) \\
= - e^{-\int_s^D \sigma(t) dt}(- \sigma(s)) \\
= \sigma(s).e^{-\int_s^D \sigma(t) dt} \\
$$

Then, we can rewrite $$C(D)$$ as:

$$
C(D) = c \int_0^D \frac{d[e^{-\int_s^D \sigma(t) dt}]}{ds} ds \\
= c \left[ e^{-\int_D^D \sigma(t) dt} - e^{-\int_0^D \sigma(t) dt} \right] \\
= c \left[ e^0 - e^{-\int_0^D \sigma(t) dt} \right] \\
C(D) = c \left[ 1 - e^{-\int_0^D \sigma(t) dt} \right]
$$

The above technique will be used to derive the discrete implementation of the integration of the transmittances along a given ray, which is what we will be using in our code.

### The Discrete Volumetric Rendering Model

We still need to translate the mathematical form of the volumetric rendering model into a form suitable for implementation in code. To do this, we will have to discretise the integration procedure.

Let's assume two samples along a ray with their distances as $$s_i$$ and $$s_{i+1}$$. The quantities $$c(s)$$ and $$\sigma(s)$$ are constant within this segment, and are denoted as $$c_i$$ and $$\sigma_i$$ respectively. Let the distance between these two samples be $$d_i$$Let us substitute these limits in $$\eqref{eq:volumetric-rendering-integration}$$ to get:

$$
C(i) = \int_{s_i}^{s_{i+1}} \sigma_i c_i.\text{exp}\left[-\int_0^s \sigma(t) dt \right] ds \\
= \sigma_i c_i \int_{s_i}^{s_{i+1}} \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right].\text{exp}\left[-\int_{s_i}^{s} \sigma(t) dt \right] ds \\
= \sigma_i c_i \int_{s_i}^{s_{i+1}} \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right].\text{exp}\left[-\int_{s_i}^{s} \sigma_i dt \right] ds \\
= \sigma_i c_i \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right] \int_{s_i}^{s_{i+1}} \text{exp}\left[-\sigma_i (s-s_i) \right] ds \\
= \sigma_i c_i \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right] \left(- \frac{1}{\sigma_i} \right) \left[ \text{exp}[-\sigma_i (s_{i+1}-s_i)] - [\text{exp}[-\sigma_i (s_i-s_i) ] \right] \\
= \sigma_i c_i \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right] \left(- \frac{1}{\sigma_i}\right) [\text{exp}\left[-\sigma_i (s_{i+1}-s_i) \right] - e^0] \\
= \sigma_i c_i \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right] \left(- \frac{1}{\sigma_i}\right) [\text{exp}\left[-\sigma_i (s_{i+1}-s_i) \right] - 1] \\
C(i) = c_i \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right] \left[1 - \text{exp}(-\sigma_i d_i)\right]
$$

If we denote the transparency as $$T(s) = \text{exp}\left[-\int_0^{s_i} \sigma(t) dt \right]$$, we can write the above:

$$
C(i) = c_i T(s_i) \left[1 - e^{-\sigma_i d_i}\right]
$$

Summing the above across all pairs of consecutive samples, gives us the volumetric raycasting equations $$\eqref{eq:accumulated-transmittance}$$ and $$\eqref{eq:sample-transmittance}$$ we will use in our actual implementation.

$$
\begin{equation}
\boxed{
C_i = \sum_{i=1}^N T_i \left[1 - e^{-\sigma_i d_i}\right] c_i \\
}
\label{eq:accumulated-transmittance}
\end{equation}
$$

$$
\begin{equation}
\boxed{
T_i = \text{exp}\left[-\sum_{i=1}^{i-1} \sigma_i d_i \right]
}
\label{eq:sample-transmittance}
\end{equation}
$$

**Notes**

- The $$i-1$$ term in the summation is so that the last pairwise distance computed is between the $$(N-1)$$th and the $$N$$the sample, since there is no $$(N+1)$$th sample.
- The enumeration of samples starts front to back, closer to the camera and then proceeding towards the volume.

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering.py' %}
```

![Volumetric Sheath Cube](/assets/images/volumetric-sheath-cube-01.png)
![Volumetric Rendering Cube without Trilinear Interpolation](/assets/images/volumetric-rendering-cube-without-trilinear-interpolation.png)

Well, that looks somewhat like a cube from the viewpoint that we chose, but it has very weird edges, as well as some very visible artifacts. This is because of the discretisation of the space and our current handling of mapping a real coordinate to voxel intensity. What we do is convert every $$(x,y,z)$$ to $$(\text{int}(x),\text{int}(y),\text{int}(z))$$, and this can assign non-zero intensities to voxels which should be empty, and vice versa.

This can be fixed by **interpolating intensities of all neighbouring voxels** and using the interpolated value as the voxel intensity.

### Incorporating Trilinear Interpolation

In general, interpolation assigns a smoothly varying value to an intermediate point lying between two points with known values. This variation is linear when we speak of linear interpolation. Assume two points $$x_1,x_2 \in \mathbb{R}$$, with $$x_1 \leq x_2$$, having intensities $$c_1$$ and $$c_2$$ respectively. Then the colour $$c$$ of any point $$x_1 \leq x \leq x_2$$ can be made to vary smoothly by writing:

$$
c(x) = c_1 + \frac{x - x_1}{x_2 - x_1} (c_2 - c_1) \\
c(x) = c_1\left(1 - \frac{x - x_1}{x_2 - x_1}\right) + \frac{x - x_1}{x_2 - x_1} c_2 \\
c(x) = c_1\left(1 - \frac{x - x_1}{x_2 - x_1}\right) + \frac{x - x_1}{x_2 - x_1} c_2 \\
c(x) = c_1(1-t) + c_2 t
$$

Bilinear and trilinear interpolation extends this concept to two and three dimensions. Let's assume a point $$C(x,y,z)$$. Obviously it lies in a particular voxel. We want to calculate the intensity of this voxel by interpolating the intensities of the voxels bordering its eight corners.

Let $$x_0$$ be the nearest voxel ordinate to $$x$$ such that $$x_0<x$$.
Let $$x_1$$ be the nearest voxel ordinate to $$x$$ such that $$x<x_1x$$.

Similarly for $$y_0$$, $$y_1$$, $$z_0$$, and $$z_1$$.
Then, $$C_{000}$$ is the intensity of $$(x_0,y_0,z_0)$$. Then, $$C_{110}$$ is the intensity of $$$(x_1,y_1,z_0)$, and so on.

This convention is shown below.

![Trilinear Interpolation](/assets/images/trilinear-interpolation.png)


We use the following calculations to find the interpolated intensity of $$(x,y,z)$$, using the following scheme.

![Trilinear Interpolation Calculation Steps](/assets/images/trilinear-interpolation-calculation.png)

Let us define the factors $$t_x$$, $$t_y$$, an $$t_z$$ as follows:

$$
t_x = \frac{x-x_0}{x_1-x_0} \\
t_y = \frac{y-y_0}{y_1-y_0} \\
t_z = \frac{z-z_0}{z_1-z_0} \\
$$

Then, following calculation scheme of the diagram above, we get the following:

$$
C_{00}=C_{000} (1-t_x) + C_{100} t_x \\
C_{10}=C_{010} (1-t_x) + C_{110} t_x \\
C_{01}=C_{001} (1-t_x) + C_{101} t_x \\
C_{11}=C_{011} (1-t_x) + C_{111} t_x \\
C_{0}=C_{00} (1-t_y) + C_{10} t_y \\
C_{1}=C_{01} (1-t_y) + C_{11} t_y \\
\boxed{
C=C_{0} (1-t_z) + C_{1} t_z
}
$$

The following is the same code; instead of simply casting the coordinate to integers and addressing the voxel grid and getting the intensity of just that voxel, we now interpolate the intensity of the eight neighbouring voxels using the calculations shown above.

```python
{% include_absolute '/code/pytorch-learn/plenoxels/volumetric-rendering-with-trilinear-interpolation.py' %}
```

Here is the result.

![Volumetric Rendering with Trilinear Interpolation with Artifacts](/assets/images/volumteric-rendering-cube-trilinear-interpolation-with-artifacts.png)

As you can, see, it clearly looks like a proper cube, with none of the lumpy borders we saw in the previous code example. However, there is still some banding that is evident. Our next code sample fixes that.

### Fixing Volumetric Rendering Artifacts

[Common Artifacts in Volume Rendering](https://arxiv.org/abs/2109.13704) calls the kind of artifacts we see "onion rings". The solution they propose is to increase the sampling rate, which we change from 100 to 200.

This is the result.

![Volumetric Rendering with Trilinear Interpolation](/assets/images/volumetric-rendering-cube-trilinear-interpolation.png)

As you can see, the rings are more or less gone.

This concludes the second part of implementing the Plenoxels paper. We will tackle integrating the spherical harmonics into our model in the sequel.

### References

- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [Plenoxels Explained](https://deeprender.ai/blog/plenoxels-radiance-fields-without-neural-networks)
- [Direct Volume Rendering](https://www.youtube.com/watch?v=hiaHlTLN9TE)
- [Optical Models for Volumetric Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
- [Local and Global Illumination in the Volume Rendering Integral](https://drops.dagstuhl.de/opus/volltexte/2010/2709/pdf/18.pdf)
- [Common Artifacts in Volume Rendering](https://arxiv.org/abs/2109.13704)
- [Trilinear Interpolation](https://en.wikipedia.org/wiki/Trilinear_interpolation)
