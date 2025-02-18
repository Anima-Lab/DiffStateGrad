# Diffusion-State Guided Projected Gradient for Inverse Problems (ICLR 2025)

TODO (add image from paper):
![example](https://github.com/soominkwon/resample/blob/main/figures/resample_ex.png)

## Abstract

TODO (Edit this): In this work, we propose ReSample, an algorithm that can solve general inverse problems with pre-trained latent diffusion models. Our algorithm incorporates data consistency by solving an optimization problem during the reverse sampling process, a concept that we term as hard data consistency. Upon solving this optimization problem, we propose a novel resampling scheme to map the measurement-consistent sample back onto the noisy data manifold.

# DiffStateGrad-ReSample

This repository contains an implementation of ReSample enhanced with DiffStateGrad, combining two powerful approaches for solving inverse problems with pre-trained latent diffusion models.

## Overview

ReSample is an algorithm that can solve general inverse problems with pre-trained latent diffusion models. The algorithm incorporates data consistency by solving an optimization problem during the reverse sampling process, a concept termed as hard data consistency. Upon solving this optimization problem, ReSample employs a novel resampling scheme to map the measurement-consistent sample back onto the noisy data manifold.

In this work, we enhance ReSample by incorporating our Diffusion State-Guided Projected Gradient (DiffStateGrad) method, which addresses the fundamental challenge of maintaining the diffusion process on the data manifold while solving inverse problems.

## Our Contributions

We propose DiffStateGrad to improve upon ReSample's capabilities by:

1. Introducing a gradient-based measurement guidance approach that uses the measurement as guidance to move the intermediate diffusion state xt toward high-probability regions of the posterior.

2. Implementing a novel projection of the measurement guidance gradient onto a low-rank subspace that captures the data statistics of the learned prior.

3. Defining a projection step that preserves the measurement gradient on the tangent space of the state manifold, achieved through:
   - Performing singular value decomposition (SVD) on the diffusion state
   - Using the r highest contributing singular vectors as our projection matrix
   - Projecting the measurement gradient onto this subspace to remove directions orthogonal to the local manifold structure

## Implementation

This repository provides a modified version of the ReSample codebase that incorporates our DiffStateGrad method. The implementation maintains the core functionality of ReSample while adding our enhancements for improved performance and stability.

## Getting Started

### 1) Clone the repository

```
git clone https://github.com/rzirvi1665/DiffStateGrad.git

cd resample
```

<br />

### 2) Download pretrained checkpoints (autoencoders and model)

```
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```

<br />

### 3) Set environment

We use the external codes for motion-blurring and non-linear deblurring following the DPS codebase.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies via

```
conda env create -f environment.yaml
```

<br />

### 4) Inference

```
python3 diffstategrad_sample_condition.py
```

The code is currently configured to do inference on FFHQ. You can download the corresponding models from https://github.com/CompVis/latent-diffusion/tree/main and modify the checkpoint paths for other datasets and models.


<br />

## Task Configurations

```
# Linear inverse problems
- configs/tasks/super_resolution_config.yaml
- configs/tasks/gaussian_deblur_config.yaml
- configs/tasks/motion_deblur_config.yaml
- configs/tasks/box_inpainting_config.yaml
- configs/tasks/rand_inpainting_config.yaml

# Non-linear inverse problems
- configs/tasks/nonlinear_deblur_config.yaml
- configs/tasks/phase_retrieval_config.yaml
- configs/tasks/hdr_config.yaml
```

<br />

## Citation
If you find our work interesting, please consider citing

```
@inproceedings{
    zirvi2025diffusion,
    title={Diffusion State-Guided Projected Gradient for Inverse Problems},
    author={Rayhan Zirvi and Bahareh Tolooshams and Anima Anandkumar},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```

