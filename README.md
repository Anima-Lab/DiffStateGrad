# Diffusion-State Guided Projected Gradient for Inverse Problems (ICLR 2025)

![example](https://github.com/rzirvi1665/DiffStateGrad/blob/main/figures/phase_example.png)

![example](https://github.com/rzirvi1665/DiffStateGrad/blob/main/figures/hdr_example.png)

## Abstract

In this work, we propose DiffStateGrad, a novel approach that enhances diffusion-based inverse problem solvers by projecting measurement guidance gradients onto a data-driven low-rank subspace defined by intermediate diffusion states. Our algorithm addresses the challenge of maintaining manifold consistency by performing singular value decomposition on intermediate diffusion states to define a projection matrix that captures local data statistics. This projection ensures that measurement guidance remains aligned with the learned data manifold while filtering out artifact-inducing components, leading to improved robustness and performance across various inverse problems. In this repository, we demonstrate the effectiveness of DiffStateGrad by applying it to ReSample's framework.

![example](https://github.com/rzirvi1665/DiffStateGrad/blob/main/figures/manifold_diffstategrad.png)

## Implementation

This repository provides a modified version of the ReSample codebase that incorporates our DiffStateGrad method. The implementation maintains the core functionality of ReSample while adding our enhancements for improved performance and stability.

Our main contributions can be found in `diffstategrad_sample_condition.py` and `ldm/models/diffusion/diffstategrad_ddim.py`. 

### DiffStateGrad Helper Methods

The core utilities of DiffStateGrad are implemented in `diffstategrad_utils.py`. These utilities can be applied to other works as a plug-and-play module. The implementation includes three main functions:

1. `compute_rank_for_explained_variance`: Determines the rank needed to explain a target variance percentage across channels
2. `compute_svd_and_adaptive_rank`: Performs SVD on diffusion state and computes adaptive rank based on variance cutoff
3. `apply_diffstategrad`: Computes the projected gradient using our DiffStateGrad algorithm

### Example Usage

```python
from diffstategrad_utils import compute_svd_and_adaptive_rank, apply_diffstategrad

# During optimization:
if iteration_count % period == 0:
    # Compute SVD and adaptive rank when needed
    U, s, Vh, adaptive_rank = compute_svd_and_adaptive_rank(z_t, var_cutoff=0.99)

# Apply DiffStateGrad to the normalized gradient
projected_grad = apply_diffstategrad(
    norm_grad=normalized_gradient,
    iteration_count=iteration_count,
    period=period,
    U=U, s=s, Vh=Vh, 
    adaptive_rank=adaptive_rank
)

# Update diffusion state with projected gradient
z_t = z_t - eta * projected_grad
```

For complete implementation details, please refer to [`diffstategrad_utils.py`](https://github.com/rzirvi1665/DiffStateGrad/blob/main/diffstategrad_utils.py) in our repository.

## Getting Started

### 1) Clone the repository

```
git clone https://github.com/rzirvi1665/DiffStateGrad.git

cd DiffStateGrad
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

## MIT License

All rights reserved unless otherwise stated by applicable licenses.
If this code includes third-party components, they remain under their original licenses and attributions.


