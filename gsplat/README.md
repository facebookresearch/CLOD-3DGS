# gsplat
This is a modified version of [gsplat](http://www.gsplat.studio/) v1.3.0 that supports CLOD training.

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features!

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## Installation
It is recommended to run this code in either Ubuntu or with WSL 2.

Create a conda environment to easily manage dependencies:
```bash
conda create -n gsplat python=3.10
conda activate gsplat

pip install -r examples/requirements.txt
```

## Evaluation
This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

Install libraries:
```bash
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## CLOD
This section describes how to train our CLOD models to reproduce the results in the paper:

### Download Models
To train the models, download the following datasets:
* [Mips-NeRF360](https://jonbarron.info/mipnerf360/): download both [Dataset Pt. 1](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and [Dataset Pt. 2](https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip)
* [Tanks&Temples](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/): download [Scenes - 650MB](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
* [Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/): download [Scenes - 650MB](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

The folder structure should look like the following:
```
~/Data
└── inputs
    ├── 360_v2
    │   ├── bicycle
    │   ├── garden
    │   ├── stump
    │   ├── room
    │   ├── counter
    │   ├── kitchen
    │   └── bonsai
    ├── 360_extra_scenes
    │   ├── flowers
    │   └── treehill
    └── tandt_db
        ├── tandt
        │   ├── truck
        │   └── train
        └── db
            ├── drjohnson
            └── playroom
```

### Train Models
To train all models, use the following command:
```bash
cd examples/
bash run_all.sh
```

It takes roughly 1 hour per scene on an RTX 4090. It is recommended to train using a GPU with at least 24GB VRAM.

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)


## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have made the mathematical supplement, with conventions and derivations, available [here](https://arxiv.org/abs/2312.02121). If you find this library useful in your projects or papers, please consider citing:

```
@misc{ye2023mathematical,
    title={Mathematical Supplement for the $\texttt{gsplat}$ Library},
    author={Vickie Ye and Angjoo Kanazawa},
    year={2023},
    eprint={2312.02121},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
