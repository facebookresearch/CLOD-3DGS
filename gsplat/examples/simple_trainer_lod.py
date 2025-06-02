# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

# custom packages
import joblib
import subprocess

from scipy.spatial.transform import Rotation as R

from simple_trainer import Config, Runner
from tools.downscale_images import downscale_dir


def save_image_filenames_to_npy(cfg: Config, dataset: Dataset):
    indices = dataset.indices.tolist()
    split = dataset.split

    filename = os.path.join(cfg.result_dir, split + ".pkl")
    output = {}
    for index in indices:
        image_path = dataset.parser.image_paths[index]
        image_name = dataset.parser.image_names[index]

        cam2world = dataset.parser.camtoworlds[index]
        quat = R.from_matrix(cam2world[:3, :3]).as_quat()
        pos = cam2world[:3, 3]
        camera_id = dataset.parser.camera_ids[index]
        Ks = dataset.parser.Ks_dict[camera_id]
        image_size = dataset.parser.imsize_dict[camera_id]

        output[image_path] = {
            "filename": image_name,     # str
            "quat": quat,               # [4]
            "pos": pos,                 # [3]
            "Ks": Ks,                   # [3, 3]
            "image_size": image_size    # [2]
        }

    joblib.dump(output, filename)


@dataclass
class ConfigLOD(Config):
    # random start iter
    clod_random_start_iter: int = 30_000
    # random end iter
    clod_random_end_iter: int = 60_000

    # min_num_splats
    clod_min_num_splats: int = 100_000

    # enable CLOD training
    enable_clod: bool = True

    def adjust_steps(self, factor: float):
        super().adjust_steps(factor)
        self.clod_random_start_iter = int(self.clod_random_start_iter * factor)
        self.clod_random_end_iter = int(self.clod_random_end_iter * factor)


class RunnerLOD(Runner):
    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        super().__init__(local_rank, world_rank, world_size, cfg)
        if cfg.clod_min_num_splats is not None and hasattr(cfg, "cap_max"):
            self.clod_min_num_splats = cfg.clod_min_num_splats
        else:
            self.clod_min_num_splats = self.splats["means"].shape[0]

        self.mode = "train"

        save_image_filenames_to_npy(cfg, self.trainset)
        save_image_filenames_to_npy(cfg, self.valset)


    def train(self):
        self.mode = "train"
        return super().train()


    def eval(self, step: int):
        self.mode = "eval"
        output = super().eval(step)
        self.mode = "train"
        return output


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if self.mode == "train" and self.cfg.enable_clod and self.step >= self.cfg.clod_random_start_iter and self.step < self.cfg.clod_random_end_iter:
            num_splats = np.random.randint(self.clod_min_num_splats, self.splats["means"].shape[0]+1)
        else:
            num_splats = self.splats["means"].shape[0]

        means = self.splats["means"][:num_splats]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"][:num_splats]  # [N, 4]
        scales = torch.exp(self.splats["scales"])[:num_splats]  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])[:num_splats]  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"][:num_splats],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"][:num_splats]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"][:num_splats], self.splats["shN"][:num_splats]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            **kwargs,
        )
        return render_colors, render_alphas, info


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = RunnerLOD(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device, weights_only=True)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        # runner.render_traj(step=ckpt["step"])
        if cfg.compression is not None:
            runner.run_compression(step=ckpt["step"])
    else:
        runner.train()

    # convert pt to ply
    pt_files = os.listdir(os.path.join(cfg.result_dir, "ckpts"))
    pt_files = [x for x in pt_files if x.endswith(".pt")]
    for pt_file in pt_files:
        filename = os.path.join(cfg.result_dir, "ckpts", pt_file)
        subprocess.run(["python", "tools/convert_pt_to_ply.py", "--src", filename])

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            ConfigLOD(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            ConfigLOD(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
