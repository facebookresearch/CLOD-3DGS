# Paper Experiments
This file describes steps for reproducing results in the paper. The experiments were run on an RTX 4090 Founders Edition with an AMD Ryzen Threadripper PRO 3975WX with 128 GB of RAM. This exact setup is not needed to reproduce the results, but a 4090 should be used to compare against the timing data.

## Common Parameters
The scripts below share many of the same parameters. For ease of use, it is best to follow the following directory structures:

Model directories (trained models):
```
<models_dir>                            # <models_dir>
├── paper                               # <method_dir>: trained with what method (e.g., 3DGS, Octree-GS, paper)
    └── <data_dir>                      # <data_dir>: scene directory
        └── ckpts
            ├── ckpt_29999_rank0.ply    # <pretrained_ply_file>: pretrained model (e.g., 3DGS-MCMC)
            └── ckpt_59999_rank0.ply    # <clod_ply_file>: CLOD model
```
**Note:** We use `paper` as a `method_dir` to represent models that are used in the paper.

Input directories (original images/COLMAP data):
```
<input_dir>
├── 360_v2
│   ├── bicycle
│   ├── bonsai
│   ├── counter
│   ├── garden
│   ├── kitchen
│   ├── room
│   └── stump
├── 360_extra_scenes
│   ├── flowers
│   └── treehill
└── tandt_db
    ├── db
    │   ├── drjohnson
    │   └── playroom
    └── tandt
        ├── train
        └── truck
```

## Results

### Ablations

### CLOD (Table 1 and Figures 1 and 6)
Run the CLOD ablation with the following:
```
run_clod_results.bat <models_dir> <input_dir>
```

To generate the plots, run:
```
python ./paper/plot_lod_ablation.py --result_dir ../results/lod/
```

This will also generate a LaTeX table that will include the results. To generate the teaser image, run the following:
```
python ./paper/plot_teaser.py --scene playroom --image_name DSC05588.png --aspect 2.5 --result_dir ../results/lod/ --offset 0 -150
```

### Distance CLOD (Figures 8 and 9)
Run the distance LOD ablations with the following script:
```
run_dist_lod_ablations.bat <method_dir> <input_dir>
```

To generate the plots:
```
python ./paper/plot_dist_lod_ablation.py --result_dir ../results/dist_lod/
```

### 3DGS vs. 3DGS-MCMC Pretraining (Figure 7)
To run a comparison between 3DGS and 3DGS-MCMC files, run the following script:
```
run_3dgs_vs_mcmc.bat <data_dir> <mcmc_clod_ply_file> <3dgs_clod_ply_file>
```

### Foveation Parameter Search (Figure 5)
To run the foveation parameter search, run with the following settings:
```
run_foveation_search.bat <method_dir>
```

### SIBR Viewer Comparisons (Table 2)
While we do not provide the SIBR code in this repository, we describe how we created these benchmarks. We used the SIBR viewer to perform some of the comparisons in our work. Since many works provide an implementation using the SIBR viewer, we use this as a common base. However, we needed to modify the viewer to reproduce the validation images correctly. Framerate is computed differently than many techniques. This is because some approaches will compute framerate based on the Python implementation, which often includes costly per-frame PyTorch allocations. We added our own logic to the SIBR viewer directly, which should produce more accurate/fair framerates across different approaches.
