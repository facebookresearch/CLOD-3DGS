:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the license found in the
:: LICENSE file in the root directory of this source tree.

:: Create comparison between 3DGS and MCMC
::
:: run_dist_lod_ablations.bat <method_dir> <input_dir>

cd ../build/Debug
nvidia-smi --lock-gpu-clocks=3105
nvidia-smi --lock-memory-clocks=10501


:: Distance LOD
python ../../scripts/optimization/test_distance_lod.py --input %1/drjohnson/ckpts/ckpt_59999_rank0.ply --data_dir %1/drjohnson/ --image_dir %2/tandt_db/db/drjohnson/images --config ../../config/full.yaml
python ../../scripts/optimization/test_distance_lod.py --input %1/playroom/ckpts/ckpt_59999_rank0.ply --data_dir %1/playroom/ --image_dir %2/tandt_db/db/playroom/images --config ../../config/full.yaml


:: Plot distance LOD
python ../../scripts/paper/plot_dist_lod_ablation.py --result_dir ../../results/dist_lod/


:: Render distance LOD images
python ../../scripts/optimization/render_single_view.py --config_gt ../../config/full.yaml --config_pred ../../config/full.yaml --data_dir %1/drjohnson/ --input %1/drjohnson/ckpts/ckpt_59999_rank0.ply --output_dir ../../results/dist_lod_renders/dist_lod --image_name IMG_6349.jpg --split val --params ../../scripts/paper/params/dist_lod_low.yaml ../../scripts/paper/params/dist_lod_med.yaml
python ../../scripts/optimization/render_single_view.py --config_gt ../../config/full.yaml --config_pred ../../config/full.yaml --data_dir %1/drjohnson/ --input %1/drjohnson/ckpts/ckpt_59999_rank0.ply --output_dir ../../results/dist_lod_renders/dist_lod --image_name IMG_6349.jpg --split val --params ../../scripts/paper/params/dist_lod_low.yaml ../../scripts/paper/params/dist_lod_med.yaml --overdraw


nvidia-smi --reset-gpu-clocks
nvidia-smi --reset-memory-clocks
cd ../../scripts