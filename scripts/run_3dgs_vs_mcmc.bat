:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the license found in the
:: LICENSE file in the root directory of this source tree.

:: Create comparison between 3DGS and MCMC
::
:: run_3dgs_vs_mcmc.bat <data_dir> <mcmc_ply_file> <3dgs_ply_file>

cd ../build/Debug

python ../../scripts/optimization/render_single_view.py --config_gt ../../config/full.yaml --config_pred ../../config/full.yaml --output_dir ../../results/3dgs_vs_mcmc/mcmc --params ../../scripts/paper/params/lod_10.yaml --image_name IMG_6358.jpg --data_dir %1 --input %2

python ../../scripts/optimization/render_single_view.py --config_gt ../../config/full.yaml --config_pred ../../config/full.yaml --output_dir ../../results/3dgs_vs_mcmc/3dgs --params ../../scripts/paper/params/lod_10.yaml --image_name IMG_6358.jpg --data_dir %1 --input %3

cd ../../scripts