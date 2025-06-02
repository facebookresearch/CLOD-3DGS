:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the license found in the
:: LICENSE file in the root directory of this source tree.

:: Run foveation parameter first
::
:: run_foveation_search.bat <method_dir>

cd ../build/Debug

python ../../scripts/optimization/test_foveation.py --config_gt ../../config/full.yaml --config_fov ../../config/fov_2.yaml --input %1/drjohnson/ckpts/ckpt_59999_rank0.ply --image_scale 1 --data_dir %1/drjohnson/ --time 0.5 --eval_every 8 --exp_name paper_0_5

python ../../scripts/optimization/test_foveation.py --config_gt ../../config/full.yaml --config_fov ../../config/fov_2.yaml --input %1/drjohnson/ckpts/ckpt_59999_rank0.ply --image_scale 2 --data_dir %1/drjohnson/ --time 1.5 --eval_every 8 --paper_1_5

cd ../../..