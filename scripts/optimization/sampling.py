# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def stratified_random(
    bounds,  # [N, 2]
    bins,    # [N]
):
    assert(len(bounds) == len(bins))

    output_list = []

    for i in range(len(bounds)):
        output = np.zeros((bins))

        low_bounds_norm = np.arange(bins[i]) / bins[i]
        high_bounds_norm = low_bounds_norm + (1.0 / bins[i])

        assert(bounds[i][1] > bounds[i][0])

        scale = bounds[i][1] - bounds[i][0]
        low_bounds = (low_bounds_norm * scale) + bounds[i][0]
        high_bounds = (high_bounds_norm * scale) + bounds[i][0]
        
        start_shape = [1] * len(bounds)
        start_shape[i] = bins[i]

        low_bounds = np.reshape(low_bounds, start_shape)
        high_bounds = np.reshape(high_bounds, start_shape)

        for j in range(len(bounds)):
            if i != j:
                low_bounds = np.repeat(low_bounds, repeats=bins[j], axis=j)
                high_bounds = np.repeat(high_bounds, repeats=bins[j], axis=j)

        output_list.append(np.random.uniform(low_bounds, high_bounds))

    output = np.stack(output_list, axis=-1)    
    return output


if __name__ == "__main__":
    bounds = [
        [-5.0, 5.0],
        [-2.0, 2.0],
        [1.0, 3.0],
    ]
    bins = [4, 2, 1]
    print(stratified_random(bounds, bins))
