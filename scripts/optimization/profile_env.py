# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess


# Sets up profiling environment. For more detail, see:
# https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/
# Note: this should be run as administrator
class ProfileEnv:
    def __init__(self):
        self.clock_pairs = []


    def setup(self, mode=None):
        # query available clocks
        output = subprocess.run(
            args=["nvidia-smi",
                  "--query-supported-clocks=memory,graphics",
                  "--format=csv"],
            universal_newlines=True,
            stdout=subprocess.PIPE)

        clock_data = output.stdout.splitlines()
        for i in range(1, len(clock_data)):
            data = clock_data[i].split(", ")
            self.clock_pairs.append({
                "memory": data[0].split(" ")[0],
                "graphics": data[1].split(" ")[0]
            })

        index = 0
        if mode is not None:
            if mode == "highest":
                index = 0
            if mode == "lowest":
                index = len(self.clock_pairs) - 1

        # pick highest clock
        subprocess.run(
            args=["nvidia-smi",
                  "--lock-gpu-clocks=" + self.clock_pairs[index]["graphics"]])
        subprocess.run(
            args=["nvidia-smi",
                  "--lock-memory-clocks=" + self.clock_pairs[index]["memory"]])
        

    def shutdown(self):
        subprocess.run(args=["nvidia-smi", "--reset-gpu-clocks"])
        subprocess.run(args=["nvidia-smi", "--reset-memory-clocks"])