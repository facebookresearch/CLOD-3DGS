# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Dict

import yaml

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


def load_data(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    import pdb; pdb.set_trace()
    return data