# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pathlib
import shutil

from PIL import Image


def images_in_dir(src):
    extensions = [".bmp", ".jpg", ".jpeg", ".png"]
    files = [x for x in os.listdir(src) if pathlib.Path(x).suffix in extensions]
    return files


def not_images_in_dir(src):
    extensions = [".bmp", ".jpg", ".jpeg", ".png"]
    files = [x for x in os.listdir(src) if pathlib.Path(x).suffix not in extensions]
    return files


def downscale_dir(src, scale):
    if scale == 1:
        return

    src_dir = os.path.join(src, "images")
    dst_dir = os.path.join(src, "images_" + str(scale))
    os.makedirs(dst_dir, exist_ok=True)

    # if downscaling has already been done
    if len(os.listdir(src_dir)) == len(os.listdir(dst_dir)):
        return

    image_filenames = images_in_dir(src_dir)
    for filename in image_filenames:
        src_filename = os.path.join(src_dir, filename)
        dst_filename = os.path.join(dst_dir, filename)

        src_image = Image.open(src_filename)
        src_image = src_image.resize((src_image.width // scale, src_image.height // scale), Image.BICUBIC)

        src_image.save(dst_filename)

    not_image_filenames = not_images_in_dir(src_dir)
    for filename in not_image_filenames:
        src_filename = os.path.join(src_dir, filename)
        dst_filename = os.path.join(dst_dir, filename)

        shutil.copyfile(src_filename, dst_filename)
