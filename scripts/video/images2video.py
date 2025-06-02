# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import cv2


def images2video(input_dir):
    images = os.listdir(input_dir)
    images = [x for x in images if x.endswith(".png")]
    images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    image_data = cv2.imread(os.path.join(input_dir, images[0]))
    height, width, _ = image_data.shape

    # create video
    basename = os.path.basename(os.path.normpath(input_dir))
    video_filename = os.path.join(os.path.dirname(input_dir), basename + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video = cv2.VideoWriter(video_filename, fourcc, 60.0, (width, height))

    # save each frame
    for image in images:
        image_data = cv2.imread(os.path.join(input_dir, image))
        video.write(image_data)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()

    images2video(args.input_dir)
