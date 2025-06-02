# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import struct
from typing import Dict

import numpy as np


def decode_next_ascii_line(bin_data, index):
    buffer = []
    for new_index in range(index, len(bin_data)):
        if bin_data[new_index] == int.from_bytes(("\n").encode("utf8"), "big"):
            new_index += 1
            break

        buffer.append(bin_data[new_index])

    output_text = str(bytes(buffer), "utf-8")
    return output_text, new_index


def decode_header(bin_data):
    decoded_line = ""

    output = {}

    index = 0
    while "end_header" not in decoded_line.strip():
        decoded_line, index = decode_next_ascii_line(bin_data, index)

        if decoded_line.startswith("format"):
            output["data_encoding_format"] = decoded_line.split(" ")[1]
        if decoded_line.startswith("property"):
            if "property" not in output:
                output["property"] = []
            output["property"].append(decoded_line.split(" ")[1:])
        elif decoded_line.startswith("element vertex"):
            output["num_points"] = int(decoded_line.split("element vertex")[1])

    return output, index


def decode_data(
    bin_data,
    header_info: Dict,
):
    num_points = header_info["num_points"]
    properties = header_info["property"]

    output = {}
    for prop in properties:
        if prop[0] == "float":
            output[prop[1]] = np.ndarray((num_points), np.float32)
        else:
            raise ValueError("Unknown Gaussian splat parameter type:", prop[0])

    type_stride = {
        "float": 4,
    }

    if header_info["data_encoding_format"] == "binary_little_endian":
        byte_order = "<"

    index = 0
    point_index = 0
    while index < len(bin_data):
        for prop in properties:
            prop_type = prop[0]
            prop_name = prop[1]
            stride = type_stride[prop_type]

            raw_data = bin_data[index:index+stride]
            if prop_type == "float":
                output[prop_name][point_index] = struct.unpack(byte_order+"f", raw_data)[0]
            index += stride
        point_index += 1

    return output


def read_ply(
    filename: str,
):
    with open(filename, "rb") as f:
        row_index = 0
        mode = None
        data_encoding_format = None
        bin_data = f.read()

        header, index = decode_header(bin_data)
        data = decode_data(bin_data[index:], header)

    return data


def save_ply(
    filename: str,
    data: Dict,
):
    with open(filename, "wb") as f:
        num_points = data["x"].shape[0]

        # save header
        f.write(("ply\n").encode("utf-8"))
        f.write(("format binary_little_endian 1.0\n").encode("utf-8"))
        f.write(("element vertex " + str(num_points) + "\n").encode("utf-8"))

        for key in data.keys():
            f.write(("property float " + key + "\n").encode("utf-8"))

        f.write(("end_header\n").encode("utf-8"))

        for p_i in range(num_points):
            for key in data.keys():
                f.write(struct.pack("<f", data[key][p_i]))
