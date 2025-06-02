// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_STRUCTS_H
#define VKGS_ENGINE_VULKAN_STRUCTS_H

#include "vkgs/engine/vulkan/buffer.h"

namespace vkgs {
namespace vk {

/**
  * @brief Frame information
  */
struct FrameInfo {
  bool drew_splats = false;
  uint32_t total_point_count = 0;
  uint32_t loaded_point_count = 0;

  uint64_t rank_time = 0;
  uint64_t sort_time = 0;
  uint64_t inverse_time = 0;
  uint64_t projection_time = 0;
  uint64_t rendering_time = 0;
  uint64_t end_to_end_time = 0;

  uint64_t present_timestamp = 0;
  uint64_t present_done_timestamp = 0;

  vk::Buffer ply_buffer;
};



};  // namespace vk
};  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_STRUCTS_H