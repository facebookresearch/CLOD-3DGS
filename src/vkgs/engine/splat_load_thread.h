// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_SPLAT_LOAD_THREAD_H
#define VKGS_ENGINE_SPLAT_LOAD_THREAD_H

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/buffer.h"

namespace vkgs {

/**
 * @brief Splat loader (from .ply file)
 */
class SplatLoadThread {
 public:
  /**
   * @brief Loading progress
   */
  struct Progress {
    uint32_t total_point_count = 0;
    uint32_t loaded_point_count = 0;

    vk::Buffer ply_buffer;

    // buffer barriers by load thread from previous to current progress() call.
    // this must be consumed by receiving thread.
    std::vector<VkBufferMemoryBarrier> buffer_barriers;
  };

 public:
  SplatLoadThread();

  SplatLoadThread(vk::Context context);

  ~SplatLoadThread();

  void Start(const std::string& ply_filepath,
             uint32_t max_splats = (std::numeric_limits<uint32_t>::max)());

  Progress GetProgress();

  void Cancel();

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_SPLAT_LOAD_THREAD_H
