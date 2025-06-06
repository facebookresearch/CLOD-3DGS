// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_BUFFER_H
#define VKGS_ENGINE_VULKAN_BUFFER_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/debug.h"

namespace vkgs {
namespace vk {

/**
  * @brief Vulkan buffer
  */
class Buffer {
 public:
  Buffer();

  Buffer(Context context, VkDeviceSize size, VkBufferUsageFlags usage,
         std::string name = "");

  ~Buffer();

  operator bool() const noexcept { return impl_ != nullptr; }

  operator VkBuffer() const;

  VkDeviceSize size() const;

  void FromCpu(VkCommandBuffer command_buffer, const void* src,
               VkDeviceSize size);

  template <typename T>
  void FromCpu(VkCommandBuffer command_buffer, const std::vector<T>& v) {
    FromCpu(command_buffer, v.data(), v.size() * sizeof(T));
  }

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_BUFFER_H
