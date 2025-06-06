// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_SHADER_MODULE_H
#define VKGS_ENGINE_VULKAN_SHADER_MODULE_H

#include <string>

#include <vulkan/vulkan.h>

namespace vkgs {
namespace vk {

/**
 * @brief Shader source code data structure
 */
class ShaderSource {
 public:
  ShaderSource() = default;

  template <size_t N>
  ShaderSource(const uint32_t (&source)[N])
      : source_(source), size_(sizeof(uint32_t) * N) {}

  template <size_t N>
  ShaderSource& operator=(const uint32_t (&source)[N]) {
    source_ = source;
    size_ = sizeof(uint32_t) * N;
    return *this;
  }

  operator bool() const noexcept { return size_ != 0; }

  // byte size
  VkDeviceSize size() const noexcept { return size_; }
  const uint32_t* data() const noexcept { return source_; }

 private:
  VkDeviceSize size_ = 0;
  const uint32_t* source_ = nullptr;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_SHADER_MODULE_H
