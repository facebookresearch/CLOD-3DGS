// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_MESH_H
#define VKGS_ENGINE_VULKAN_MESH_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/buffer.h"
#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {

/**
 * @brief General mesh class
 */
class Mesh {
 public:
  Mesh() = delete;
  Mesh(Context& context);
  ~Mesh() = default;

  void Upload(VkCommandBuffer& cb);
  void Draw(VkCommandBuffer& cb);

 protected:
  vk::Buffer position_buffer_;
  vk::Buffer color_buffer_;
  vk::Buffer index_buffer_;
  int index_count_;

  std::vector<float> position_;
  std::vector<float> color_;
  std::vector<uint32_t> index_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_MESH_H
