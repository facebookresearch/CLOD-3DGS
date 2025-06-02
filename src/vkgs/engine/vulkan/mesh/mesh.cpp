// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/mesh/mesh.h"

namespace vkgs {
namespace vk {

Mesh::Mesh(Context& context) {}
  
/**
 * @brief Draw mesh
 * @param cb command buffer
 */
void Mesh::Draw(VkCommandBuffer& cb) {
  std::vector<VkBuffer> vbs = {position_buffer_, color_buffer_};
  std::vector<VkDeviceSize> vb_offsets = {0, 0};
  vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(), vb_offsets.data());
  vkCmdBindIndexBuffer(cb, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cb, index_count_, 1, 0, 0, 0);
}

/**
 * @brief Upload data to the GPU
 * @param cb command buffer
 */
void Mesh::Upload(VkCommandBuffer& cb) {
  position_buffer_.FromCpu(cb, position_);
  color_buffer_.FromCpu(cb, color_);
  index_buffer_.FromCpu(cb, index_);
  index_count_ = index_.size();
}

}  // namespace vk
}  // namespace vkgs

