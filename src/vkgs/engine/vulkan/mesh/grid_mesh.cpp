// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/mesh/grid_mesh.h"

namespace vkgs {
namespace vk {

GridMesh::GridMesh(Context& context) : Mesh(context) {
  float scale = 10;
  constexpr int grid_size = 10;
  for (int i = 0; i < grid_size * 2 + 1; ++i) {
    index_.push_back(4 * i + 0);
    index_.push_back(4 * i + 1);
    index_.push_back(4 * i + 2);
    index_.push_back(4 * i + 3);
  }
  for (int i = -grid_size; i <= grid_size; ++i) {
    float t = static_cast<float>(i) / grid_size;
    position_.push_back(-1.f * scale);
    position_.push_back(0);
    position_.push_back(t * scale);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(1.f);

    position_.push_back(1.f * scale);
    position_.push_back(0);
    position_.push_back(t * scale);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(1.f);

    position_.push_back(t * scale);
    position_.push_back(0);
    position_.push_back(-1.f * scale);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(1.f);

    position_.push_back(t * scale);
    position_.push_back(0);
    position_.push_back(1.f * scale);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(0.5f);
    color_.push_back(1.f);
  }
  position_buffer_ = vk::Buffer(
      context, position_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      "buffer_grid_position_buffer");
  color_buffer_ = vk::Buffer(
      context, color_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      "buffer_grid_color_buffer");
  index_buffer_ = vk::Buffer(
      context, index_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      "buffer_grid_index_buffer");
}

}  // namespace vk
}  // namespace vkgs
