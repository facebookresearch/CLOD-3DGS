// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/mesh/axis_mesh.h"

namespace vkgs {
namespace vk {

AxisMesh::AxisMesh(Context& context) : Mesh(context) {
  float scale = 10.0;
  position_ = {
      0.f, 0.f, 0.f, scale, 0.f,   0.f,    // x
      0.f, 0.f, 0.f, 0.f,   scale, 0.f,    // y
      0.f, 0.f, 0.f, 0.f,   0.f,   scale,  // z
  };
  color_ = {
      1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f,  // x
      0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,  // y
      0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,  // z
  };
  index_ = {
      0, 1, 2, 3, 4, 5,
  };
  position_buffer_ = vk::Buffer(
      context, position_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      "buffer_axis_position_buffer");
  color_buffer_ = vk::Buffer(
      context, color_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      "buffer_axis_color_buffer");
  index_buffer_ = vk::Buffer(
      context, index_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      "buffer_axis_index_buffer");
};

}  // namespace vk
}  // namespace vkgs

