// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_GRID_MESH_H
#define VKGS_ENGINE_VULKAN_GRID_MESH_H

#include "vkgs/engine/vulkan/mesh/mesh.h"

namespace vkgs {
namespace vk {

/**
 * @brief Grid mesh
 */
class GridMesh : public Mesh {
 public:
  GridMesh() = delete;
  GridMesh(Context& context);
  ~GridMesh() = default;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_GRID_MESH_H
