// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_AXIS_MESH_H
#define VKGS_ENGINE_VULKAN_AXIS_MESH_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/mesh/mesh.h"

namespace vkgs {
namespace vk {

/**
 * @brief Axis mesh
 */
class AxisMesh : public Mesh {
 public:
  AxisMesh() = delete;
  AxisMesh(Context& context);
  ~AxisMesh() = default;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_AXIS_MESH_H
