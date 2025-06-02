// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_FRUSTUM_MESH_H
#define VKGS_ENGINE_VULKAN_FRUSTUM_MESH_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/mesh/mesh.h"
#include "vkgs/engine/view_frustum_dataset.h"
#include "foveation/math.h"

namespace vkgs {
namespace vk {

/**
 * @brief Frustum mesh (for displaying view frustums of camera)
 */
class FrustumMesh : public Mesh {
 public:
  FrustumMesh() = delete;
  FrustumMesh(Context& context) = delete;
  FrustumMesh(Context& context, ViewFrustum& view_frustum);
  ~FrustumMesh() = default;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_FRUSTUM_MESH_H
