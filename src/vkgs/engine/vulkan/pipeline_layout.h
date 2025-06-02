// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H
#define VKGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/descriptor_layout.h"

namespace vkgs {
namespace vk {

/**
  * @brief Vulkan pipeline layout create info
  */
struct PipelineLayoutCreateInfo {
  std::vector<DescriptorLayout> layouts;
  std::vector<VkPushConstantRange> push_constants;
};

/**
 * @brief Vulkan pipeline layout
 */
class PipelineLayout {
 public:
  PipelineLayout();

  PipelineLayout(Context context, const PipelineLayoutCreateInfo& create_info);

  ~PipelineLayout();

  operator VkPipelineLayout() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_PIPELINE_LAYOUT_H
