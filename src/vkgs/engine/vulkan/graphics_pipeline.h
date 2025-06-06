// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_GRAPHICS_PIPELINE_H
#define VKGS_ENGINE_VULKAN_GRAPHICS_PIPELINE_H

#include <memory>
#include <string>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/pipeline_layout.h"
#include "vkgs/engine/vulkan/shader_module.h"

namespace vkgs {
namespace vk {

/**
 * @brief Graphics shader pipeline create info
 */
struct GraphicsPipelineCreateInfo {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  uint32_t subpass = 0;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  ShaderSource vertex_shader;
  ShaderSource geometry_shader;
  ShaderSource fragment_shader;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  std::vector<VkVertexInputBindingDescription> input_bindings;
  std::vector<VkVertexInputAttributeDescription> input_attributes;
  bool depth_test = false;
  bool depth_write = false;
  std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments;
  bool use_specialization = false;
  VkSpecializationInfo specialization_info;
};


/**
 * @brief Graphics shader pipeline
 */
class GraphicsPipeline {
 public:
  GraphicsPipeline();

  GraphicsPipeline(Context context,
                   const GraphicsPipelineCreateInfo& create_info);

  ~GraphicsPipeline();

  operator VkPipeline() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_GRAPHICS_PIPELINE_H
