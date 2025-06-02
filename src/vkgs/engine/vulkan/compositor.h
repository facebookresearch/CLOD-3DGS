// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_COMPOSITOR_H
#define VKGS_ENGINE_VULKAN_COMPOSITOR_H

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/buffer.h"
#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/debug.h"
#include "vkgs/engine/vulkan/descriptor.h"
#include "vkgs/engine/vulkan/graphics_pipeline.h"
#include "vkgs/engine/vulkan/pipeline_layout.h"

#include "generated/compositor_vert.h"
#include "generated/compositor_frag.h"

namespace vkgs {
namespace vk {

/**
 * @brief Compositor specialization data
 */
struct CompositorSpecializationData {
  uint32_t num_levels;
  VkBool32 debug;
};


/**
 * @brief Compositor push constant data
 */
struct CompositorPushConstantData {
  float center[2];
  float levels[4][2];
  float res_scales[4];
  int full_res[2];
  int blending;
  int eye;
};


/**
 * @brief Compositor pass for compositing multiple foveation layers
 */
class Compositor {
 public:
  Compositor();
  ~Compositor();

  void initialize(VkCommandBuffer cb, Context context, VkRenderPass render_pass,
                  uint32_t num_levels, uint32_t num_views, bool debug = false);
  
  void createVertexBuffer(VkCommandBuffer cb, Context context_);
  void createSampler(VkCommandBuffer cb, Context context_);
  void createDescriptorSets(VkCommandBuffer cb, Context context_, uint32_t num_views);
  void createDescriptorSetLayout(VkCommandBuffer cb, Context context_, int num_levels);

  void render(VkCommandBuffer cb);

  VkImageView& getImageView();

  void createGraphicsPipeline(Context context_, VkRenderPass render_pass, int num_levels, bool debug=false);
  void updateDescriptor(std::vector<VkImageView>& image_views, VkImageLayout image_layout, uint32_t view_index);

  Buffer& getVertexBuffer();
  Buffer& getIndexBuffer();
  Descriptor& getDescriptorSet(uint32_t view_index);
  GraphicsPipeline& getGraphicsPipeline();
  PipelineLayout& getPipelineLayout();

  void destroyResources(Context& context_);

private:
  Context* context_ = nullptr;
	
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkImage image_ = VK_NULL_HANDLE;
  VkImageView imageview_ = VK_NULL_HANDLE;

  Buffer vertex_buffer;
  Buffer index_buffer;

  DescriptorLayout descriptor_set_layout_;

  std::unique_ptr<PipelineLayout> pipeline_layout_;
  std::unique_ptr<GraphicsPipeline> pipeline_;
  std::vector<std::unique_ptr<Descriptor>> descriptor_sets_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_COMPOSITOR_H
