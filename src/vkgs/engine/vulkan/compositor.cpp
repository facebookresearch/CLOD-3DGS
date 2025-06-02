// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/compositor.h"

namespace vkgs {
namespace vk {

Compositor::Compositor() {};

void Compositor::initialize(VkCommandBuffer cb, Context context,
                            VkRenderPass render_pass, uint32_t num_levels,
                            uint32_t num_views, bool debug) {
  this->createVertexBuffer(cb, context);
  this->createSampler(cb, context);
  this->createDescriptorSetLayout(cb, context, num_levels);
  this->createDescriptorSets(cb, context, num_views);
  this->createGraphicsPipeline(context, render_pass, num_levels, debug);
}

Compositor::~Compositor(){};

void Compositor::createVertexBuffer(VkCommandBuffer cb, Context context_) {
  std::vector<float> vertices(20);
  vertices[0] = -1;
  vertices[1] = -1;
  vertices[2] = 0;
  vertices[3] = 0;
  vertices[4] = 0;

  vertices[5] = -1;
  vertices[6] = 1;
  vertices[7] = 0;
  vertices[8] = 0;
  vertices[9] = 1;

  vertices[10] = 1;
  vertices[11] = -1;
  vertices[12] = 0;
  vertices[13] = 1;
  vertices[14] = 0;

  vertices[15] = 1;
  vertices[16] = 1;
  vertices[17] = 0;
  vertices[18] = 1;
  vertices[19] = 1;

  std::vector<uint32_t> indices(6);
  indices[0] = 0;
  indices[1] = 1;
  indices[2] = 2;

  indices[3] = 3;
  indices[4] = 2;
  indices[5] = 1;

  this->vertex_buffer = Buffer(
    context_, 20 * sizeof(float),
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "compositor_vertex_buffer"
  );

  this->index_buffer = Buffer(
      context_, 6 * sizeof(uint32_t),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      "compositor_index_buffer"
  );

  this->vertex_buffer.FromCpu(cb, vertices);
  this->index_buffer.FromCpu(cb, indices);
};

void Compositor::createSampler(VkCommandBuffer cb, Context context_) {
  VkSamplerCreateInfo sampler_info;
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.pNext = VK_NULL_HANDLE;
  sampler_info.flags = 0;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sampler_info.mipLodBias = 0.0;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.maxAnisotropy = 1.0;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.minLod = 0;
  sampler_info.maxLod = 0;
  sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;

  vkCreateSampler(context_.device(), &sampler_info, nullptr, &sampler_);
}

void Compositor::createDescriptorSetLayout(VkCommandBuffer cb, Context context_, int num_levels){
  // create descriptor layout
  DescriptorLayoutCreateInfo descriptor_layout_info;
  descriptor_layout_info.bindings.resize(2);
  descriptor_layout_info.bindings[0] = {};
  descriptor_layout_info.bindings[0].binding = 0;
  descriptor_layout_info.bindings[0].descriptor_type = VK_DESCRIPTOR_TYPE_SAMPLER;
  descriptor_layout_info.bindings[0].descriptor_count = 1;
  descriptor_layout_info.bindings[0].stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT;

  descriptor_layout_info.bindings[1] = {};
  descriptor_layout_info.bindings[1].binding = 1;
  descriptor_layout_info.bindings[1].descriptor_type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  descriptor_layout_info.bindings[1].descriptor_count = num_levels;
  descriptor_layout_info.bindings[1].stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT;

  descriptor_set_layout_ = DescriptorLayout(context_, descriptor_layout_info);
}

void Compositor::createDescriptorSets(VkCommandBuffer cb, Context context_, uint32_t num_views) {
  descriptor_sets_.resize(num_views);
  for (uint32_t view = 0; view < num_views; view++) {
    descriptor_sets_[view] =
        std::make_unique<Descriptor>(context_, descriptor_set_layout_);
  }
}

void Compositor::createGraphicsPipeline(Context context_, VkRenderPass render_pass, int num_levels, bool debug){
  // create pipeline layout
  vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
  pipeline_layout_info.layouts = {descriptor_set_layout_};
  
  pipeline_layout_info.push_constants.resize(1);
  pipeline_layout_info.push_constants[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pipeline_layout_info.push_constants[0].offset = 0;
  pipeline_layout_info.push_constants[0].size = sizeof(CompositorPushConstantData);
  
  pipeline_layout_ = std::make_unique<PipelineLayout>(context_, pipeline_layout_info);
  
  std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(1);
  color_blend_attachments[0] = {};
  color_blend_attachments[0].blendEnable = VK_TRUE;
  color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
  color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
  color_blend_attachments[0].colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  std::vector<VkVertexInputBindingDescription> input_bindings(1);
  // xyz
  input_bindings[0].binding = 0;
  input_bindings[0].stride = sizeof(float) * 5;
  input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::vector<VkVertexInputAttributeDescription> input_attributes(2);
  // xyz
  input_attributes[0].location = 0;
  input_attributes[0].binding = 0;
  input_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  input_attributes[0].offset = 0;

  // uv
  input_attributes[1].location = 1;
  input_attributes[1].binding = 0;
  input_attributes[1].format = VK_FORMAT_R32G32_SFLOAT;
  input_attributes[1].offset = sizeof(float) * 3;
  
  CompositorSpecializationData specialization_data;
  specialization_data.num_levels = num_levels;
  specialization_data.debug = debug;

  std::vector<VkSpecializationMapEntry> specialization_map_entries(2);
  specialization_map_entries[0].constantID = 0;
  specialization_map_entries[0].size = sizeof(specialization_data.num_levels);
  specialization_map_entries[0].offset = offsetof(CompositorSpecializationData, num_levels);

  specialization_map_entries[1].constantID = 1;
  specialization_map_entries[1].size = sizeof(specialization_data.debug);
  specialization_map_entries[1].offset = offsetof(CompositorSpecializationData, debug);

  VkSpecializationInfo specialization_info;
  specialization_info.mapEntryCount = (uint32_t)specialization_map_entries.size();
  specialization_info.pMapEntries = &specialization_map_entries[0];
  specialization_info.dataSize = sizeof(specialization_data);
  specialization_info.pData = &specialization_data;
  
  vk::GraphicsPipelineCreateInfo pipeline_info = {};
  pipeline_info.layout = *pipeline_layout_;
  pipeline_info.vertex_shader = compositor_vert;
  pipeline_info.fragment_shader = compositor_frag;
  pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  pipeline_info.depth_test = false;
  pipeline_info.depth_write = false;
  pipeline_info.input_bindings = std::move(input_bindings);
  pipeline_info.input_attributes = std::move(input_attributes);
  pipeline_info.color_blend_attachments = std::move(color_blend_attachments);
  pipeline_info.render_pass = render_pass;
  pipeline_info.use_specialization = true;
  pipeline_info.specialization_info = specialization_info;

  pipeline_ = std::make_unique<GraphicsPipeline>(context_, pipeline_info);
};

void Compositor::updateDescriptor(std::vector<VkImageView>& image_views, VkImageLayout image_layout, uint32_t view_index)
{
  descriptor_sets_[view_index]->Update(0, sampler_, image_views, image_layout, 0, 0);
}

void Compositor::destroyResources(Context& context) {
  vkDestroySampler(context.device(), sampler_, nullptr);
}

Buffer& Compositor::getVertexBuffer() { return this->vertex_buffer; }
Buffer& Compositor::getIndexBuffer() { return this->index_buffer; }
Descriptor& Compositor::getDescriptorSet(uint32_t view_index) { return *descriptor_sets_[view_index]; };
GraphicsPipeline& Compositor::getGraphicsPipeline() { return *pipeline_; };
PipelineLayout& Compositor::getPipelineLayout() { return *pipeline_layout_; };

}  // namespace vk
}  // namespace vkgs