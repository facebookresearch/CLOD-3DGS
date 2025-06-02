// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <vkgs/engine/engine.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <map>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <vkgs/util/clock.h>

#include "vkgs/engine/vulkan/barrier.h"
#include "vkgs/engine/vulkan/descriptor.h"
#include "vkgs/engine/vulkan/descriptor_layout.h"
#include "vkgs/engine/vulkan/graphics_pipeline.h"
#include "vkgs/engine/vulkan/pipeline_layout.h"
#include "vkgs/engine/vulkan/utils_io.h"

#include "generated/parse_ply_comp.h"
#include "generated/projection_comp.h"
#include "generated/rank_comp.h"
#include "generated/inverse_index_comp.h"
#include "generated/splat_vert.h"
#include "generated/splat_frag.h"
#include "generated/color_vert.h"
#include "generated/color_frag.h"

namespace vkgs {
namespace {
  
bool terminate_vr_ = false;
void CloseVR(int) {
  terminate_vr_ = true;
}

struct SplatPushConstantData {
  glm::mat4 model;
  uint32_t time;
  uint32_t vis_mode;
  float vis_mode_scale;
};

struct RankPushConstantData {
  glm::mat4 model;
  glm::vec4 lod_params;  // [min LOD, max LOD, min distance, max distance]
};

struct SplatSpecializationConstantData {
  VkBool32 debug;
  VkBool32 dither;
};

}  // namespace
Engine::Engine(Config config, bool enable_validation)
    : benchmark_(config.num_levels()), enable_validation_(enable_validation) {
  config_ = config;

  // setup recorder
  benchmark_.num_frames(config_.num_frames_benchmark());
  recorder_.num_frames(config_.num_frames_recorder());

  if (glfwInit() == GLFW_FALSE)
    throw std::runtime_error("Failed to initialize glfw.");

  context_ = vk::Context(config.mode() == Mode::VR, enable_validation);

  if (config.mode() == Mode::VR) {
    num_views_ = 2;
  } else {
    num_views_ = 1;
  }

  center_.resize(num_views_);

  if (config.mode() == Mode::Desktop) {
    camera_ = std::make_shared<CameraLookAt>();
  } else {
    camera_ = std::make_shared<CameraGlobal>();
  }

  depth_format_ = VK_FORMAT_D32_SFLOAT;
  samples_ = VK_SAMPLE_COUNT_1_BIT;

  if (config_.color_mode() == std::string("unorm8")) {
    color_format_ = VK_FORMAT_B8G8R8A8_UNORM;
  } else if (config_.color_mode() == std::string("sfloat32")) {
    color_format_ = VK_FORMAT_R32G32B32A32_SFLOAT;
  } else {
    color_format_ = VK_FORMAT_R16G16B16A16_SFLOAT;
  }

  // initialize vectors
  framebuffer_compositor_.resize(num_views_);
  depth_compositor_attachment_.resize(num_views_);
  framebuffer_.resize(num_views_);
  color_attachment_.resize(num_views_);
  depth_attachment_.resize(num_views_);
  for (uint32_t view = 0; view < num_views_; view++) {
    framebuffer_compositor_[view].resize(num_frames_);
    depth_compositor_attachment_[view].resize(num_frames_);
    framebuffer_[view].resize(config_.num_levels());
    color_attachment_[view].resize(config_.num_levels());
    depth_attachment_[view].resize(config_.num_levels());
  }
  lod_levels_.resize(config_.num_levels(), 1.0);
  res_scales_.resize(config_.num_levels(), 1.0);
  last_lod_levels_.resize(config_.num_levels(), 1.0);
  //radii_scales_.resize(config_.num_levels(), 1.0);
  lod_params_.resize(config_.num_levels(), glm::vec4(config.lod_params()[0], config.lod_params()[1],
                config.lod_params()[2], config.lod_params()[3]));

  // render pass for splats
  render_passes_.resize(num_views_);
  for (uint32_t view = 0; view < num_views_; view++) {
    render_passes_[view].resize(config_.num_levels());
    for (uint32_t level = 0; level < config_.num_levels(); level++) {
      render_passes_[view][level] =
          vk::RenderPass(context_, samples_, color_format_, depth_format_, 1,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
  }

  // render pass for compositor
  if (config_.mode() == Mode::VR) {
    render_pass_compositor_ = vk::RenderPass(
        context_, samples_, VK_FORMAT_B8G8R8A8_UNORM, depth_format_, 1,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  } else {
    render_pass_compositor_ =
        vk::RenderPass(context_, samples_, VK_FORMAT_B8G8R8A8_UNORM,
                       depth_format_, 1, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  }

  {
    vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
    descriptor_layout_info.bindings.resize(1);
    descriptor_layout_info.bindings[0] = {};
    descriptor_layout_info.bindings[0].binding = 0;
    descriptor_layout_info.bindings[0].descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_layout_info.bindings[0].stage_flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    camera_descriptor_layout_ = vk::DescriptorLayout(context_, descriptor_layout_info);
  }

  {
    vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
    descriptor_layout_info.bindings.resize(5);
    descriptor_layout_info.bindings[0] = {};
    descriptor_layout_info.bindings[0].binding = 0;
    descriptor_layout_info.bindings[0].descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_layout_info.bindings[0].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[1] = {};
    descriptor_layout_info.bindings[1].binding = 1;
    descriptor_layout_info.bindings[1].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[1].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[2] = {};
    descriptor_layout_info.bindings[2].binding = 2;
    descriptor_layout_info.bindings[2].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[2].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[3] = {};
    descriptor_layout_info.bindings[3].binding = 3;
    descriptor_layout_info.bindings[3].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[3].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[4] = {};
    descriptor_layout_info.bindings[4].binding = 4;
    descriptor_layout_info.bindings[4].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[4].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    gaussian_descriptor_layout_ = vk::DescriptorLayout(context_, descriptor_layout_info);
  }

  {
    vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
    descriptor_layout_info.bindings.resize(6);
    descriptor_layout_info.bindings[0] = {};
    descriptor_layout_info.bindings[0].binding = 0;
    descriptor_layout_info.bindings[0].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[0].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[1] = {};
    descriptor_layout_info.bindings[1].binding = 1;
    descriptor_layout_info.bindings[1].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[1].stage_flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[2] = {};
    descriptor_layout_info.bindings[2].binding = 2;
    descriptor_layout_info.bindings[2].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[2].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[3] = {};
    descriptor_layout_info.bindings[3].binding = 3;
    descriptor_layout_info.bindings[3].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[3].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[4] = {};
    descriptor_layout_info.bindings[4].binding = 4;
    descriptor_layout_info.bindings[4].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[4].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptor_layout_info.bindings[5] = {};
    descriptor_layout_info.bindings[5].binding = 5;
    descriptor_layout_info.bindings[5].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[5].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    instance_layout_ = vk::DescriptorLayout(context_, descriptor_layout_info);
  }

  {
    vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
    descriptor_layout_info.bindings.resize(1);
    descriptor_layout_info.bindings[0] = {};
    descriptor_layout_info.bindings[0].binding = 0;
    descriptor_layout_info.bindings[0].descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_layout_info.bindings[0].stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;

    ply_descriptor_layout_ = vk::DescriptorLayout(context_, descriptor_layout_info);
  }

  // compute pipeline layout
  {
    vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.layouts = {
      camera_descriptor_layout_,
      gaussian_descriptor_layout_,
      instance_layout_,
      ply_descriptor_layout_,
    };

    pipeline_layout_info.push_constants.resize(1);
    pipeline_layout_info.push_constants[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_layout_info.push_constants[0].offset = 0;
    pipeline_layout_info.push_constants[0].size = sizeof(RankPushConstantData);

    compute_pipeline_layout_ = vk::PipelineLayout(context_, pipeline_layout_info);
  }

  // graphics pipeline layout
  {
    vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.layouts = {camera_descriptor_layout_, instance_layout_};

    pipeline_layout_info.push_constants.resize(1);
    pipeline_layout_info.push_constants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pipeline_layout_info.push_constants[0].offset = 0;
    pipeline_layout_info.push_constants[0].size = sizeof(SplatPushConstantData);

    graphics_pipeline_layout_ = vk::PipelineLayout(context_, pipeline_layout_info);
  }

  // parse ply pipeline
  {
    vk::ComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.layout = compute_pipeline_layout_;
    pipeline_info.source = parse_ply_comp;
    parse_ply_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
  }

  // rank pipeline
  {
    vk::ComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.layout = compute_pipeline_layout_;
    pipeline_info.source = rank_comp;
    rank_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
  }

  // inverse index pipeline
  {
    vk::ComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.layout = compute_pipeline_layout_;
    pipeline_info.source = inverse_index_comp;
    inverse_index_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
  }

  // projection pipeline
  {
    vk::ComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.layout = compute_pipeline_layout_;
    pipeline_info.source = projection_comp;      
    projection_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
  }

  // splat pipeline
  {
    uint32_t num_graphics_pipelines = 1;
    if (config_.debug()) {
      num_graphics_pipelines = VisModeCount;
    }

    for (uint32_t gp_index = 0; gp_index < num_graphics_pipelines; gp_index++) {
      std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(1);
      color_blend_attachments[0] = {};
      color_blend_attachments[0].blendEnable = VK_TRUE;
      color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      if (gp_index == (uint32_t)VisMode::OverdrawAlpha) {
        color_blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      }
      color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

      SplatSpecializationConstantData specialization_data = {};
      specialization_data.debug = config_.debug();
      specialization_data.dither = config_.dither();

      std::vector<VkSpecializationMapEntry> specialization_map_entries(2);
      specialization_map_entries[0] = {};
      specialization_map_entries[0].constantID = 0;
      specialization_map_entries[0].size =
          sizeof(SplatSpecializationConstantData::debug);
      specialization_map_entries[0].offset =
          offsetof(SplatSpecializationConstantData, debug);

      specialization_map_entries[1] = {};
      specialization_map_entries[1].constantID = 1;
      specialization_map_entries[1].size =
          sizeof(SplatSpecializationConstantData::dither);
      specialization_map_entries[1].offset =
          offsetof(SplatSpecializationConstantData, dither);

      VkSpecializationInfo specialization_info = {};
      specialization_info.mapEntryCount =
          (uint32_t)specialization_map_entries.size();
      specialization_info.pMapEntries = &specialization_map_entries[0];
      specialization_info.dataSize = sizeof(specialization_data);
      specialization_info.pData = &specialization_data;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = graphics_pipeline_layout_;
      pipeline_info.vertex_shader = splat_vert;
      pipeline_info.fragment_shader = splat_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      pipeline_info.depth_test = true;
      pipeline_info.depth_write = false;
      pipeline_info.color_blend_attachments =
          std::move(color_blend_attachments);
      pipeline_info.samples = samples_;
      pipeline_info.render_pass = render_passes_[0][0];
      pipeline_info.use_specialization = true;
      pipeline_info.specialization_info = specialization_info;
      splat_pipelines_.push_back(vk::GraphicsPipeline(context_, pipeline_info));

    }
  }

  // color pipeline
  {
    std::vector<VkVertexInputBindingDescription> input_bindings(2);
    // xyz
    input_bindings[0].binding = 0;
    input_bindings[0].stride = sizeof(float) * 3;
    input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // rgba
    input_bindings[1].binding = 1;
    input_bindings[1].stride = sizeof(float) * 4;
    input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> input_attributes(2);
    // xyz
    input_attributes[0].location = 0;
    input_attributes[0].binding = 0;
    input_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    input_attributes[0].offset = 0;

    // rgba
    input_attributes[1].location = 1;
    input_attributes[1].binding = 1;
    input_attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    input_attributes[1].offset = 0;

    std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(1);
    color_blend_attachments[0] = {};
    color_blend_attachments[0].blendEnable = VK_TRUE;
    color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    vk::GraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.layout = graphics_pipeline_layout_;
    pipeline_info.vertex_shader = color_vert;
    pipeline_info.fragment_shader = color_frag;
    pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    pipeline_info.input_bindings = std::move(input_bindings);
    pipeline_info.input_attributes = std::move(input_attributes);
    pipeline_info.depth_test = true;
    pipeline_info.depth_write = true;
    pipeline_info.color_blend_attachments = std::move(color_blend_attachments);
    pipeline_info.samples = samples_;
    pipeline_info.render_pass = render_passes_[0][0];
    color_line_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
  }

  // uniforms and descriptors
  camera_buffer_.resize(num_views_);
  descriptors_.resize(num_views_);
  splat_info_buffer_.resize(num_views_);
  splat_visible_point_count_.resize(num_views_);
  splat_draw_indirect_.resize(num_views_);
  visible_point_count_cpu_buffer_.resize(num_views_);

  for (int view = 0; view < num_views_; view++) {
    camera_buffer_[view].resize(config_.num_levels());
    descriptors_[view].resize(config_.num_levels());
    splat_info_buffer_[view].resize(config_.num_levels());
    splat_visible_point_count_[view].resize(config_.num_levels());
    splat_draw_indirect_[view].resize(config_.num_levels());
    visible_point_count_cpu_buffer_[view].resize(config_.num_levels());
    for (int level = 0; level < config_.num_levels(); level++) {
      camera_buffer_[view][level] =
          vk::UniformBuffer<vk::shader::Camera>(context_, num_frames_);
      visible_point_count_cpu_buffer_[view][level] =
          vk::CpuBuffer(context_, num_frames_ * sizeof(uint32_t));
      descriptors_[view][level].resize(num_frames_);
      for (int i = 0; i < num_frames_; ++i) {
        descriptors_[view][level][i].camera =
            vk::Descriptor(context_, camera_descriptor_layout_);
        descriptors_[view][level][i].camera.Update(
            0, camera_buffer_[view][level], camera_buffer_[view][level].offset(i),
            camera_buffer_[view][level].element_size());

        descriptors_[view][level][i].gaussian =
            vk::Descriptor(context_, gaussian_descriptor_layout_);
        descriptors_[view][level][i].splat_instance =
            vk::Descriptor(context_, instance_layout_);
        descriptors_[view][level][i].ply =
            vk::Descriptor(context_, ply_descriptor_layout_);
      }

      splat_info_buffer_[view][level] =
          vk::UniformBuffer<vk::shader::SplatInfo>(context_, num_frames_);
      splat_visible_point_count_[view][level] = vk::Buffer(
          context_, sizeof(uint32_t),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          "buffer_splat_visible_point_count_" + std::to_string(level));
      splat_draw_indirect_[view][level] =
          vk::Buffer(context_, 12 * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                     "buffer_splat_draw_indirect_" + std::to_string(level));
    }
  }

  // commands and synchronizations
  draw_command_buffers_.resize(num_cb_);
  VkCommandBufferAllocateInfo command_buffer_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_buffer_info.commandPool = context_.command_pool();
  command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_info.commandBufferCount = draw_command_buffers_.size();
  vkAllocateCommandBuffers(context_.device(), &command_buffer_info,
                           draw_command_buffers_.data());

  VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  render_finished_semaphores_.resize(num_frames_);
  render_finished_fences_.resize(num_frames_);
  for (int i = 0; i < num_frames_; ++i) {
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                      &render_finished_semaphores_[i]);
    vkCreateFence(context_.device(), &fence_info, NULL,
                  &render_finished_fences_[i]);
  }

  image_acquired_semaphores_.resize(num_cb_);
  for (int i = 0; i < num_cb_; ++i) {
    vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                      &image_acquired_semaphores_[i]);
  }

  {
    VkSemaphoreTypeCreateInfo semaphore_type_info = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    semaphore_type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &semaphore_type_info;

    vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &transfer_semaphore_);
  }

  // create query pools
  timestamp_query_pools_.resize(num_views_);
  for (uint32_t view = 0; view < num_views_; view++) {
    timestamp_query_pools_[view].resize(config_.num_levels());
    for (int level = 0; level < config_.num_levels(); level++) {
      timestamp_query_pools_[view][level].resize(num_frames_);
      for (int i = 0; i < num_frames_; ++i) {
        VkQueryPoolCreateInfo query_poolevelnfo = {
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_poolevelnfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_poolevelnfo.queryCount = timestamp_count_;
        vkCreateQueryPool(context_.device(), &query_poolevelnfo, NULL,
                          &timestamp_query_pools_[view][level][i]);
      }
    }
  }

  // preallocate splat storage
  frame_infos_.resize(num_views_);
  splat_storage_.resize(num_views_);
  sort_storage_.resize(num_views_);

  uint32_t max_splat_count = (std::min)(config_.max_splats(), MAX_SPLAT_COUNT);
  for (int view = 0; view < num_views_; view++) {
    frame_infos_[view].resize(config_.num_levels());
    splat_storage_[view].resize(config_.num_levels());
    sort_storage_[view].resize(config_.num_levels());

    for (int level = 0; level < config_.num_levels(); level++) {
      // frame info
      frame_infos_[view][level].resize(num_frames_);

      splat_storage_[view][level].position = vk::Buffer(
          context_, max_splat_count * 3 * sizeof(float),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          "buffer_splat_storage_position_" + std::to_string(level));
      splat_storage_[view][level].cov3d = vk::Buffer(
          context_, max_splat_count * 6 * sizeof(float),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          "buffer_splat_storage_cov3d_" + std::to_string(level));
      splat_storage_[view][level].opacity = vk::Buffer(
          context_, max_splat_count * sizeof(float),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          "buffer_splat_storage_opacity_" + std::to_string(level));
      splat_storage_[view][level].sh = vk::Buffer(
          context_, max_splat_count * 48 * sizeof(float),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          "buffer_splat_storage_sh_" + std::to_string(level));

      splat_storage_[view][level].key =
          vk::Buffer(context_, max_splat_count * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     "buffer_splat_storage_key_" + std::to_string(level));
      splat_storage_[view][level].index =
          vk::Buffer(context_, max_splat_count * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     "buffer_splat_storage_index_" + std::to_string(level));
      splat_storage_[view][level].inverse_index = vk::Buffer(
          context_, max_splat_count * sizeof(uint32_t),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          "buffer_splat_storage_inverse_index_" + std::to_string(level));

      splat_storage_[view][level].instance =
          vk::Buffer(context_, max_splat_count * 10 * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     "buffer_splat_storage_instance_" + std::to_string(level));
    }
  }

  {
    // create splat load thread
    splat_load_thread_.resize(num_views_);
    loaded_point_count_.resize(num_views_);
    loaded_point_.resize(num_views_);
    for (int view = 0; view < num_views_; view++) {
      if (config_.lod()) {
        for (int level = 0; level < config_.num_levels(); level++) {
          splat_load_thread_[view].push_back(SplatLoadThread(context_));
          loaded_point_count_[view].push_back(0);
        }
      } else {
          splat_load_thread_[view].push_back(SplatLoadThread(context_));
          loaded_point_count_[view].push_back(0);
      }
      for (int frame_i = 0; frame_i < num_frames_; frame_i++) {
        loaded_point_[view].push_back(false);
      }
    }
  }

  {
    // create sorter
    sorter_.resize(num_views_);
    for (int view = 0; view < num_views_; view++) {
      sorter_[view] = std::vector<VrdxSorter>(config_.num_levels());
      for (int level = 0; level < config_.num_levels(); level++) {
        VrdxSorterCreateInfo sorter_info = {};
        sorter_info.physicalDevice = context_.physical_device();
        sorter_info.device = context_.device();
        sorter_info.pipelineCache = context_.pipeline_cache();
        vrdxCreateSorter(&sorter_info, &sorter_[view][level]);

        // preallocate sorter storage
        VrdxSorterStorageRequirements requirements;
        vrdxGetSorterKeyValueStorageRequirements(
            sorter_[view][level], MAX_SPLAT_COUNT, &requirements);

        sort_storage_[view][level] =
            vk::Buffer(context_, requirements.size, requirements.usage);
      }
    }
  }

  PreparePrimitives();

  if (config_.mode() == Mode::Desktop) {
    gui_.prepare();
  }
}

Engine::~Engine() {
  splat_load_thread_ = {};

  vkDeviceWaitIdle(context_.device());

  compositor_.destroyResources(context_);
  for (int view = 0; view < num_views_; view++) {
    for (int level = 0; level < config_.num_levels(); level++) {
      vrdxDestroySorter(sorter_[view][level]);
    }
  }

  for (auto semaphore : image_acquired_semaphores_)
    vkDestroySemaphore(context_.device(), semaphore, NULL);
  for (auto semaphore : render_finished_semaphores_)
    vkDestroySemaphore(context_.device(), semaphore, NULL);
  for (auto fence : render_finished_fences_)
    vkDestroyFence(context_.device(), fence, NULL);
  vkDestroySemaphore(context_.device(), transfer_semaphore_, NULL);

  for (int view = 0; view < num_views_; view++) {
    for (int level = 0; level < config_.num_levels(); level++) {
      for (auto query_pool : timestamp_query_pools_[view][level]) {
        vkDestroyQueryPool(context_.device(), query_pool, NULL);
      }
    }
  }
    
  if (config_.mode() == Mode::Desktop) {
    gui_.destroy();
    glfwTerminate();
  }
}

void Engine::LoadSplats(const std::string& ply_filepath) {
  // check if file exists
  if (!std::filesystem::exists(ply_filepath)) {
    throw std::invalid_argument(("File not found: " + ply_filepath).c_str());
  }

  for (int view = 0; view < num_views_; view++) {
    for (int slt_i = 0; slt_i < splat_load_thread_[view].size(); slt_i++) {
      auto temp_filename = ply_filepath;
      if (config_.lod()) {
        temp_filename =
            ply_filepath.substr(0, ply_filepath.find_last_of("."));
        temp_filename = temp_filename + std::to_string(slt_i) + ".ply";
      }
      
      splat_load_thread_[view][slt_i].Cancel();
      splat_load_thread_[view][slt_i].Start(temp_filename, config_.max_splats());
    }
  }
  last_loaded_ply_filepath_ = ply_filepath;
  vk::utils_io::load_camera_parameters(
      vk::utils_io::get_camera_filename(ply_filepath), translation_,
      rotation_, scale_);
}


void Engine::LoadSplatsAsync(const std::string& ply_filepath) {
  std::unique_lock<std::mutex> guard{mutex_};
  pending_ply_filepath_ = ply_filepath;
  last_loaded_ply_filepath_ = ply_filepath;
  vk::utils_io::load_camera_parameters(
      vk::utils_io::get_camera_filename(ply_filepath), translation_,
      rotation_, scale_);
}


void Engine::Start() {
  // setup foveated layers
  fov_info_.num_levels = config_.num_levels();
  
  if (config_.time_budget() > 0.0f) {
    time_budget_ = config_.time_budget();
    lazy_budget_mode_ = true;
  }
  
  if (config_.debug()) {
    vis_mode_ = static_cast<int>(config_.vis_mode());
    vis_mode_scale_ = config_.vis_scale();
  }

  if (config_.mode() == Mode::VR) {
    swapchain_ =
        std::make_shared<vk::SwapchainVR>(context_, context_.xr_manager());
    width_ = swapchain_->width();
    height_ = swapchain_->height();
    fov_info_.res[0] = width_;
    fov_info_.res[1] = height_;
  } else {
    // create window
    fov_info_.res[0] = config_.res()[0];
    fov_info_.res[1] = config_.res()[1];
    width_ = fov_info_.res[0];
    height_ = fov_info_.res[1];

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);
    window_ = glfwCreateWindow(width_, height_, "vkgs", NULL, NULL);

    // create swapchain
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(context_.instance(), window_, NULL, &surface);

    swapchain_ =
        std::make_shared<vk::SwapchainDesktop>(context_, surface, num_frames_);
    if (config_.mode() == Mode::Desktop) {
      gui_.initialize(*this);
    }
  }
  fov_layers_.initialize(fov_info_, &config_.fov_res(), &config_.radii_levels());

  for (uint32_t view = 0; view < num_views_; view++) {
    for (uint32_t level = 0; level < config_.num_levels(); level++) {
      auto layer = fov_layers_.getLayers()[level];
      RecreateFramebuffer(view, level, layer);
    }
  }
  RecreateCompositorFramebuffer();

  terminate_ = false;
  if (config_.mode() == Mode::Desktop) {
    glfwShowWindow(window_);
  }
}

void Engine::End() {
  vkDeviceWaitIdle(context_.device());

  glfwDestroyWindow(window_);

  if (config_.mode() == Mode::Desktop) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }

  terminate_ = false;
}

SampleResult Engine::Sample(SampleParams& params, SampleState& state) {
  // save original values
  auto original_num_frames_benchmark = benchmark_.num_frames();
  auto original_num_frames_recorder = recorder_.num_frames();

  // set new values
  benchmark_.num_frames(params.num_frames_benchmark);
  recorder_.num_frames(params.num_frames_recorder);

  SampleResult output{};
  
  if (params.num_frames_benchmark < params.num_frames_recorder) {
    throw std::invalid_argument(
        "SampleParams.num_frames_benchmark cannot be less than SampleParams.num_frames_recorder");
  }
  
  if (state.pos.size() != params.num_frames_recorder && params.num_frames_recorder > 0) {
    throw std::invalid_argument(
        "SampleState.pos must have the same number of elements as "
        "SampleParams.num_frames");
  }

  if (state.quat.size() != params.num_frames_recorder && params.num_frames_recorder > 0) {
    throw std::invalid_argument(
        "SampleState.quat must have the same number of elements as "
        "SampleParams.num_frames");    
  }
  
  auto camera_global = std::dynamic_pointer_cast<CameraGlobal>(camera_);

  // get timing data
  camera_->SetWindowSize(width_, height_);

  // warmup
  for (uint32_t i = 0; i < params.num_frames_benchmark; i++) {
    camera_global->pos(state.pos[0]);
    camera_global->quat(state.quat[0]);
    camera_global->view_angles(state.view_angles[0]);
    for (uint32_t level = 0; level < config_.num_levels(); level++) {
      lod_levels_[level] = params.lod[0][level];
      res_scales_[level] = params.res[0][level];
      lod_params_[level] = params.lod_params[0][level];
    }
    for (uint32_t view_index = 0; view_index < num_views_; view_index++) {
      center_[view_index] = state.center[0];
    }
    Draw();
  }

  benchmark_.resetRecording();
  benchmark_mode_ = true;

  // record timing data
  while (!benchmark_.isRecordingDone()) {
    uint32_t benchmark_frame = benchmark_.frame();
    if (params.num_frames_recorder > 0) {
      benchmark_frame = benchmark_frame % params.num_frames_recorder;
    }
    camera_global->pos(state.pos[benchmark_frame]);
    camera_global->quat(state.quat[benchmark_frame]);
    camera_global->view_angles(state.view_angles[benchmark_frame]);
    for (uint32_t level = 0; level < config_.num_levels(); level++) {
      lod_levels_[level] = params.lod[benchmark_frame][level];
      res_scales_[level] = params.res[benchmark_frame][level];
      lod_params_[level] = params.lod_params[benchmark_frame][level];
    }
    for (uint32_t view_index = 0; view_index < num_views_; view_index++) {
      center_[view_index] = state.center[benchmark_frame];
    }
    Draw();
  }

  output.time = benchmark_.end_to_end_time();

  // record image data
  recorder_.resetSaving(width_, height_);
  recorder_mode_ = true;
  while (!recorder_.isRenderingDone()) {
    camera_global->pos(state.pos[recorder_.frame()]);
    camera_global->quat(state.quat[recorder_.frame()]);
    camera_global->view_angles(state.view_angles[recorder_.frame()]);
    for (uint32_t level = 0; level < config_.num_levels(); level++) {
      lod_levels_[level] = params.lod[recorder_.frame()][level];
      res_scales_[level] = params.res[recorder_.frame()][level];
      lod_params_[level] = params.lod_params[recorder_.frame()][level];
    }
    for (uint32_t view_index = 0; view_index < num_views_; view_index++) {
      center_[view_index] = state.center[recorder_.frame()];
    }
    Draw();
  }
  while (!recorder_.isSavingDone()) {}

  // export image to array
  if (params.num_frames_recorder > 0) {
    auto image_data = recorder_.image_data();
    uint32_t num_frames = image_data.size();
    uint32_t image_size = image_data[0].size();

    output.data.clear();
    output.data.resize(num_frames * image_size);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_frames; i++) {
      for (uint32_t j = 0; j < image_size; j++) {
        output.data[offset+j] = image_data[i][j];
      }
      offset += image_size;
    }
  }

  output.shape = {params.num_frames_recorder, height_, width_, 3};

  benchmark_.num_frames(original_num_frames_benchmark);
  recorder_.num_frames(original_num_frames_recorder);

  return output;
}

bool Engine::SplatsLoaded(uint32_t view_index) {
  bool all_buffers_full = true;
  for (int slt_i = 0; slt_i < loaded_point_count_[view_index].size(); slt_i++) {
    all_buffers_full &= (loaded_point_count_[view_index][slt_i] > 0);
  }
  return all_buffers_full;
}

glm::mat4 Engine::GetModelMatrix() {
  return vkgs::math::ToScaleMatrix4(scale_) *
         vkgs::math::ToTranslationMatrix4(translation_) *
         glm::toMat4(rotation_);
}

void Engine::Run() {
  Start();

  // main loop
  if (config_.mode() == Mode::VR) {
    std::signal(SIGINT, CloseVR);
    while (!terminate_vr_) {
      // load pending file from async request
      {
        std::unique_lock<std::mutex> guard{mutex_};
        if (!pending_ply_filepath_.empty()) {
          LoadSplats(pending_ply_filepath_);
          pending_ply_filepath_.clear();
        }
      }
      vk::XREventState event_state = context_.xr_manager()->poll_events();

      if (event_state.running_) {
        auto xr_swapchain =
            std::dynamic_pointer_cast<vk::SwapchainVR>(swapchain_)
                ->xr_swapchain();

        camera_->SetWindowSize(width_, height_);

        context_.updateEyeTracking();

        context_.xr_manager()->begin_frame();
        context_.xr_manager()->poll_actions(*this);
        Draw();
        context_.xr_manager()->end_frame(xr_swapchain, swapchain_->width(),
                                         swapchain_->height());
      }
    }
  } else {
    while (!glfwWindowShouldClose(window_) && !terminate_) {
      glfwPollEvents();

      // load pending file from async request
      {
        std::unique_lock<std::mutex> guard{mutex_};
        if (!pending_ply_filepath_.empty()) {
          LoadSplats(pending_ply_filepath_);
          pending_ply_filepath_.clear();
        }
      }

      int width, height;
      glfwGetFramebufferSize(window_, &width, &height);
      camera_->SetWindowSize(width, height);

      Draw();
    }
  }
  
  End();
}

void Engine::Close() { terminate_ = true; }

void Engine::PreparePrimitives() {
  std::vector<uint32_t> splat_index;
  uint32_t max_splats = (std::min)(config_.max_splats(), MAX_SPLAT_COUNT);
  splat_index.reserve(max_splats * 6);
  for (int i = 0; i < max_splats; ++i) {
    splat_index.push_back(4 * i + 0);
    splat_index.push_back(4 * i + 1);
    splat_index.push_back(4 * i + 2);
    splat_index.push_back(4 * i + 2);
    splat_index.push_back(4 * i + 1);
    splat_index.push_back(4 * i + 3);
  }

  splat_index_buffer_.resize(num_views_);
  for (int view = 0; view < num_views_; view++) {
    splat_index_buffer_[view].resize(config_.num_levels());
    for (int level = 0; level < config_.num_levels(); level++) {
      splat_index_buffer_[view][level] = vk::Buffer(
          context_, splat_index.size() * sizeof(float),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
          "buffer_splat_index_buffer_" + std::to_string(level));
    }
  }

  axis_ = std::make_shared<vk::AxisMesh>(context_);
  grid_ = std::make_shared<vk::GridMesh>(context_);
  if (config_.view_filename() != std::string("")) {
    view_frustum_dataset_ = std::make_shared<ViewFrustumDataset>(config_.view_filename());
    for (uint32_t i = 0; i < view_frustum_dataset_->size(); i++) {
      auto view_frustum = (*view_frustum_dataset_.get())[i];
      view_frustum_meshes_.push_back(
          std::make_shared<vk::FrustumMesh>(context_, view_frustum));
    }
  }

  VkCommandBufferAllocateInfo command_buffer_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_buffer_info.commandPool = context_.command_pool();
  command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_info.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(context_.device(), &command_buffer_info, &cb);

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  begin_info.pInheritanceInfo = nullptr;
  vkBeginCommandBuffer(cb, &begin_info);

  for (int view = 0; view < num_views_; view++) {
    for (int level = 0; level < config_.num_levels(); level++) {
      splat_index_buffer_[view][level].FromCpu(cb, splat_index);
    }
  }

  // Upload meshes to the GPU
  axis_->Upload(cb);
  grid_->Upload(cb);
  for (uint32_t i = 0; i < view_frustum_meshes_.size(); i++) {
    view_frustum_meshes_[i]->Upload(cb);
  }

  compositor_.initialize(cb, context_, render_pass_compositor_,
                       config_.num_levels(), num_views_, config_.debug());

  vkEndCommandBuffer(cb);

  uint64_t wait_value = transfer_timeline_;
  uint64_t signal_value = transfer_timeline_ + 1;
  VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info = {
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline_semaphore_submit_info.waitSemaphoreValueCount = 1;
  timeline_semaphore_submit_info.pWaitSemaphoreValues = &wait_value;
  timeline_semaphore_submit_info.signalSemaphoreValueCount = 1;
  timeline_semaphore_submit_info.pSignalSemaphoreValues = &signal_value;

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.pNext = &timeline_semaphore_submit_info;
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &transfer_semaphore_;
  submit_info.pWaitDstStageMask = &wait_stage;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &transfer_semaphore_;
  vkQueueSubmit(context_.graphics_queue(), 1, &submit_info, NULL);

  transfer_timeline_++;
}

void Engine::Draw() {
  // handle case where window is minimized
  if (config_.mode() == Mode::Desktop || config_.mode() == Mode::Immediate) {
    int temp_height, temp_width;
    glfwGetWindowSize(window_, &temp_width, &temp_height);
    if (temp_width == 0 || temp_height == 0) {
      return;
    }
  }

  // recreate swapchain if need resize
  if (swapchain_->ShouldRecreate()) {
    vkWaitForFences(context_.device(), render_finished_fences_.size(),
                    render_finished_fences_.data(), VK_TRUE, UINT64_MAX);
    vkDeviceWaitIdle(context_.device());
    swapchain_->Recreate();

    width_ = swapchain_->width();
    height_ = swapchain_->height();

    fov_info_.res[0] = width_;
    fov_info_.res[1] = height_;
    fov_layers_.initialize(fov_info_, &config_.fov_res(),&config_.radii_levels());

    for (uint32_t view = 0; view < num_views_; view++) {
      for (uint32_t level = 0; level < config_.num_levels(); level++) {
        auto layer = fov_layers_.getLayers()[level];
        RecreateFramebuffer(view, level, layer);
      }
    }
    RecreateCompositorFramebuffer();
  }

  int32_t acquire_index = frame_counter_ % num_cb_;
  int32_t frame_index = frame_counter_ % num_frames_;
  VkSemaphore image_acquired_semaphore =
      image_acquired_semaphores_[acquire_index];
  VkSemaphore render_finished_semaphore =
      render_finished_semaphores_[frame_index];
  VkFence render_finished_fence = render_finished_fences_[frame_index];
  VkCommandBuffer cb = draw_command_buffers_[frame_index];

  // record command buffer
  vkWaitForFences(context_.device(), 1, &render_finished_fence, VK_TRUE,
                  UINT64_MAX);
  vkResetFences(context_.device(), 1, &render_finished_fence);

  uint32_t image_index;
  if (swapchain_->AcquireNextImage(image_acquired_semaphore, &image_index)) {
    bool depth_format_changed = false;
    static int depth_format = 1;

    VkCommandBufferBeginInfo command_begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &command_begin_info);

    for (uint32_t view = 0; view < num_views_; view++) {
      DrawView(view, frame_index, acquire_index, image_index, cb);
    }

    vkEndCommandBuffer(cb);

    std::vector<VkSemaphore> wait_semaphores = {image_acquired_semaphore,
                                                transfer_semaphore_};
    std::vector<VkPipelineStageFlags> wait_stages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    std::vector<uint64_t> wait_values = {0, transfer_timeline_};

    VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_semaphore_submit_info.waitSemaphoreValueCount = wait_values.size();
    timeline_semaphore_submit_info.pWaitSemaphoreValues = wait_values.data();

    if (config_.mode() == Mode::VR) {
      VkPipelineStageFlags stage_mask =
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

      VkSubmitInfo submit_info{};
      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount = 1;
      submit_info.pCommandBuffers = &cb;
      submit_info.signalSemaphoreCount = 0;
      submit_info.pSignalSemaphores = nullptr;

      vkQueueSubmit(context_.graphics_queue(), 1, &submit_info,
                    render_finished_fence);
    } else {
      VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit_info.pNext = &timeline_semaphore_submit_info;
      submit_info.waitSemaphoreCount = wait_semaphores.size();
      submit_info.pWaitSemaphores = wait_semaphores.data();
      submit_info.pWaitDstStageMask = wait_stages.data();
      submit_info.commandBufferCount = 1;
      submit_info.pCommandBuffers = &cb;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores = &render_finished_semaphore;
      vkQueueSubmit(context_.graphics_queue(), 1, &submit_info,
                    render_finished_fence);

      VkSwapchainKHR swapchain_handle = swapchain_->swapchain();
      VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      present_info.waitSemaphoreCount = 1;
      present_info.pWaitSemaphores = &render_finished_semaphore;
      present_info.swapchainCount = 1;
      present_info.pSwapchains = &swapchain_handle;
      present_info.pImageIndices = &image_index;
      frame_infos_[0][0][frame_index].present_timestamp = Clock::timestamp();

      VkResult result =
          vkQueuePresentKHR(context_.graphics_queue(), &present_info);
      if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_->ShouldRecreate(true);
      }
      frame_infos_[0][0][frame_index].present_done_timestamp = Clock::timestamp();
    }

    // save image
    if (recorder_mode_) {
      std::string save_type;
      if (config_.mode() == Mode::Immediate) {
        save_type = "array";
      } else {
        save_type = "image";
      }
      
      recorder_.saveImage(std::string(recorder_path_), context_,
                          swapchain_->image(image_index), save_type);
    }

    frame_counter_++;
    swapchain_->ReleaseImage();
  }
}

void Engine::DrawView(uint32_t view_index, uint32_t frame_index,
                      uint32_t acquire_index, uint32_t image_index,
                      VkCommandBuffer cb) {

  // camera matrix
  glm::mat4 model(1.0);

  // draw ui
  if (config_.mode() == Mode::Desktop) {
    auto gui_info = gui_.update(*this, frame_index);
    model = gui_info.model_;
  } else {
    model = GetModelMatrix();
  }

  // get timestamps
  if (frame_infos_[view_index][0][frame_index].drew_splats) {
    for (int level = 0; level < config_.num_levels(); level++) {
      std::vector<uint64_t> timestamps(timestamp_count_);
      VkResult query_results = vkGetQueryPoolResults(
          context_.device(), timestamp_query_pools_[view_index][level][frame_index], 0,
          timestamps.size(), timestamps.size() * sizeof(uint64_t),
          timestamps.data(), sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

      frame_infos_[view_index][level][frame_index].rank_time =
          timestamps[2] - timestamps[1];
      frame_infos_[view_index][level][frame_index].sort_time =
          timestamps[4] - timestamps[3];
      frame_infos_[view_index][level][frame_index].inverse_time =
          timestamps[6] - timestamps[5];
      frame_infos_[view_index][level][frame_index].projection_time =
          timestamps[8] - timestamps[7];
      frame_infos_[view_index][level][frame_index].rendering_time =
          timestamps[10] - timestamps[9];
      frame_infos_[view_index][level][frame_index].end_to_end_time =
          timestamps[11] - timestamps[0];

      if (benchmark_mode_) {
        benchmark_.recordTimingData(frame_infos_[view_index][level][frame_index], level);
        benchmark_.recordLODData(level, lod_levels_[level]);
      }
    }
    if (benchmark_mode_) {
      benchmark_.incrementFrame();
    }
  }

  if (benchmark_mode_ && benchmark_.isRecordingDone() && config_.mode() != Mode::Immediate) {
    benchmark_.saveTimingData(std::string(recorder_path_));
  }

  // Add eye tracking logic here
  if (config_.mode() == Mode::VR) {
    if (context_.enable_eye_tracker()) {
      auto xr_view = context_.xr_manager()->views()[view_index];
      auto projection_matrix = camera_->ProjectionMatrixXR(
          xr_view.fov.angleRight, xr_view.fov.angleLeft, xr_view.fov.angleDown,
          xr_view.fov.angleUp, width_, height_, width_, height_, 0.5, 0.5);
      auto view_matrix = context_.xr_manager()->view_matrix(view_index);

      auto eye_position = context_.eye_tracker()->getLastEyePosition(
          view_index, view_matrix, projection_matrix);
      for (uint32_t i = 0; i < num_views_; i++) {
        center_[i][0] = eye_position.x;
        center_[i][1] = eye_position.y;
      }
    } else {
      for (uint32_t i = 0; i < num_views_; i++) {
        auto xr_view = context_.xr_manager()->views()[i];
        auto x_l_h = 1.0f / cos(abs(xr_view.fov.angleLeft));
        auto x_l_o = sqrt((x_l_h * x_l_h) + (1 * 1));
        auto x_r_h = 1.0f / cos(abs(xr_view.fov.angleRight));
        auto x_r_o = sqrt((x_r_h * x_r_h) + (1 * 1));

        auto y_angle = abs(xr_view.fov.angleUp);
        auto y_u_h = 1.0f / cos(abs(xr_view.fov.angleUp));
        auto y_u_o = sqrt((y_u_h * y_u_h) + (1 * 1));
        auto y_d_h = 1.0f / cos(abs(xr_view.fov.angleDown));
        auto y_d_o = sqrt((y_d_h * y_d_h) + (1 * 1));

        center_[i][0] = x_l_o / (x_l_o + x_r_o);
        center_[i][1] = y_u_o / (y_u_o + y_d_o);
      }
    }
  } else if (config_.mode() == Mode::Immediate) {
    // don't overwrite center
  } else if (gui_.show_center_) {
    auto mouse_pos = ImGui::GetMousePos();
    auto display_size = ImGui::GetIO().DisplaySize;
    center_[view_index] =
        glm::vec2((float)mouse_pos[0] / (float)display_size.x,
                  (float)mouse_pos[1] / (float)display_size.y);
  } else {
    center_[view_index] =
        glm::vec2((float)(fov_info_.res[0] / 2) / (float)(fov_info_.res[0]),
                  (float)(fov_info_.res[1] / 2) / (float)(fov_info_.res[1]));
  }

  center_[view_index] = glm::clamp(center_[view_index], glm::vec2(0.0), glm::vec2(1.0));
  glm::vec3 eye;

  for (int level = 0; level < config_.num_levels(); level++) {
    auto layer = fov_layers_.getLayers();
    float layer_width = (float)layer[level].projection_res[0];
    float layer_height = (float)layer[level].projection_res[1];
    float res_x = (float)fov_info_.res[0];
    float res_y = (float)fov_info_.res[1];
    float cx = (level == config_.num_levels() - 1) ? 0.5f : center_[view_index].x;
    float cy = (level == config_.num_levels() - 1) ? 0.5f : center_[view_index].y;    
    
    if (config_.mode() == Mode::VR) {
      auto xr_view = context_.xr_manager()->views()[view_index];
      camera_buffer_[view_index][level][frame_index].projection =
          camera_->ProjectionMatrixXR(
              xr_view.fov.angleRight, xr_view.fov.angleLeft,
              xr_view.fov.angleDown, xr_view.fov.angleUp, layer_width,
              layer_height, res_x, res_y, cx, cy);
      camera_buffer_[view_index][level][frame_index].view =
          context_.xr_manager()->view_matrix(view_index);
      camera_buffer_[view_index][level][frame_index].eye = glm::mat4(1.0);
      eye = camera_->Eye();
    } else {
      core::ViewFrustumAngles view_frustum_angles;
      if (config_.mode() == Mode::Desktop) {
        auto camera_lookat = std::dynamic_pointer_cast<vkgs::CameraLookAt>(camera_);
        view_frustum_angles = fov::cameraFovyToAngles(camera_lookat->fov(), width_, height_);
      } else if (config_.mode() == Mode::Immediate) {
        auto camera_global = std::dynamic_pointer_cast<vkgs::CameraGlobal>(camera_);
        view_frustum_angles = camera_global->view_angles();
      }

      glm::mat4 view_matrix;
      if (recorder_mode_ && config_.mode() == Mode::Desktop) {
        float alpha = (float)recorder_.frame() / (float)config_.num_frames_recorder();
        auto cam_matrix = view_frustum_dataset_->GetMatrixInterpolated(alpha, 6);
        view_matrix = glm::inverse(cam_matrix);
        eye = glm::vec3(cam_matrix[3]);
      } else if (benchmark_mode_ && config_.mode() == Mode::Desktop) {
        float alpha = (float)benchmark_.frame() / (float)config_.num_frames_benchmark();
        auto cam_matrix = view_frustum_dataset_->GetMatrixInterpolated(alpha, 6);
        view_matrix = glm::inverse(cam_matrix);
        eye = glm::vec3(cam_matrix[3]);
      } else {
        view_matrix = camera_->ViewMatrix();
        eye = camera_->Eye();
      }

      if (view_frustum_dataset_ && view_frustum_dataset_->size() > 0) {
        view_frustum_angles = (*view_frustum_dataset_)[0].sample_state.view_angles[0];
      }

      auto projection_matrix = camera_->ProjectionMatrixXR(
          view_frustum_angles.angle_right, view_frustum_angles.angle_left,
          view_frustum_angles.angle_down, view_frustum_angles.angle_up,
          layer_width, layer_height, res_x, res_y, cx, cy);
      camera_buffer_[view_index][level][frame_index].projection =
          projection_matrix;
      camera_buffer_[view_index][level][frame_index].view = view_matrix;
    }
    camera_buffer_[view_index][level][frame_index].z_near = camera_->Near();
    camera_buffer_[view_index][level][frame_index].z_far = camera_->Far();
    camera_buffer_[view_index][level][frame_index].camera_position = eye;
    camera_buffer_[view_index][level][frame_index].screen_size = {camera_->width(),
                                                      camera_->height()};
    if (level == config_.num_levels() - 1) {      
      camera_buffer_[view_index][level][frame_index].frustum_pad_x = 1.3f;
      camera_buffer_[view_index][level][frame_index].frustum_pad_y = 1.3f;
    } else {
      auto diameter_full = std::min<uint32_t>(camera_->width(), camera_->height());
      auto diameter_layer = std::min<uint32_t>(layer_width, layer_height);
      auto padding_full = diameter_full * 0.3f;
      auto padding_factor = padding_full / diameter_layer;
      camera_buffer_[view_index][level][frame_index].frustum_pad_x = 1.0 + padding_factor;
      camera_buffer_[view_index][level][frame_index].frustum_pad_y = 1.0 + padding_factor;
    }
  }

  if (lazy_budget_mode_) {
    float last_render_time =
        frame_infos_[0][0][frame_index].end_to_end_time / (1e6);
    if (lazy_budget_frame_times_.size() >= 10) {
      lazy_budget_frame_times_.pop_front();
    }
    lazy_budget_frame_times_.push_back(last_render_time);

    for (int32_t i = 0; i < config_.num_levels(); i++) {
      auto lazy_budget_frame_times = std::vector<float>{
          lazy_budget_frame_times_.begin(), lazy_budget_frame_times_.end()};
      size_t mid_index = (size_t)(lazy_budget_frame_times.size() / 2);

      float mean_time = 0;
      for (int32_t j = 0; j < lazy_budget_frame_times.size(); j++) {
        mean_time += lazy_budget_frame_times[j];
      }
      mean_time /= lazy_budget_frame_times.size();
      std::nth_element(lazy_budget_frame_times.begin(),
                       lazy_budget_frame_times.begin() + mid_index,
                       lazy_budget_frame_times.end());
      float median_time = lazy_budget_frame_times[mid_index];

      float lod_scaled = (time_budget_ / median_time) * last_lod_levels_[i];
      lod_scaled = glm::clamp(lod_scaled, 0.05f, 1.0f);
      float lod_smooth = lod_scaled * 0.2 + 0.8 * last_lod_levels_[i];
      last_lod_levels_[i] = lod_smooth;
      lod_levels_[i] = lod_smooth;
    }
  }

  // if recording, overwrite with benchmark data
  if (recorder_mode_ && config_.mode() == Mode::Desktop) {
    for (int level = 0; level < config_.num_levels(); level++) {
      lod_levels_[level] = benchmark_.getLODLevel(level, recorder_.frame());
    }
  }

  // check loading status
  std::vector<bool> progress_barrier_buffers_empty;
  std::vector<vkgs::SplatLoadThread::Progress> progress(config_.num_levels());

  for (int slt_i = 0; slt_i < splat_load_thread_[view_index].size(); slt_i++) {
    progress[slt_i] = splat_load_thread_[view_index][slt_i].GetProgress();

    frame_infos_[view_index][slt_i][frame_index].total_point_count =
        progress[slt_i].total_point_count;
    frame_infos_[view_index][slt_i][frame_index].loaded_point_count =
        progress[slt_i].loaded_point_count;
    frame_infos_[view_index][slt_i][frame_index].ply_buffer =
        progress[slt_i].ply_buffer;

    progress_barrier_buffers_empty.push_back(
        progress[slt_i].buffer_barriers.empty());
  }

  for (int level = 0; level < config_.num_levels(); level++) {
    VkQueryPool timestamp_query_pool =
        timestamp_query_pools_[view_index][level][frame_index];

    vkCmdResetQueryPool(cb, timestamp_query_pool, 0, timestamp_count_);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        timestamp_query_pool, 0);

    // update descriptor
    descriptors_[view_index][level][frame_index].gaussian.Update(
        0, splat_info_buffer_[view_index][level],
        splat_info_buffer_[view_index][level].offset(frame_index),
        splat_info_buffer_[view_index][level].element_size());

    descriptors_[view_index][level][frame_index].splat_instance.Update(
        0, splat_draw_indirect_[view_index][level], 0,
        splat_draw_indirect_[view_index][level].size());

    // count loaded splats
    uint32_t loaded_point_count = 0;
    if (config_.lod()) {
      if (!progress_barrier_buffers_empty[level] &&
          (loaded_point_count_[view_index][level] == 0)) {
        loaded_point_count_[view_index][level] = progress[level].loaded_point_count;
      }
      loaded_point_count = loaded_point_count_[view_index][level];
    } else {
      if (!progress_barrier_buffers_empty[0] &&
          (loaded_point_count_[view_index][0] == 0)) {
        loaded_point_count_[view_index][0] = progress[0].loaded_point_count;
      }
      loaded_point_count = loaded_point_count_[view_index][0];
    }

    // update descriptors
    if (loaded_point_count > 0) {
      descriptors_[view_index][level][frame_index].gaussian.Update(
          1, splat_storage_[view_index][level].position, 0,
          loaded_point_count * 3 * sizeof(float));
      descriptors_[view_index][level][frame_index].gaussian.Update(
          2, splat_storage_[view_index][level].cov3d, 0,
          loaded_point_count * 6 * sizeof(float));
      descriptors_[view_index][level][frame_index].gaussian.Update(
          3, splat_storage_[view_index][level].opacity, 0,
          loaded_point_count * 1 * sizeof(float));
      descriptors_[view_index][level][frame_index].gaussian.Update(
          4, splat_storage_[view_index][level].sh, 0,
          loaded_point_count * 48 * sizeof(float));

      descriptors_[view_index][level][frame_index].splat_instance.Update(
          1, splat_storage_[view_index][level].instance, 0,
          loaded_point_count * 10 * sizeof(float));
      descriptors_[view_index][level][frame_index].splat_instance.Update(
          2, splat_visible_point_count_[view_index][level], 0,
          splat_visible_point_count_[view_index][level].size());
      descriptors_[view_index][level][frame_index].splat_instance.Update(
          3, splat_storage_[view_index][level].key, 0,
          loaded_point_count * sizeof(uint32_t));
      descriptors_[view_index][level][frame_index].splat_instance.Update(
          4, splat_storage_[view_index][level].index, 0,
          loaded_point_count * sizeof(uint32_t));
      descriptors_[view_index][level][frame_index].splat_instance.Update(
          5, splat_storage_[view_index][level].inverse_index, 0,
          loaded_point_count * sizeof(uint32_t));

      // update uniform buffer
      splat_info_buffer_[view_index][level][frame_index].point_count =
           (uint32_t)(loaded_point_count * lod_levels_[level]);
    }
  }

  VkMemoryBarrier barrier;
  
  // acquire ownership
  // according to spec:
  //   The buffer range or image subresource range specified in an
  //   acquireoperation must match exactly that of a previous release
  //   operation.
  bool all_buffers_full = SplatsLoaded(view_index);

  if (all_buffers_full && !loaded_point_[view_index][frame_index]) {
      for (int level = 0; level < config_.num_levels(); level++) {
        vkgs::SplatLoadThread::Progress* progress_single;

        uint32_t loaded_point_count = 0;
        VkBuffer ply_buffer;
        if (config_.lod()) {
          progress_single = &progress[level];
          loaded_point_count = loaded_point_count_[view_index][level];
          ply_buffer = frame_infos_[view_index][level][frame_index].ply_buffer;
        } else {
          progress_single = &progress[0];
          loaded_point_count = loaded_point_count_[view_index][0];
          ply_buffer = frame_infos_[view_index][0][frame_index].ply_buffer;
        }

        std::vector<VkBufferMemoryBarrier> buffer_barriers =
            std::move(progress_single->buffer_barriers);

        // change src/dst synchronization scope
        for (auto& buffer_barrier : buffer_barriers) {
          buffer_barrier.srcAccessMask = 0;
          buffer_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }

        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                             buffer_barriers.size(), buffer_barriers.data(), 0,
                             NULL);

        // parse ply file
        // TODO: make parse async
        descriptors_[view_index][level][frame_index].ply.Update(0, ply_buffer, 0);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, parse_ply_pipeline_);

        VkDescriptorSet descriptor =
            descriptors_[view_index][level][frame_index].gaussian;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_layout_, 1, 1, &descriptor, 0,
                                NULL);

        descriptor = descriptors_[view_index][level][frame_index].ply;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_layout_, 3, 1, &descriptor, 0,
                                NULL);

        constexpr int local_size = 256;

        vk::debug::set_cmd_insert_label(context_, cb,
                                        std::string("Loading .ply"));
        vkCmdDispatch(cb, (loaded_point_count + local_size - 1) / local_size, 1,
                      1);

        buffer_barriers.resize(4);
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);

        // hold buffer until the end of frame
        frame_infos_[view_index][level][frame_index].ply_buffer =
            progress_single->ply_buffer;
      }
      loaded_point_[view_index][frame_index] = true;
    }
  for (int level = 0; level < config_.num_levels(); level++) {
    uint32_t loaded_point_count = 0;
    if (config_.lod()) {
      loaded_point_count = loaded_point_count_[view_index][level];
    } else {
      loaded_point_count = loaded_point_count_[view_index][0];
    }

    if (loaded_point_count != 0) {
      uint32_t loaded_point_count_lod =
          (uint32_t)(loaded_point_count * lod_levels_[level]);

      RankPushConstantData rank_push_constant_data{};
      rank_push_constant_data.model = model;
      rank_push_constant_data.lod_params = lod_params_[level];
      if (lod_params_[level].x * loaded_point_count_lod < loaded_point_count * 0.05) {
        rank_push_constant_data.lod_params.x = (loaded_point_count * 0.05) / (loaded_point_count_lod);
      }
      if (lod_params_[level].y * loaded_point_count_lod < loaded_point_count * 0.05) {
        rank_push_constant_data.lod_params.y = (loaded_point_count * 0.05) / (loaded_point_count_lod);
      }

      // rank
      {
        vkCmdFillBuffer(cb, splat_visible_point_count_[view_index][level], 0,
                        sizeof(uint32_t), 0);

        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                              &barrier, 0, NULL, 0, NULL);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          rank_pipeline_);

        std::vector<VkDescriptorSet> descriptors = {
            descriptors_[view_index][level][frame_index].camera,
            descriptors_[view_index][level][frame_index].gaussian,
            descriptors_[view_index][level][frame_index].splat_instance,
        };
        vkCmdBindDescriptorSets(
            cb, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0,
            descriptors.size(), descriptors.data(), 0, nullptr);

        vkCmdPushConstants(cb, compute_pipeline_layout_,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RankPushConstantData),
                            &rank_push_constant_data);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 1);

        constexpr int local_size = 256;
        vkCmdDispatch(
            cb, (loaded_point_count_lod + local_size - 1) / local_size, 1, 1);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 2);
      }

      // visible point count to CPU
      {
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                              0, NULL, 0, NULL);

        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = sizeof(uint32_t) * frame_index;
        region.size = sizeof(uint32_t);
        vkCmdCopyBuffer(cb, splat_visible_point_count_[view_index][level],
                        visible_point_count_cpu_buffer_[view_index][level], 1,
                        &region);
      }

      // radix sort
      {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 3);

        vrdxCmdSortKeyValueIndirect(
            cb, sorter_[view_index][level], loaded_point_count_lod,
            splat_visible_point_count_[view_index][level], 0,
            splat_storage_[view_index][level].key, 0,
            splat_storage_[view_index][level].index, 0,
            sort_storage_[view_index][level], 0, NULL, 0);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 4);
      }

      // inverse map
      {
        vkCmdFillBuffer(cb, splat_storage_[view_index][level].inverse_index, 0,
                        loaded_point_count_lod * sizeof(uint32_t), -1);

        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);

        std::vector<VkDescriptorSet> descriptors = {
            descriptors_[view_index][level][frame_index].camera,
            descriptors_[view_index][level][frame_index].gaussian,
            descriptors_[view_index][level][frame_index].splat_instance,
        };
        vkCmdBindDescriptorSets(
            cb, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0,
            descriptors.size(), descriptors.data(), 0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          inverse_index_pipeline_);

        vkCmdPushConstants(cb, compute_pipeline_layout_,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RankPushConstantData),
                            &rank_push_constant_data);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 5);

        constexpr int local_size = 256;
        vkCmdDispatch(
            cb, (loaded_point_count_lod + local_size - 1) / local_size, 1, 1);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 6);
      }

      // projection
      {
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                  VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                              &barrier, 0, NULL, 0, NULL);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          projection_pipeline_);

        vkCmdPushConstants(cb, compute_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RankPushConstantData),
                           &rank_push_constant_data);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 7);

        constexpr int local_size = 256;
        vkCmdDispatch(
            cb, (loaded_point_count_lod + local_size - 1) / local_size, 1, 1);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 8);
      }

      // draw
      {
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;

        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                  VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                              0, 1, &barrier, 0, NULL, 0, NULL);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 9);

        auto layer = fov_layers_.getLayers()[level];
        auto image_spec = color_attachment_[view_index][level].image_spec();
        DrawNormalPass(cb, view_index, frame_index,
                       ceil(image_spec.width * res_scales_[level]),
                       ceil(image_spec.height * res_scales_[level]),
                       color_attachment_[view_index][level], level);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                            timestamp_query_pools_[view_index][level][frame_index], 10);
      }
    }
  }

  if (all_buffers_full) {
    // run compositor
    {
      DrawCompositorPass(cb, view_index, frame_index, image_index, swapchain_->width(),
                         swapchain_->height(),
                         swapchain_->image_view(view_index, image_index));
    }

    frame_infos_[view_index][0][frame_index].drew_splats = true;
  } else {
    for (int level = 0; level < config_.num_levels(); level++) {
      auto layer = fov_layers_.getLayers()[level];
      auto image_spec = color_attachment_[view_index][level].image_spec();
      DrawNormalPass(cb, view_index, frame_index, image_spec.width,
                      image_spec.height, color_attachment_[view_index][level], level);
    }

    DrawCompositorPass(cb, view_index, frame_index, image_index, swapchain_->width(),
                       swapchain_->height(),
                       swapchain_->image_view(view_index, image_index));
    frame_infos_[view_index][0][frame_index].drew_splats = false;
  }

  for (int level = 0; level < config_.num_levels(); level++) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        timestamp_query_pools_[view_index][level][frame_index],
                        11);
  }
}

void Engine::DrawCompositorPass(
    VkCommandBuffer cb,
    uint32_t view_index,
    uint32_t frame_index,
    uint32_t image_index,
    uint32_t width,
    uint32_t height,
    VkImageView target_image_view
) {
  std::vector<VkClearValue> clear_values(2);
  clear_values[0].color.float32[0] = 0.0f;
  clear_values[0].color.float32[1] = 0.0f;
  clear_values[0].color.float32[2] = 0.0f;
  clear_values[0].color.float32[3] = 1.f;
  clear_values[1].depthStencil.depth = 1.f;

  std::vector<VkImageView> render_pass_attachments;

  render_pass_attachments = {
      target_image_view,
      depth_compositor_attachment_[view_index][frame_index],
  };

  VkRenderPassBeginInfo render_pass_begin_info = {
      VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  render_pass_begin_info.framebuffer = framebuffer_compositor_[view_index][image_index];
  render_pass_begin_info.renderArea.offset = {0, 0};
  render_pass_begin_info.renderArea.extent = {width, height};
  render_pass_begin_info.clearValueCount = clear_values.size();
  render_pass_begin_info.pClearValues = clear_values.data();
  render_pass_begin_info.renderPass = render_pass_compositor_;

  vkCmdBeginRenderPass(cb, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport = {};
  viewport.x = 0.f;
  viewport.y = 0.f;
  viewport.width = static_cast<float>(width);
  viewport.height = static_cast<float>(height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  vkCmdSetViewport(cb, 0, 1, &viewport);

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = {width, height};
  vkCmdSetScissor(cb, 0, 1, &scissor);

  std::vector<VkDescriptorSet> descriptors = {compositor_.getDescriptorSet(view_index)};
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          compositor_.getPipelineLayout(), 0,
                          descriptors.size(), descriptors.data(), 0, nullptr);

  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, compositor_.getGraphicsPipeline());
    
  
  vk::CompositorPushConstantData compositor_push_data;
  compositor_push_data.center[0] = center_[view_index].x;
  compositor_push_data.center[1] = center_[view_index].y;
  for (int level = 0; level < config_.num_levels(); level++) {
    auto layer_res = fov_layers_.getLayers()[level].render_res;

    float s_i = pow(2.0, level);
    compositor_push_data.levels[level][0] =
        ((float)layer_res[0] * s_i) / (float)fov_info_.res[0];
    compositor_push_data.levels[level][1] =
        ((float)layer_res[1] * s_i) / (float)fov_info_.res[1];

    compositor_push_data.res_scales[level] = res_scales_[level];
  }
  compositor_push_data.full_res[0] = width_;
  compositor_push_data.full_res[1] = height_;
  compositor_push_data.blending = gui_.blending_mode_;
  compositor_push_data.eye = view_index;

  vkCmdPushConstants(
      cb, compositor_.getPipelineLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
      sizeof(vk::CompositorPushConstantData), &compositor_push_data);

  std::vector<VkBuffer> vbs = {compositor_.getVertexBuffer()};
  std::vector<VkDeviceSize> vb_offsets = {0};
  vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(), vb_offsets.data());

  vkCmdBindIndexBuffer(cb, compositor_.getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

  vkCmdDrawIndexed(cb, 6, 1, 0, 0, 0);

  // draw ui
  if (!recorder_mode_ && !benchmark_mode_ && config_.mode() == Mode::Desktop) {
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, cb);
  }

  vkCmdEndRenderPass(cb);
}

void Engine::DrawNormalPass(VkCommandBuffer cb, uint32_t view_index,
                            uint32_t frame_index, uint32_t width,
                            uint32_t height, VkImageView target_image_view,
                            int level) {
  std::vector<VkClearValue> clear_values(2);
  clear_values[0].color.float32[0] = 0.0f;
  clear_values[0].color.float32[1] = 0.0f;
  clear_values[0].color.float32[2] = 0.0f;
  clear_values[0].color.float32[3] = 1.f;
  clear_values[1].depthStencil.depth = 1.f;

  VkRenderPassBeginInfo render_pass_begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  render_pass_begin_info.framebuffer = framebuffer_[view_index][level];
  render_pass_begin_info.renderArea.offset = {0, 0};
  render_pass_begin_info.renderArea.extent = {width, height};
  render_pass_begin_info.clearValueCount = clear_values.size();
  render_pass_begin_info.pClearValues = clear_values.data();
  render_pass_begin_info.renderPass = render_passes_[view_index][level];

  vkCmdBeginRenderPass(cb, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport = {};
  viewport.x = 0.f;
  viewport.y = 0.f;
  viewport.width = static_cast<float>(width);
  viewport.height = static_cast<float>(height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  vkCmdSetViewport(cb, 0, 1, &viewport);

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = {width, height};
  vkCmdSetScissor(cb, 0, 1, &scissor);

  std::vector<VkDescriptorSet> descriptors = {
    descriptors_[view_index][level][frame_index].camera,
    descriptors_[view_index][level][frame_index].splat_instance,
  };
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          graphics_pipeline_layout_, 0, descriptors.size(),
                          descriptors.data(), 0, nullptr);

  // draw axis and grid
  {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, color_line_pipeline_);

    glm::mat4 model(1.f);
    auto time_point = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto time = (uint32_t)(time_point & 0xffffffff);

    SplatPushConstantData splat_push_constant_data;
    splat_push_constant_data.model = model;
    splat_push_constant_data.time = time;
    splat_push_constant_data.vis_mode = vis_mode_;
    splat_push_constant_data.vis_mode_scale = vis_mode_scale_;
    vkCmdPushConstants(cb, graphics_pipeline_layout_,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SplatPushConstantData), &splat_push_constant_data);

    if (gui_.show_axis_) {
      axis_->Draw(cb);
    }

    if (gui_.show_grid_) {
      grid_->Draw(cb);
    }
    if (gui_.show_views_) {
      for (uint32_t i = 0; i < view_frustum_meshes_.size(); i++) {
        view_frustum_meshes_[i]->Draw(cb);
      }
    }
  }

  // draw splat
  bool all_buffers_full = true;
  for (int slt_i = 0; slt_i < loaded_point_count_[view_index].size(); slt_i++) {
    all_buffers_full &= (loaded_point_count_[view_index][slt_i] > 0);
  }

  if (all_buffers_full) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, splat_pipelines_[(uint32_t)vis_mode_]);
    vkCmdBindIndexBuffer(cb, splat_index_buffer_[view_index][level], 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(cb, splat_draw_indirect_[view_index][level], 0, 1, 0);
  }

  vkCmdEndRenderPass(cb);
}

void Engine::RecreateCompositorFramebuffer()
{
  for (uint32_t view = 0; view < num_views_; view++) {
    for (uint32_t frame = 0; frame < num_frames_; frame++) {
      depth_compositor_attachment_[view][frame] = vk::Attachment(
          context_, swapchain_->width(), swapchain_->height(), depth_format_,
          samples_, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, num_views_);

      // Update compositor descriptor
      std::vector<VkImageView> descriptors;
      for (int level = 0; level < config_.num_levels(); level++) {
        descriptors.push_back(color_attachment_[view][level].image_view());
      }
      compositor_.updateDescriptor(descriptors,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, view);

      vk::FramebufferCreateInfo framebuffer_info;
      framebuffer_info.width = swapchain_->width();
      framebuffer_info.height = swapchain_->height();
      framebuffer_info.render_pass = render_pass_compositor_;
      framebuffer_info.image_views = {swapchain_->image_view(view, frame),
          depth_compositor_attachment_[view][frame]};
      framebuffer_compositor_[view][frame] = vk::Framebuffer(context_, framebuffer_info);
    }
  }
}

void Engine::RecreateFramebuffer(uint32_t view_index, uint32_t level, fov::FoveatedLayer& layer) {
  uint32_t width = 0;
  uint32_t height = 0;
  if (config_.dynamic_res()) {
    width = layer.projection_res[0];
    height = layer.projection_res[1];
  } else {
    width = layer.render_res[0];
    height = layer.render_res[1];
  }

  color_attachment_[view_index][level] = vk::Attachment(context_, width, height, color_format_, samples_, VK_IMAGE_USAGE_SAMPLED_BIT);
  depth_attachment_[view_index][level] = vk::Attachment(context_, width, height, depth_format_, samples_, VK_IMAGE_USAGE_SAMPLED_BIT);
    
  vk::FramebufferCreateInfo framebuffer_info;
  framebuffer_info.width = width;
  framebuffer_info.height = height;
  framebuffer_info.render_pass = render_passes_[view_index][level];
  framebuffer_info.image_views = {color_attachment_[view_index][level],
                                  depth_attachment_[view_index][level]};
  framebuffer_[view_index][level] = vk::Framebuffer(context_, framebuffer_info);
}

}  // namespace vkgs
