// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/pipeline_layout.h"

namespace vkgs {
namespace vk {


/**
 * @brief Implementation of Vulkan PipelineLayout
 */
class PipelineLayout::Impl {
 public:
  Impl() = delete;

  Impl(Context context, const PipelineLayoutCreateInfo& create_info)
      : context_(context) {
    std::vector<VkDescriptorSetLayout> layouts;
    for (auto layout : create_info.layouts) layouts.push_back(layout);

    VkPipelineLayoutCreateInfo layout_info = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = layouts.size();
    layout_info.pSetLayouts = layouts.data();
    layout_info.pushConstantRangeCount = create_info.push_constants.size();
    layout_info.pPushConstantRanges = create_info.push_constants.data();
    vkCreatePipelineLayout(context.device(), &layout_info, NULL, &layout_);
  }

  ~Impl() { vkDestroyPipelineLayout(context_.device(), layout_, NULL); }

  operator VkPipelineLayout() const noexcept { return layout_; }

 private:
  Context context_;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

PipelineLayout::PipelineLayout() = default;

PipelineLayout::PipelineLayout(Context context,
                               const PipelineLayoutCreateInfo& create_info)
    : impl_(std::make_shared<Impl>(context, create_info)) {}

PipelineLayout::~PipelineLayout() = default;

PipelineLayout::operator VkPipelineLayout() const { return *impl_; }

}  // namespace vk
}  // namespace vkgs
