// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/descriptor_layout.h"

#include <unordered_map>

namespace vkgs {
namespace vk {

class DescriptorLayout::Impl {
 public:
  Impl() = delete;

  Impl(Context context, const DescriptorLayoutCreateInfo& create_info)
      : context_(context) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto& binding : create_info.bindings) {
      VkDescriptorSetLayoutBinding raw_binding;
      raw_binding.binding = binding.binding;
      raw_binding.descriptorType = binding.descriptor_type;
      raw_binding.descriptorCount = binding.descriptor_count;
      raw_binding.stageFlags = binding.stage_flags;
      raw_binding.pImmutableSamplers = VK_NULL_HANDLE;
      bindings.push_back(raw_binding);

      types_[binding.binding] = binding.descriptor_type;
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(context.device(), &layout_info, NULL, &layout_);
  }

  ~Impl() { vkDestroyDescriptorSetLayout(context_.device(), layout_, NULL); }

  operator VkDescriptorSetLayout() const noexcept { return layout_; }

  VkDescriptorType type(uint32_t binding) const { return types_.at(binding); }

 private:
  Context context_;
  VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
  std::unordered_map<uint32_t, VkDescriptorType> types_;
};

DescriptorLayout::DescriptorLayout() = default;

DescriptorLayout::DescriptorLayout(
    Context context, const DescriptorLayoutCreateInfo& create_info)
    : impl_(std::make_shared<Impl>(context, create_info)) {}

DescriptorLayout::~DescriptorLayout() = default;

DescriptorLayout::operator VkDescriptorSetLayout() const { return *impl_; }

VkDescriptorType DescriptorLayout::type(uint32_t binding) const {
  return impl_->type(binding);
}

}  // namespace vk
}  // namespace vkgs
