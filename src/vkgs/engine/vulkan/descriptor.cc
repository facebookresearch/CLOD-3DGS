// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/descriptor.h"

namespace vkgs {
namespace vk {


/**
 * @brief Implementation of Vulkan Descriptor
 */
class Descriptor::Impl {
 public:
  Impl() = delete;

  Impl(Context context, DescriptorLayout layout)
      : context_(context), layout_(layout)
  {
    this->initialize(context, layout);
  }

  void initialize(Context context, DescriptorLayout layout)
  {
    VkDescriptorSetLayout layout_handle = layout;

    VkDescriptorSetAllocateInfo descriptor_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptor_info.descriptorPool = context.descriptor_pool();
    descriptor_info.descriptorSetCount = 1;
    descriptor_info.pSetLayouts = &layout_handle;
    vkAllocateDescriptorSets(context.device(), &descriptor_info, &descriptor_);
  }

  ~Impl() {}

  operator VkDescriptorSet() const noexcept { return descriptor_; }

  void Update(
    uint32_t binding,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size
  ) {
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = size > 0 ? size : VK_WHOLE_SIZE;

    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = layout_.type(binding);
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(context_.device(), 1, &write, 0, NULL);
  }

  void Update(uint32_t binding, VkSampler sampler,
              std::vector<VkImageView>& image_views, VkImageLayout image_layout,
              VkDeviceSize offset, VkDeviceSize size
  ) {
    VkDescriptorImageInfo sampler_info;
    sampler_info.sampler = sampler;
    sampler_info.imageView = VK_NULL_HANDLE;
    sampler_info.imageLayout = image_layout;

    std::vector<VkDescriptorImageInfo> image_infos(image_views.size());
    for (int level = 0; level < image_views.size(); level++)
    {
      image_infos[level].sampler = VK_NULL_HANDLE;
      image_infos[level].imageView = image_views[level];
      image_infos[level].imageLayout = image_layout;
    }

    std::vector<VkWriteDescriptorSet> write(2);

    write[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write[0].pNext = nullptr;
    write[0].dstSet = descriptor_;
    write[0].dstBinding = 0;
    write[0].dstArrayElement = 0;
    write[0].descriptorCount = 1;
    write[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write[0].pBufferInfo = nullptr;
    write[0].pImageInfo = &sampler_info;
    write[0].pTexelBufferView = nullptr;

    write[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write[1].pNext = nullptr;
    write[1].dstSet = descriptor_;
    write[1].dstBinding = 1;
    write[1].dstArrayElement = 0;
    write[1].descriptorCount = (uint32_t)image_infos.size();
    write[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write[1].pBufferInfo = nullptr;
    write[1].pImageInfo = &image_infos[0];
    write[1].pTexelBufferView = nullptr;
    vkUpdateDescriptorSets(context_.device(), (uint32_t)write.size(), &write[0], 0, nullptr);
  }

  void UpdateInputAttachment(uint32_t binding, VkImageView image_view) {
    VkDescriptorImageInfo image_info = {};
    image_info.sampler = VK_NULL_HANDLE;
    image_info.imageView = image_view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor_;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = layout_.type(binding);
    write.pImageInfo = &image_info;
    vkUpdateDescriptorSets(context_.device(), 1, &write, 0, NULL);
  }

 private:
  Context context_;
  DescriptorLayout layout_;
  VkDescriptorSet descriptor_ = VK_NULL_HANDLE;
};

Descriptor::Descriptor() = default;

Descriptor::Descriptor(Context context, DescriptorLayout layout)
    : impl_(std::make_shared<Impl>(context, layout)) {}

void Descriptor::initialize(Context context, DescriptorLayout layout)
{
  impl_->initialize(context, layout);
}

Descriptor::~Descriptor() = default;

Descriptor::operator VkDescriptorSet() const { return *impl_; }

void Descriptor::Update(
  uint32_t binding,
  VkBuffer buffer,
  VkDeviceSize offset,
  VkDeviceSize size
) {
  impl_->Update(binding, buffer, offset, size);
}

void Descriptor::Update(uint32_t binding, VkSampler sampler,
                        std::vector<VkImageView>& image_views,
                        VkImageLayout image_layout, VkDeviceSize offset,
                        VkDeviceSize size) {
  impl_->Update(binding, sampler, image_views, image_layout, offset, size);
}

void Descriptor::UpdateInputAttachment(uint32_t binding,
                                       VkImageView image_view) {
  impl_->UpdateInputAttachment(binding, image_view);
}

}  // namespace vk
}  // namespace vkgs
