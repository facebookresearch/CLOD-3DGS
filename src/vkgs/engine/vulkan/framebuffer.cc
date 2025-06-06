// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/framebuffer.h"

namespace vkgs {
namespace vk {

/**
  * @brief Implementation of Vulkan Framebuffer
  */
class Framebuffer::Impl {
 public:
  Impl() = delete;

  Impl(Context context, const FramebufferCreateInfo& create_info)
      : context_(context)
  { 
    VkFramebufferCreateInfo framebuffer_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.renderPass = create_info.render_pass;
    framebuffer_info.width = create_info.width;
    framebuffer_info.height = create_info.height;
    framebuffer_info.layers = 1;

    uint32_t image_count = 0;
    
    if (create_info.image_specs.size() > 0) {
      imageless_ = true;
      image_count = create_info.image_specs.size();

      std::vector<VkFramebufferAttachmentImageInfo> attachment_images(image_count);

      for (int i = 0; i < image_count; ++i) {
        attachment_images[i] = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO};
        attachment_images[i].usage = create_info.image_specs[i].usage;
        attachment_images[i].width = create_info.image_specs[i].width;
        attachment_images[i].height = create_info.image_specs[i].height;
        attachment_images[i].layerCount = 1;
        attachment_images[i].viewFormatCount = 1;
        attachment_images[i].pViewFormats = &create_info.image_specs[i].format;
      }

      VkFramebufferAttachmentsCreateInfo attachments_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO};
      attachments_info.attachmentImageInfoCount = attachment_images.size();
      attachments_info.pAttachmentImageInfos = attachment_images.data();

      framebuffer_info.pNext = &attachments_info;
      framebuffer_info.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT;
      framebuffer_info.attachmentCount = image_count;

      vkCreateFramebuffer(context_.device(), &framebuffer_info, NULL, &framebuffer_);
    } else {
      imageless_ = false;
      image_count = create_info.image_views.size();

      framebuffer_info.pAttachments = &create_info.image_views[0];
      framebuffer_info.attachmentCount = image_count;

      vkCreateFramebuffer(context_.device(), &framebuffer_info, NULL, &framebuffer_);
    }
  }

  ~Impl() { vkDestroyFramebuffer(context_.device(), framebuffer_, NULL); }

  operator VkFramebuffer() const noexcept { return framebuffer_; }

 private:
  Context context_;
  bool imageless_;
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
};

Framebuffer::Framebuffer() = default;

Framebuffer::Framebuffer(
  Context context,
  const FramebufferCreateInfo& create_info
) : impl_(std::make_shared<Impl>(context, create_info)) {}

Framebuffer::~Framebuffer() = default;

Framebuffer::operator VkFramebuffer() const { return *impl_; }

}  // namespace vk
}  // namespace vkgs
