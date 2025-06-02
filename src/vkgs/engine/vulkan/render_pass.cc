// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/render_pass.h"

namespace vkgs {
namespace vk {

/**
 * @brief Implementation of Vulkan RenderPass class
 */
class RenderPass::Impl {
 public:
  Impl() = delete;

  Impl(Context context, VkSampleCountFlagBits samples, VkFormat color_format, VkFormat depth_format,
       uint32_t num_views,
       VkImageLayout color_final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
      : context_(context) {
    std::vector<VkAttachmentDescription> attachments;
    attachments.resize(2);
    attachments[0].format = color_format;
    attachments[0].samples = samples;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = color_final_layout;
    attachments[0].flags = 0;

    attachments[1].format = depth_format;
    attachments[1].samples = samples;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].flags = 0;

    std::vector<VkAttachmentReference> pass0_colors(1);
    pass0_colors[0].attachment = 0;
    pass0_colors[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference pass0_depth{};
    pass0_depth.attachment = 1;
    pass0_depth.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<VkSubpassDescription> subpasses(1);
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].flags = 0;
    subpasses[0].inputAttachmentCount = 0;
    subpasses[0].pInputAttachments = nullptr;
    subpasses[0].colorAttachmentCount = pass0_colors.size();
    subpasses[0].pColorAttachments = pass0_colors.data();
    subpasses[0].pResolveAttachments = nullptr;
    subpasses[0].pDepthStencilAttachment = &pass0_depth;
    subpasses[0].preserveAttachmentCount = 0;
    subpasses[0].pPreserveAttachments = nullptr;

    std::vector<VkSubpassDependency> dependencies;  
    {
      VkSubpassDependency dependency{};
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;
      dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
      dependencies.push_back(dependency);
    }

    {
      VkSubpassDependency dependency{};
      dependency.srcSubpass = 0;
      dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
      dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
      dependencies.push_back(dependency);
    }

    VkRenderPassCreateInfo render_pass_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass_info.pNext = nullptr;
    render_pass_info.attachmentCount = attachments.size();
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = subpasses.size();
    render_pass_info.pSubpasses = subpasses.data();
    render_pass_info.dependencyCount = dependencies.size();
    render_pass_info.pDependencies = dependencies.data();
    vkCreateRenderPass(context_.device(), &render_pass_info, NULL, &render_pass_);
  }

  ~Impl() { vkDestroyRenderPass(context_.device(), render_pass_, NULL); }

  operator VkRenderPass() const noexcept { return render_pass_; }

 private:
  Context context_;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
};

RenderPass::RenderPass() = default;

RenderPass::RenderPass(Context context, VkSampleCountFlagBits samples,
                       VkFormat color_format, VkFormat depth_format,
                       uint32_t num_views, VkImageLayout color_final_layout)
    : impl_(std::make_shared<Impl>(context, samples, color_format, depth_format, num_views,
                                   color_final_layout)) {}

RenderPass::~RenderPass() = default;

RenderPass::operator VkRenderPass() const { return *impl_; }

}  // namespace vk
}  // namespace vkgs
