// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/swapchain_vr.h"

namespace vkgs {
namespace vk {

SwapchainVR::SwapchainVR(Context context, std::shared_ptr<XRManager> xr_manager,
                         bool vsync)
    : Swapchain(context, vsync), xr_manager_(xr_manager) {
  num_views_ = 2;

  // get view configuration
  XrViewConfigurationView config_view_template{};
  config_view_template.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
  std::vector<XrViewConfigurationView> config_views(num_views_,
                                                    config_view_template);

  xrEnumerateViewConfigurationViews(
      xr_manager->instance(), xr_manager->system(),
      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, num_views_, &num_views_,
      config_views.data());

  width_ = config_views[0].recommendedImageRectWidth;
  height_ = config_views[0].recommendedImageRectHeight;
  usage_ =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  format_ = VK_FORMAT_B8G8R8A8_UNORM;

  // create swapchain
  XrSwapchainCreateInfo swapchain_create_info{};
  swapchain_create_info.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
  swapchain_create_info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                                     XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT;
  swapchain_create_info.format = format_;
  swapchain_create_info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
  swapchain_create_info.width = width_;
  swapchain_create_info.height = height_;
  swapchain_create_info.faceCount = 1;
  swapchain_create_info.arraySize = num_views_;
  swapchain_create_info.mipCount = 1;

  auto result = xrCreateSwapchain(xr_manager->session(), &swapchain_create_info,
                    &xr_swapchain_);

  // get images
  uint32_t image_count;
  xrEnumerateSwapchainImages(xr_swapchain_, 0, &image_count, nullptr);

  XrSwapchainImageVulkanKHR image_template{};
  image_template.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR;
  std::vector<XrSwapchainImageVulkanKHR> images(image_count, image_template);

  xrEnumerateSwapchainImages(xr_swapchain_, image_count, &image_count,
                             (XrSwapchainImageBaseHeader*)images.data());

  for (int i = 0; i < image_count; ++i) {
    images_.push_back(images[i].image);
  }  
  
  image_views_.resize(num_views_);
  for (uint32_t view = 0; view < num_views_; view++) {
    image_views_[view].resize(image_count);
    for (uint32_t frame = 0; frame < image_count; frame++) {
      VkImageViewCreateInfo image_view_info = {
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      image_view_info.image = images_[frame];
      image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      image_view_info.format = format_;
      image_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, view,
                                          1};
      vkCreateImageView(context_.device(), &image_view_info, NULL,
                        &image_views_[view][frame]);
    }
  }
}

SwapchainVR::~SwapchainVR() {
  xrDestroySwapchain(xr_swapchain_);
}

bool SwapchainVR::AcquireNextImage(VkSemaphore semaphore,
                                        uint32_t* image_index) {
  XrSwapchainImageAcquireInfo acquire_image_info{};
  acquire_image_info.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO;
  
  xrAcquireSwapchainImage(xr_swapchain_, &acquire_image_info, image_index);

  XrSwapchainImageWaitInfo wait_image_info{};
  wait_image_info.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO;
  wait_image_info.timeout = (std::numeric_limits<int64_t>::max)();

  auto result = xrWaitSwapchainImage(xr_swapchain_, &wait_image_info);
  return (result == XR_SUCCESS);
}

void SwapchainVR::ReleaseImage() {
  XrSwapchainImageReleaseInfo release_image_info{};
  release_image_info.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO;

  xrReleaseSwapchainImage(xr_swapchain_, &release_image_info);
}

void SwapchainVR::Recreate() {}

}  // namespace vk
}  // namespace vkgs
