// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/swapchain_desktop.h"

namespace vkgs {
namespace vk {

SwapchainDesktop::SwapchainDesktop(Context context, VkSurfaceKHR surface, uint32_t num_frames, bool vsync)
    : Swapchain(context, vsync), surface_(surface), num_frames_(num_frames) {
  num_views_ = 1;

  if (vsync) {
    present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  } else {
    present_mode_ = VK_PRESENT_MODE_MAILBOX_KHR;
  }

  usage_ = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  format_ = VK_FORMAT_B8G8R8A8_UNORM;

  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physical_device(),
                                            surface_, &surface_capabilities);

  VkSwapchainCreateInfoKHR swapchain_info = {
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain_info.surface = surface_;
  swapchain_info.minImageCount = num_frames_;
  swapchain_info.imageFormat = format_;
  swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  swapchain_info.imageExtent = surface_capabilities.currentExtent;
  swapchain_info.imageArrayLayers = 1;
  swapchain_info.imageUsage = usage_;
  swapchain_info.preTransform = surface_capabilities.currentTransform;
  swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_info.presentMode = present_mode_;
  swapchain_info.clipped = VK_TRUE;
  vkCreateSwapchainKHR(context.device(), &swapchain_info, NULL, &swapchain_);
  width_ = swapchain_info.imageExtent.width;
  height_ = swapchain_info.imageExtent.height;

  uint32_t image_count = 0;
  vkGetSwapchainImagesKHR(context.device(), swapchain_, &image_count, NULL);
  images_.resize(image_count);
  vkGetSwapchainImagesKHR(context.device(), swapchain_, &image_count,
                          images_.data());

  
  image_views_.resize(num_views_);
  for (uint32_t view = 0; view < num_views_; view++) {
    image_views_[view].resize(image_count);
    for (uint32_t i = 0; i < image_count; i++) {
      VkImageViewCreateInfo image_view_info = {
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      image_view_info.image = images_[i];
      image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      image_view_info.format = swapchain_info.imageFormat;
      image_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};
      vkCreateImageView(context.device(), &image_view_info, NULL,
                        &image_views_[view][i]);
    }
  }
}

SwapchainDesktop::~SwapchainDesktop() {
  for (uint32_t view = 0; view < num_views_; view++) {
    for (uint32_t i = 0; i < image_views_[view].size(); i++) {
      vkDestroyImageView(context_.device(), image_views_[view][i], NULL);
    }
  }

  vkDestroySwapchainKHR(context_.device(), swapchain_, NULL);
  vkDestroySurfaceKHR(context_.instance(), surface_, NULL);
}

bool SwapchainDesktop::AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index) {
  VkResult result = vkAcquireNextImageKHR(context_.device(), swapchain_, UINT64_MAX,
                              semaphore, NULL, image_index);
  
  switch (result) {
    case VK_SUCCESS:
      return true;

    case VK_SUBOPTIMAL_KHR:
      should_recreate_ = true;
      return true;

    case VK_ERROR_OUT_OF_DATE_KHR:
      should_recreate_ = true;
      return false;

    default:
      return false;
  }
}

void SwapchainDesktop::ReleaseImage() {}

void SwapchainDesktop::Recreate() {
  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context_.physical_device(),
                                            surface_, &surface_capabilities);

  VkSwapchainCreateInfoKHR swapchain_info = {
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain_info.surface = surface_;
  swapchain_info.minImageCount = num_frames_;
  swapchain_info.imageFormat = format_;
  swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  swapchain_info.imageExtent = surface_capabilities.currentExtent;
  swapchain_info.imageArrayLayers = 1;
  swapchain_info.imageUsage = usage_;
  swapchain_info.preTransform = surface_capabilities.currentTransform;
  swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_info.presentMode = present_mode_;
  swapchain_info.clipped = VK_TRUE;
  swapchain_info.oldSwapchain = swapchain_;

  VkSwapchainKHR new_swapchain;
  vkCreateSwapchainKHR(context_.device(), &swapchain_info, NULL,
                       &new_swapchain);
  vkDestroySwapchainKHR(context_.device(), swapchain_, NULL);
  swapchain_ = new_swapchain;

  width_ = swapchain_info.imageExtent.width;
  height_ = swapchain_info.imageExtent.height;

  uint32_t image_count = 0;
  vkGetSwapchainImagesKHR(context_.device(), swapchain_, &image_count, NULL);
  images_.resize(image_count);
  vkGetSwapchainImagesKHR(context_.device(), swapchain_, &image_count,
                          images_.data());

  for (uint32_t view = 0; view < num_views_; view++) {
    for (uint32_t i = 0; i < image_views_[view].size(); i++) {
      vkDestroyImageView(context_.device(), image_views_[view][i], NULL);
    }
  }

  image_views_.resize(num_views_);
  for (uint32_t view = 0; view < num_views_; view++) {
    image_views_[view].resize(image_count);
    for (uint32_t i = 0; i < image_count; i++) {
      VkImageViewCreateInfo image_view_info = {
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      image_view_info.image = images_[i];
      image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      image_view_info.format = swapchain_info.imageFormat;
      image_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                          1};
      vkCreateImageView(context_.device(), &image_view_info, NULL,
                        &image_views_[view][i]);
    }
  }

  should_recreate_ = false;
}

}  // namespace vk
}  // namespace vkgs
