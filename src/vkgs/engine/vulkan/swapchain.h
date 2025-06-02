// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_SWAPCHAIN_H
#define VKGS_ENGINE_VULKAN_SWAPCHAIN_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/image_spec.h"

namespace vkgs {
namespace vk {


/**
 * @brief Swapchain abstract class
 */
class Swapchain {
 public:
  Swapchain() = delete;
  Swapchain(Context context, bool vsync = true);
  ~Swapchain() = default;

  virtual bool AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index) = 0;
  virtual void ReleaseImage() = 0;

  VkSwapchainKHR swapchain() const noexcept { return swapchain_; }
  uint32_t width() const noexcept { return width_; }
  uint32_t height() const noexcept { return height_; }
  VkImageUsageFlags usage() const noexcept { return usage_; }
  VkFormat format() const noexcept { return format_; }
  uint32_t image_count() const noexcept { return images_.size(); }
  VkImage image(uint32_t index) const { return images_[index]; }
  VkImageView image_view(uint32_t view_index, uint32_t frame_index) const { return image_views_[view_index][frame_index]; }

  ImageSpec image_spec() const noexcept;

  void SetVsync(bool flag = true);
  bool ShouldRecreate() const;
  bool ShouldRecreate(bool flag);
  virtual void Recreate() = 0;


 protected:   
  Context context_;
  VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  VkImageUsageFlags usage_ = 0;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t num_views_ = 0;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;
  std::vector<std::vector<VkImageView>> image_views_;
  bool should_recreate_ = false;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_SWAPCHAIN_H
