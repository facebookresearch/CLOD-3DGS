// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_SWAPCHAIN_DESKTOP_H
#define VKGS_ENGINE_VULKAN_SWAPCHAIN_DESKTOP_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/image_spec.h"
#include "vkgs/engine/vulkan/swapchain.h"

namespace vkgs {
namespace vk {


/**
 * @brief Swapchain for desktop viewer
 */
class SwapchainDesktop : public Swapchain {
 public:
  SwapchainDesktop() = delete;
  SwapchainDesktop(Context context, VkSurfaceKHR surface, uint32_t num_frames, bool vsync = true);
  ~SwapchainDesktop();

  virtual bool AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index);
  virtual void ReleaseImage();

  virtual void Recreate();

 protected:
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  uint32_t num_frames_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_SWAPCHAIN_DESKTOP_H
