// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_SWAPCHAIN_VR_H
#define VKGS_ENGINE_VULKAN_SWAPCHAIN_VR_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/image_spec.h"
#include "vkgs/engine/vulkan/swapchain.h"
#include "vkgs/engine/vulkan/xr_manager.h"

namespace vkgs {
namespace vk {

// adapted from:
// https://amini-allight.org/post/openxr-tutorial-part-0
// https://gitlab.com/amini-allight/openxr-tutorial

/**
 * @brief Swapchain for VR viewer
 */  
class SwapchainVR : public Swapchain {
 public:
  SwapchainVR() = delete;
  SwapchainVR(Context context, std::shared_ptr<XRManager> xr_manager, bool vsync = true);
  ~SwapchainVR();

  virtual bool AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index);
  virtual void ReleaseImage();

  virtual void Recreate();

  XrSwapchain xr_swapchain() const noexcept { return xr_swapchain_; }

 protected:
  std::shared_ptr<XRManager> xr_manager_;
  XrSwapchain xr_swapchain_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_SWAPCHAIN_VR_H
