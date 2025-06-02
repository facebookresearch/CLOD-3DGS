// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/swapchain.h"

namespace vkgs {
namespace vk {

Swapchain::Swapchain(Context context, bool vsync)
    : context_(context) {}

ImageSpec Swapchain::image_spec() const noexcept {
  return ImageSpec{width_, height_, usage_, format_};
}

void Swapchain::SetVsync(bool flag) {
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  if (flag) {
    present_mode = VK_PRESENT_MODE_FIFO_KHR;
  } else {
    present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
  }

  if (present_mode_ != present_mode) {
    present_mode_ = present_mode;
    should_recreate_ = true;
  }
}

bool Swapchain::ShouldRecreate() const { return should_recreate_; }

bool Swapchain::ShouldRecreate(bool flag) {
  should_recreate_ = flag;
  return should_recreate_;
}

}  // namespace vk
}  // namespace vkgs
