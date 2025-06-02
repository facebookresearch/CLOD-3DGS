// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_CONTEXT_H
#define VKGS_ENGINE_VULKAN_CONTEXT_H

#include <memory>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include "vk_mem_alloc.h"

#include "core/string.h"
#include "vkgs/engine/vulkan/xr_manager.h"

#include "eye_tracker/eye_tracker.h"
#include "eye_tracker/eye_tracker_openxr.h"


namespace vkgs {
namespace vk {


/**
 * @brief Context for Engine
 */
class Context {
 public:
  Context();

  explicit Context(bool vr_mode, bool enable_validation);

  ~Context();

  const std::string& device_name() const;
  VkInstance instance() const;
  std::shared_ptr<XRManager> xr_manager() const;
  VkPhysicalDevice physical_device() const;
  VkDevice device() const;
  uint32_t graphics_queue_family_index() const;
  uint32_t transfer_queue_family_index() const;
  VkQueue graphics_queue() const;
  VkQueue transfer_queue() const;
  VmaAllocator allocator() const;
  VkCommandPool command_pool() const;
  VkDescriptorPool descriptor_pool() const;
  VkPipelineCache pipeline_cache() const;

  bool geometry_shader_available() const;

  VkResult GetMemoryFdKHR(const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd);
  VkResult GetSemaphoreFdKHR(const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
                             int* pFd);

#ifdef _WIN32
  VkResult Context::GetMemoryWin32HandleKHR(
      const VkMemoryGetWin32HandleInfoKHR* pGetFdInfo, HANDLE* handle);
  VkResult Context::GetSemaphoreWin32HandleKHR(
      const VkSemaphoreGetWin32HandleInfoKHR* pGetFdInfo, HANDLE* handle);
#endif

  bool enable_eye_tracker() const;
  std::shared_ptr<eye_tracker::EyeTracker> eye_tracker() const;
  void updateEyeTracking();
  glm::vec2 getEyePosition(uint32_t view_index, glm::mat4& view_matrix,
                           glm::mat4& projection_matrix);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_CONTEXT_H
