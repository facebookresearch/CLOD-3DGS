// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_XR_MANAGER_H
#define VKGS_ENGINE_VULKAN_XR_MANAGER_H

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "core/string.h"
#include "vkgs/engine/vulkan/xr_input_manager.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

namespace vkgs {
class Engine;
namespace vk {

/**
 * @brief XR event state
 */
struct XREventState {
  bool terminate_;
  bool running_;
};


/**
 * @brief XR manager
 */
class XRManager {
 public:
  XRManager();
  ~XRManager();

  std::vector<std::string> get_instance_extensions();
  std::vector<std::string> get_device_extensions(VkInstance& vk_instance);

  void start_session(VkInstance& vk_instance,
                     VkPhysicalDevice& vk_physical_device, VkDevice& vk_device,
                     uint32_t& queue_family_index);
  
  XrInstance& instance() { return xr_instance_; }
  XrSystemId& system() { return xr_system_; }
  XrSession& session() { return xr_session_; }

  XREventState poll_events();
  void poll_actions(Engine& engine);

  void begin_frame();
  void end_frame(XrSwapchain& swapchain, uint32_t width, uint32_t height);

  static void log_event(const char* message);

  void create_xr_validation_layer();

  static XrBool32 report_error(
      XrDebugUtilsMessageSeverityFlagsEXT severity,
      XrDebugUtilsMessageTypeFlagsEXT type,
      const XrDebugUtilsMessengerCallbackDataEXT* callback_data,
      void* user_data);

  glm::mat4 view_matrix(uint32_t view_index);
  const std::vector<XrView>& views() { return views_; };
  XrFrameState frame_state() { return frame_state_; };
  XrSpace space() { return xr_space_; };
  const bool& eye_tracker_enabled() { return eye_tracker_enabled_; };

 private:	 
  XrInstance xr_instance_ = nullptr;
  XrSystemId xr_system_ = 0;
  XrSession xr_session_ = nullptr;
  XrSpace xr_space_ = nullptr;
  XrDebugUtilsMessengerEXT xr_debug_messenger_;

  XrActionSet xr_action_set_ = nullptr;
  XrAction xr_action_ = nullptr;
  XrSpace xr_action_space_ = nullptr;
  XrSpace xr_reference_space_ = nullptr;

  XREventState event_state_{};
  XrFrameState frame_state_{};
  static const uint32_t num_views_ = 2;
  std::vector<XrView> views_;
  bool eye_tracker_enabled_ = false;

  XRInputManager* xr_input_manager_ = nullptr;
  glm::vec3 translation_offset_ = glm::vec3(0.0);
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_XR_MANAGER_H