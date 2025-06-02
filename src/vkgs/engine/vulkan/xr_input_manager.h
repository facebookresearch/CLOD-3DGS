// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_XR_INPUT_MANAGER_H
#define VKGS_ENGINE_VULKAN_XR_INPUT_MANAGER_H

#include <map>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>


namespace vkgs {
namespace vk {

/**
  * @brief XR input state (from controller)
  */
struct XRInputState {
  float move_x;
  float move_y;
  bool lod_up;
  bool lod_down;
  bool res_up;
  bool res_down;
};

/**
 * @brief XR input manager
 * Processes inputs
 */
class XRInputManager {
 public:
  XRInputManager(XrInstance xr_instance, XrSession xr_session);
  ~XRInputManager();

  void addAction(
    XrInstance xr_instance,
    const char* name,
    XrActionType action_type,
    const char* binding_path);
  XRInputState pollActions(XrSession xr_session);
  float getActionFloat(XrSession xr_session, XrAction xr_action);
  bool getActionBoolean(XrSession xr_session, XrAction xr_action);

private:
  XrActionSet xr_action_set_;
  std::map<std::string, XrActionSuggestedBinding> action_bindings_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_XR_INPUT_MANAGER_H