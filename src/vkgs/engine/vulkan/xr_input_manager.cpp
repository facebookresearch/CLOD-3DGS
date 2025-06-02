// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/xr_input_manager.h"

namespace vkgs {
namespace vk {
  
// adapted from:
// https://amini-allight.org/post/openxr-tutorial-part-0
// https://gitlab.com/amini-allight/openxr-tutorial

XRInputManager::XRInputManager(
  XrInstance xr_instance,
  XrSession xr_session
) {
  {
    XrActionSetCreateInfo action_set_info{};
    action_set_info.type = XR_TYPE_ACTION_SET_CREATE_INFO;
    strcpy(action_set_info.actionSetName, "controller_inputs");
    strcpy(action_set_info.localizedActionSetName, "control inputs");
    action_set_info.priority = 0;

    auto result = xrCreateActionSet(xr_instance, &action_set_info, &xr_action_set_);
  }

  {
    // actions
    addAction(xr_instance, "move_x", XrActionType::XR_ACTION_TYPE_FLOAT_INPUT,
              "/user/hand/left/input/thumbstick/x");
    addAction(xr_instance, "move_y", XrActionType::XR_ACTION_TYPE_FLOAT_INPUT,
              "/user/hand/left/input/thumbstick/y");
    addAction(xr_instance, "lod_up", XrActionType::XR_ACTION_TYPE_BOOLEAN_INPUT,
              "/user/hand/left/input/y/click");
    addAction(xr_instance, "lod_down", XrActionType::XR_ACTION_TYPE_BOOLEAN_INPUT,
              "/user/hand/left/input/x/click");
    addAction(xr_instance, "res_up", XrActionType::XR_ACTION_TYPE_BOOLEAN_INPUT,
              "/user/hand/right/input/b/click");
    addAction(xr_instance, "res_down", XrActionType::XR_ACTION_TYPE_BOOLEAN_INPUT,
              "/user/hand/right/input/a/click");

    // action bindings
    std::vector<XrActionSuggestedBinding> suggested_bindings;
    for (auto const& action_binding : action_bindings_) {
      suggested_bindings.push_back(action_binding.second);
    };
    
    XrPath profile_path;
    xrStringToPath(xr_instance, "/interaction_profiles/oculus/touch_controller", &profile_path);
    
    XrInteractionProfileSuggestedBinding suggested_bindings_info{};
    suggested_bindings_info.type = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING;
    suggested_bindings_info.interactionProfile = profile_path;
    suggested_bindings_info.countSuggestedBindings = suggested_bindings.size();
    suggested_bindings_info.suggestedBindings = &suggested_bindings[0];

    auto result = xrSuggestInteractionProfileBindings(xr_instance, &suggested_bindings_info);
    int x = 0;
  }

  XrSessionActionSetsAttachInfo action_sets_attach_info{};
  action_sets_attach_info.type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO;
  action_sets_attach_info.countActionSets = 1;
  action_sets_attach_info.actionSets = &xr_action_set_;

  auto result = xrAttachSessionActionSets(xr_session, &action_sets_attach_info);
}

void XRInputManager::addAction(
  XrInstance xr_instance,
  const char* name,
  XrActionType action_type,
  const char* binding_path
) {
  XrAction xr_action;

  XrActionCreateInfo action_info{};
  action_info.type = XR_TYPE_ACTION_CREATE_INFO;
  strcpy(action_info.actionName, name);
  strcpy(action_info.localizedActionName, name);
  action_info.actionType = action_type;

  auto result = xrCreateAction(xr_action_set_, &action_info, &xr_action);

  XrPath path;
  xrStringToPath(xr_instance, binding_path, &path);

  XrActionSuggestedBinding action_binding{};
  action_binding.action = xr_action;
  action_binding.binding = path;

  action_bindings_.insert(std::make_pair(std::string(name), action_binding));
}

XRInputState XRInputManager::pollActions(XrSession xr_session) {
  XrActiveActionSet active_action_set{};
  active_action_set.actionSet = xr_action_set_;
  active_action_set.subactionPath = XR_NULL_PATH;

  XrActionsSyncInfo sync_info{};
  sync_info.type = XR_TYPE_ACTIONS_SYNC_INFO;
  sync_info.countActiveActionSets = 1;
  sync_info.activeActionSets = &active_action_set;

  xrSyncActions(xr_session, &sync_info);

  XRInputState input_state;
  input_state.move_x = getActionFloat(xr_session, action_bindings_[std::string("move_x")].action);
  input_state.move_y = getActionFloat(xr_session, action_bindings_[std::string("move_y")].action);
  input_state.lod_up = getActionBoolean(xr_session, action_bindings_[std::string("lod_up")].action);
  input_state.lod_down = getActionBoolean(xr_session, action_bindings_[std::string("lod_down")].action);
  input_state.res_up = getActionBoolean(xr_session, action_bindings_[std::string("res_up")].action);
  input_state.res_down = getActionBoolean(xr_session, action_bindings_[std::string("res_down")].action);

  return input_state;
}

float XRInputManager::getActionFloat(XrSession xr_session, XrAction xr_action) {
  XrActionStateGetInfo info{};
  info.type = XR_TYPE_ACTION_STATE_GET_INFO;
  info.action = xr_action;

  XrActionStateFloat action_state{};
  action_state.type = XR_TYPE_ACTION_STATE_FLOAT;

  auto result = xrGetActionStateFloat(xr_session, &info, &action_state);
  return action_state.currentState;
}

bool XRInputManager::getActionBoolean(XrSession xr_session, XrAction xr_action) {
  XrActionStateGetInfo info{};
  info.type = XR_TYPE_ACTION_STATE_GET_INFO;
  info.action = xr_action;

  XrActionStateBoolean action_state{};
  action_state.type = XR_TYPE_ACTION_STATE_BOOLEAN;

  auto result = xrGetActionStateBoolean(xr_session, &info, &action_state);
  return action_state.currentState;
}

XRInputManager::~XRInputManager() {
  for (auto const& xr_action : action_bindings_) {
    xrDestroyAction(xr_action.second.action);
  }
  xrDestroyActionSet(xr_action_set_);
}

};  // namespace vk
};  // namespace vkgs
