// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/xr_manager.h"

#include "vkgs/engine/engine.h"


namespace vkgs {
namespace vk {
  
// adapted from:
// https://amini-allight.org/post/openxr-tutorial-part-0
// https://gitlab.com/amini-allight/openxr-tutorial

XRManager::XRManager() {
  // create instance
  {
    const char* const layer_names[] = {"XR_APILAYER_LUNARG_core_validation"};
    std::vector<const char*> extension_names = {
        "XR_KHR_vulkan_enable", "XR_KHR_vulkan_enable2", "XR_EXT_debug_utils"};

    // query all available extensions
    uint32_t property_count;
    xrEnumerateInstanceExtensionProperties(NULL, 0, &property_count, NULL);
    std::vector<XrExtensionProperties> extension_properties(property_count, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(
        NULL, property_count, &property_count, &extension_properties[0]);

    for (uint32_t i = 0; i < property_count; i++) {
      if (strcmp(extension_properties[i].extensionName, "XR_FB_eye_tracking_social") == 0) {
        extension_names.push_back("XR_FB_eye_tracking_social");
        eye_tracker_enabled_ = true;
      }
      printf("%s\n", extension_properties[i].extensionName);
    }

    // create instance
    XrInstanceCreateInfo instance_create_info{};
    instance_create_info.type = XR_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.createFlags = 0;
    strcpy(instance_create_info.applicationInfo.applicationName, "test");
    strcpy(instance_create_info.applicationInfo.engineName, "test");
    instance_create_info.applicationInfo.apiVersion = XR_API_VERSION_1_0;
    instance_create_info.applicationInfo.engineVersion =
        XR_MAKE_VERSION(0, 0, 1);
    instance_create_info.enabledApiLayerCount = 1;
    instance_create_info.enabledApiLayerNames = layer_names;
    instance_create_info.enabledExtensionCount = extension_names.size();
    instance_create_info.enabledExtensionNames = extension_names.data();

    auto result = xrCreateInstance(&instance_create_info, &xr_instance_);
  
    XrView view_template{};
    view_template.type = XR_TYPE_VIEW;
    for (uint32_t i = 0; i < num_views_; i++) {
      views_.push_back(view_template);
    }
  }

  create_xr_validation_layer();

  // get system
  {
    XrSystemGetInfo system_get_info{};
    system_get_info.type = XR_TYPE_SYSTEM_GET_INFO;
    system_get_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    auto result = xrGetSystem(xr_instance_, &system_get_info, &xr_system_);
  }

  frame_state_.type = XR_TYPE_FRAME_STATE;
}

std::vector<std::string> XRManager::get_instance_extensions() {
  {
    PFN_xrVoidFunction func;
    xrGetInstanceProcAddr(xr_instance_, "xrGetVulkanGraphicsRequirementsKHR",
                          &func);
    auto xrGetVulkanGraphicsRequirementsKHR =
        (PFN_xrGetVulkanGraphicsRequirementsKHR)func;

    XrGraphicsRequirementsVulkanKHR graphics_requirements{};
    graphics_requirements.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR;
    xrGetVulkanGraphicsRequirementsKHR(xr_instance_, xr_system_,
                                       &graphics_requirements);
  }
  {
    PFN_xrVoidFunction func;
    xrGetInstanceProcAddr(xr_instance_, "xrGetVulkanInstanceExtensionsKHR",
                          &func);
    auto xrGetVulkanInstanceExtensionsKHR =
        (PFN_xrGetVulkanInstanceExtensionsKHR)func;

    uint32_t instance_extensions_size;
    xrGetVulkanInstanceExtensionsKHR(xr_instance_, xr_system_, 0,
                                     &instance_extensions_size, nullptr);

    char* instance_extensions_data = new char[instance_extensions_size];
    xrGetVulkanInstanceExtensionsKHR(
        xr_instance_, xr_system_, instance_extensions_size,
        &instance_extensions_size, instance_extensions_data);
    auto instance_extensions = str::split(std::string(instance_extensions_data));
    delete[] instance_extensions_data;

    return instance_extensions;
  }
}

std::vector<std::string> XRManager::get_device_extensions(VkInstance& vk_instance) {
  {
    PFN_xrVoidFunction func;
    xrGetInstanceProcAddr(xr_instance_, "xrGetVulkanGraphicsDeviceKHR", &func);    
    auto xrGetVulkanGraphicsDeviceKHR = (PFN_xrGetVulkanGraphicsDeviceKHR)func;

    VkPhysicalDevice physical_device;
    xrGetVulkanGraphicsDeviceKHR(xr_instance_, xr_system_, vk_instance, &physical_device);
  }
  {
    PFN_xrVoidFunction func;
    xrGetInstanceProcAddr(xr_instance_, "xrGetVulkanDeviceExtensionsKHR", &func);
    auto xrGetVulkanDeviceExtensionsKHR = (PFN_xrGetVulkanDeviceExtensionsKHR)func;

    uint32_t device_extensions_size;
    xrGetVulkanDeviceExtensionsKHR(xr_instance_, xr_system_, 0, &device_extensions_size, nullptr);

    char* device_extensions_data = new char[device_extensions_size];
    xrGetVulkanDeviceExtensionsKHR(xr_instance_, xr_system_, device_extensions_size, &device_extensions_size, device_extensions_data);
    auto device_extensions = str::split(std::string(device_extensions_data));
    delete[] device_extensions_data;

    return device_extensions;
  }
};

void XRManager::start_session(
  VkInstance& vk_instance,
  VkPhysicalDevice& vk_physical_device,
  VkDevice& vk_device,
  uint32_t& queue_family_index
) {
  
  XrGraphicsBindingVulkanKHR graphics_binding{};
  graphics_binding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR;
  graphics_binding.instance = vk_instance;
  graphics_binding.physicalDevice = vk_physical_device;
  graphics_binding.device = vk_device;
  graphics_binding.queueFamilyIndex = queue_family_index;
  graphics_binding.queueIndex = 0;

  XrSessionCreateInfo session_create_info{};
  session_create_info.type = XR_TYPE_SESSION_CREATE_INFO;
  session_create_info.next = &graphics_binding;
  session_create_info.createFlags = 0;
  session_create_info.systemId = xr_system_;

  auto session_result = xrCreateSession(xr_instance_, &session_create_info, &xr_session_);

  event_state_.running_ = false;
  event_state_.terminate_ = false;

  XrReferenceSpaceCreateInfo space_create_info{};
  space_create_info.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
  space_create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
  space_create_info.poseInReferenceSpace = {{0, 0, 0, 1}, {0, 0, 0}};
  XrResult result =
      xrCreateReferenceSpace(xr_session_, &space_create_info, &xr_space_);

  if (result != VK_SUCCESS) {
    log_event("failed to create space");
  }
  xr_input_manager_ = new XRInputManager(xr_instance_, xr_session_);
}

XREventState XRManager::poll_events() {
  XrEventDataBuffer event_data{};
  event_data.type = XR_TYPE_EVENT_DATA_BUFFER;
  XrResult result = xrPollEvent(xr_instance_, &event_data);

  if (result == XR_EVENT_UNAVAILABLE) {
    if (event_state_.running_) {
      XrFrameWaitInfo frame_wait_info{};
      frame_wait_info.type = XR_TYPE_FRAME_WAIT_INFO;

      XrResult wait_frame_result = xrWaitFrame(xr_session_, &frame_wait_info, &frame_state_);
    }
  } else if (result != XR_SUCCESS) {
    printf("Error with OpenXR event polling");
  } else {
    switch (event_data.type) {
      default:
        log_event("unknown event error");
        break;
      case XR_TYPE_EVENT_DATA_EVENTS_LOST:
        log_event("events lost");
        break;
      case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
        log_event("instance lost pending");
        event_state_.terminate_ = true;
        break;
      case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
        log_event("interaction profile changed");
        break;
      case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
        log_event("referenced space change pending");
        break;
      case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
        auto event = (XrEventDataSessionStateChanged*)&event_data;
        switch (event->state){
          case XR_SESSION_STATE_UNKNOWN:
          case XR_SESSION_STATE_MAX_ENUM:
            log_event("changed to unknown state");
            break;
          case XR_SESSION_STATE_IDLE:
            event_state_.running_ = false;
            break;
          case XR_SESSION_STATE_READY: {
            XrSessionBeginInfo session_begin_info{};
            session_begin_info.type = XR_TYPE_SESSION_BEGIN_INFO;
            session_begin_info.primaryViewConfigurationType =
                XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

            xrBeginSession(xr_session_, &session_begin_info);
            event_state_.running_ = true;
            log_event("begin session");
            break;
          }
          case XR_SESSION_STATE_SYNCHRONIZED:
          case XR_SESSION_STATE_VISIBLE:
          case XR_SESSION_STATE_FOCUSED:
            event_state_.running_ = true;
            break;
          case XR_SESSION_STATE_STOPPING:
            event_state_.running_ = false;
            xrEndSession(xr_session_);
            log_event("end session");
            break;
          case XR_SESSION_STATE_LOSS_PENDING:
            event_state_.terminate_ = true;
            log_event("loss pending");
            break;
          case XR_SESSION_STATE_EXITING:
            event_state_.terminate_ = true;
            log_event("exiting");
            break;
        }
        break;
      }
    }
  }

  return event_state_;
}

void XRManager::begin_frame() {
  XrFrameBeginInfo frame_begin_info{};
  frame_begin_info.type = XR_TYPE_FRAME_BEGIN_INFO;
  xrBeginFrame(xr_session_, &frame_begin_info);

  XrViewLocateInfo view_locate_info{};
  view_locate_info.type = XR_TYPE_VIEW_LOCATE_INFO;
  view_locate_info.viewConfigurationType =
      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  view_locate_info.displayTime = frame_state_.predictedDisplayTime;
  view_locate_info.space = xr_space_;

  XrViewState view_state{};
  view_state.type = XR_TYPE_VIEW_STATE;

  uint32_t num_views = num_views_;
  xrLocateViews(xr_session_, &view_locate_info, &view_state, num_views,
                &num_views, views_.data());
}

void XRManager::end_frame(XrSwapchain& xr_swapchain, uint32_t width, uint32_t height) {
  XrCompositionLayerProjectionView projection_views[num_views_]{};
  for (uint32_t i = 0; i < num_views_; i++) {
    projection_views[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
    projection_views[i].pose = views_[i].pose;
    projection_views[i].fov = views_[i].fov;
    projection_views[i].subImage = {
        xr_swapchain,
        {{0, 0}, {(int32_t)width, (int32_t)height}},
        i};
  }

  XrCompositionLayerProjection composition_layer_projection{};
  composition_layer_projection.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION;
  composition_layer_projection.space = xr_space_;
  composition_layer_projection.viewCount = num_views_;
  composition_layer_projection.views = projection_views;

  auto p_layer =
      (const XrCompositionLayerBaseHeader*)&composition_layer_projection;
  
  XrFrameEndInfo frame_end_info{};
  frame_end_info.type = XR_TYPE_FRAME_END_INFO;
  frame_end_info.displayTime = frame_state_.predictedDisplayTime;
  frame_end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  frame_end_info.layerCount = 1;
  frame_end_info.layers = &p_layer;

  xrEndFrame(xr_session_, &frame_end_info);
}

void XRManager::log_event(const char* message) {
  auto output = std::string("XRManager: ") + std::string(message) + std::string("\n");
  printf(output.c_str());
}

void XRManager::create_xr_validation_layer() {
  XrDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info{};
  debug_messenger_create_info.type =
      XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  debug_messenger_create_info.messageSeverities =
      XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debug_messenger_create_info.messageTypes =
      XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
      XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
      XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;
  debug_messenger_create_info.userCallback = XRManager::report_error;
  debug_messenger_create_info.userData = nullptr;

  PFN_xrVoidFunction func;
  xrGetInstanceProcAddr(xr_instance_, "xrCreateDebugUtilsMessengerEXT", &func);
  auto xrCreateDebugUtilsMessengerEXT =
      (PFN_xrCreateDebugUtilsMessengerEXT)func;
  xrCreateDebugUtilsMessengerEXT(xr_instance_, &debug_messenger_create_info,
                                 &xr_debug_messenger_);
}

void XRManager::poll_actions(Engine& engine) {
  float move_speed = 0.01;
  auto input_state = xr_input_manager_->pollActions(xr_session_);
  translation_offset_.x += input_state.move_x * move_speed;
  translation_offset_.z += input_state.move_y * move_speed;

  auto last_index = engine.config_.num_levels() - 1;
  if (input_state.lod_down) {
    engine.lod_levels_[last_index] = std::clamp(engine.lod_levels_[last_index] - 0.01f, 0.01f, 1.0f);
  }
  if (input_state.lod_up) {
    engine.lod_levels_[last_index] = std::clamp(engine.lod_levels_[last_index] + 0.01f, 0.01f, 1.0f);
  }
  if (input_state.res_down) {
    engine.res_scales_[last_index] = std::clamp(engine.res_scales_[last_index] - 0.01f, 0.1f, 1.0f);
  }
  if (input_state.res_up) {
    engine.res_scales_[last_index] = std::clamp(engine.res_scales_[last_index] + 0.01f, 0.1f, 1.0f);
  }

}

XrBool32 XRManager::report_error(
    XrDebugUtilsMessageSeverityFlagsEXT severity,
    XrDebugUtilsMessageTypeFlagsEXT type,
    const XrDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
  std::cout << callback_data->message << std::endl;
  return XR_FALSE;
}

glm::mat4 XRManager::view_matrix(uint32_t view_index) {
  // OpenXR uses right-handed coordinates
  auto view = views_[view_index];
  glm::vec3 pos(view.pose.position.x, view.pose.position.y,
                view.pose.position.z);
  glm::quat quat(view.pose.orientation.w, view.pose.orientation.x,
                 view.pose.orientation.y, view.pose.orientation.z);
  pos += translation_offset_;
  glm::mat4 matrix = glm::translate(glm::mat4(1.0), pos) * glm::mat4_cast(quat);  

  return glm::inverse(matrix);
}

XRManager::~XRManager() {
  delete xr_input_manager_;
  xrDestroySpace(xr_space_);
  xrDestroyInstance(xr_instance_);
  xrDestroySession(xr_session_);
}

};  // namespace vk
};  // namespace vkgs
