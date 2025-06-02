// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/gui.h"

#include "vkgs/engine/engine.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "implot.h"

namespace vkgs {

static void check_vk_result(VkResult err) {
  if (err == 0) return;
  std::cerr << "[imgui vulkan] Error: VkResult = " << err << std::endl;
  if (err < 0) abort();
}

static void AddTooltip(const char* message) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(message);
  }
}

static void AddTooltip(const std::string& message) {
  AddTooltip(message.c_str());
}

GUI::GUI() = default;
    
GUI::~GUI() = default;

void GUI::prepare() {
  // Setup Dear ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGui::StyleColorsDark();
}

void GUI::initialize(Engine& engine) {
  ImGui_ImplGlfw_InitForVulkan(engine.window_, true);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = engine.context_.instance();
  init_info.PhysicalDevice = engine.context_.physical_device();
  init_info.Device = engine.context_.device();
  init_info.QueueFamily = engine.context_.graphics_queue_family_index();
  init_info.Queue = engine.context_.graphics_queue();
  init_info.PipelineCache = engine.context_.pipeline_cache();
  init_info.DescriptorPool = engine.context_.descriptor_pool();
  init_info.Subpass = 0;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.RenderPass = engine.render_pass_compositor_;
  init_info.MSAASamples = engine.samples_;
  init_info.Allocator = VK_NULL_HANDLE;
  init_info.CheckVkResultFn = check_vk_result;
  ImGui_ImplVulkan_Init(&init_info);

  ImGuiIO& io = ImGui::GetIO();

  if (engine.lazy_budget_mode_) {
    budget_mode_ = 1;
  }
}

GUIInfo GUI::update(Engine& engine, uint32_t frame_index) {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  static glm::vec3 lt(0.f);
  static glm::vec3 gt(0.f);
  static glm::vec3 lr(0.f);
  static glm::quat lq;
  static glm::vec3 gr(0.f);
  static glm::quat gq;
  static float scale = 1.f;
  glm::mat4 model(1.f);
  
  const auto& io = ImGui::GetIO();
  
  auto camera = std::dynamic_pointer_cast<CameraLookAt>(engine.camera_);

  // handle events
  if (!demo_mode_ && !engine.recorder_mode_ && !engine.benchmark_mode_) {
    // mouse input
    if (!io.WantCaptureMouse) {
      bool left = io.MouseDown[ImGuiMouseButton_Left];
      bool right = io.MouseDown[ImGuiMouseButton_Right];
      float dx = io.MouseDelta.x;
      float dy = io.MouseDelta.y;

      if (left && !right) {
        camera->Rotate(dx, dy);
      } else if (!left && right) {
        camera->Translate(dx, dy);
      } else if (left && right) {
        camera->Zoom(dy);
      }

      if (io.MouseWheel != 0.f) {
        if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
          camera->DollyZoom(io.MouseWheel);
        } else {
          camera->Zoom(io.MouseWheel * 10.f);
        }
      }
    }
    if (!io.WantCaptureKeyboard) {
      constexpr float speed = 1000.f;
      float dt = io.DeltaTime;
      if (ImGui::IsKeyDown(ImGuiKey_W)) {
        camera->Translate(0.f, 0.f, speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_S)) {
        camera->Translate(0.f, 0.f, -speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_A)) {
        camera->Translate(speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_D)) {
        camera->Translate(-speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_Space)) {
        camera->Translate(0.f, speed * dt);
      }
    }
  } else {
    camera->Rotate((2.0 * 100.0 * glm::pi<float>()) / 200.0, 0.0);
  }

  // change to display per layer
  if (ImGui::Begin("vkgs")) {
    ImGui::SetWindowFontScale(font_size_);

    ImGui::Text("%s", engine.context_.device_name().c_str());
    ImGui::Text("%d total splats",
                engine.frame_infos_[0][0][frame_index].total_point_count);
    ImGui::Text("%d loaded splats",
                engine.frame_infos_[0][0][frame_index].loaded_point_count);
    ImGui::Text("res: %d x %d", engine.width_, engine.height_);

    auto loading_progress =
        engine.frame_infos_[0][0][frame_index].total_point_count > 0
            ? static_cast<float>(
                  engine.frame_infos_[0][0][frame_index].loaded_point_count) /
                  engine.frame_infos_[0][0][frame_index].total_point_count
            : 1.f;
    ImGui::Text("loading:");
    ImGui::SameLine();
    ImGui::ProgressBar(loading_progress, ImVec2(-1.f, 16.f));
    if (ImGui::Button("cancel")) {
      for (int view = 0; view < engine.splat_load_thread_.size(); view++) {
        for (int frame = 0; frame < engine.splat_load_thread_.size(); frame++) {
          engine.splat_load_thread_[view][frame].Cancel();
        }
      }
    }

    const auto* visible_point_count_buffer = reinterpret_cast<const uint32_t*>(
        engine.visible_point_count_cpu_buffer_[0][0].data());
    uint32_t visible_point_count = visible_point_count_buffer[frame_index];
    float visible_points_ratio =
        engine.frame_infos_[0][0][frame_index].loaded_point_count > 0
            ? static_cast<float>(visible_point_count) /
                  engine.frame_infos_[0][0][frame_index].loaded_point_count * 100.f
            : 0.f;

    ImGui::Text("%d (%.2f%%) visible splats", visible_point_count,
                visible_points_ratio);

    auto e2e_time =
        static_cast<double>(
            engine.frame_infos_[0][0][frame_index].end_to_end_time) /
        1e6;
    ImGui::Text("size      : %dx%d", engine.swapchain_->width(), engine.swapchain_->height());
    ImGui::Text("fps       : %7.3f", io.Framerate);
    ImGui::Text("            %7.3fms", 1e3 / io.Framerate);
    ImGui::Text("frame e2e : %7.3fms", e2e_time);
    const char* button_text = performance_window_ ? "Hide Graph" : "Show Graph";
    if (ImGui::Button(button_text)) {
      performance_window_ = !performance_window_;
    }
    
    for (int level = 0; level < engine.config_.num_levels(); level++) {
      if (ImGui::CollapsingHeader(("Level: " + std::to_string(level)).c_str())) {
        uint64_t total_time = engine.frame_infos_[0][level][frame_index].rank_time +
                              engine.frame_infos_[0][level][frame_index].sort_time +
                              engine.frame_infos_[0][level][frame_index].inverse_time +
                              engine.frame_infos_[0][level][frame_index].projection_time +
                              engine.frame_infos_[0][level][frame_index].rendering_time;
        ImGui::Text("total     : %7.3fms", static_cast<double>(total_time) / 1e6);
        ImGui::Text(
            "rank      : %7.3fms (%5.2f%%)",
            static_cast<double>(engine.frame_infos_[0][level][frame_index].rank_time) / 1e6,
            static_cast<double>(engine.frame_infos_[0][level][frame_index].rank_time) /
                total_time * 100.);
        ImGui::Text(
            "sort      : %7.3fms (%5.2f%%)",
            static_cast<double>(engine.frame_infos_[0][level][frame_index].sort_time) / 1e6,
            static_cast<double>(engine.frame_infos_[0][level][frame_index].sort_time) /
                total_time * 100.);
        ImGui::Text(
            "inverse   : %7.3fms (%5.2f%%)",
            static_cast<double>(engine.frame_infos_[0][level][frame_index].inverse_time) /
                1e6,
            static_cast<double>(engine.frame_infos_[0][level][frame_index].inverse_time) /
                total_time * 100.);
        ImGui::Text("projection: %7.3fms (%5.2f%%)",
                    static_cast<double>(
                        engine.frame_infos_[0][level][frame_index].projection_time) /
                        1e6,
                    static_cast<double>(
                        engine.frame_infos_[0][level][frame_index].projection_time) /
                        total_time * 100.);
        ImGui::Text(
            "rendering : %7.3fms (%5.2f%%)",
            static_cast<double>(engine.frame_infos_[0][level][frame_index].rendering_time) /
                1e6,
            static_cast<double>(engine.frame_infos_[0][level][frame_index].rendering_time) /
                total_time * 100.);
      }
    }

    ImGui::Text(
        "present   : %7.3fms",
        static_cast<double>(engine.frame_infos_[0][0][frame_index].present_done_timestamp -
                            engine.frame_infos_[0][0][frame_index].present_timestamp) /
            1e6);

    static int vsync = 0;
    ImGui::Text("Vsync");
    ImGui::SameLine();
    ImGui::RadioButton("on", &vsync, 1);
    AddTooltip("turn on Vsync");
    ImGui::SameLine();
    ImGui::RadioButton("off", &vsync, 0);
    AddTooltip("turn off Vsync");

    if (vsync)
      engine.swapchain_->SetVsync(true);
    else
      engine.swapchain_->SetVsync(false);

    ImGui::Checkbox("Axis", &show_axis_);
    AddTooltip("visualize axis");
    ImGui::SameLine();
    ImGui::Checkbox("Grid", &show_grid_);
    AddTooltip("visualize grid");
    ImGui::SameLine();
    if (engine.view_frustum_meshes_.size() > 0) {
      ImGui::Checkbox("View", &show_views_);
      ImGui::SameLine();
      AddTooltip("visualize view frustums of dataset");
    }
    ImGui::Checkbox("Center", &show_center_);
    AddTooltip("use mouse to move gaze point");

    // other options
    ImGui::Separator();
    ImGui::RadioButton("None", &budget_mode_, 0);
    AddTooltip("disable budget-based rendering");
    ImGui::SameLine();
    ImGui::RadioButton("Budget", &budget_mode_, 1);
    AddTooltip("enable budget-based rendering");
    
    ImGui::InputFloat("Budget (ms)", & engine.time_budget_);

    if (budget_mode_ == 0) {
      engine.lazy_budget_mode_ = false;
    } else if (budget_mode_ == 1) {
      engine.lazy_budget_mode_ = true;
    }

    float fov_degree = glm::degrees(camera->fov());
    ImGui::SliderFloat("Fov Y", &fov_degree, glm::degrees(camera->min_fov()),
                       glm::degrees(camera->max_fov()));
    AddTooltip("vertical field-of-view (degrees)");
    camera->SetFov(glm::radians(fov_degree));

    ImGui::Text("LOD");
    for (int level = 0; level < engine.config_.num_levels(); level++) {
      auto lod_label = std::string("LOD level ") + std::to_string(level);
      ImGui::SliderFloat(lod_label.c_str(), &engine.lod_levels_[level], 0.05, 1.0);
      AddTooltip(std::string("global LOD level for foveal level ") + std::to_string(level));
    }
    if (ImGui::CollapsingHeader("LOD Params")) {
      for (int level = 0; level < engine.config_.num_levels(); level++) {
        ImGui::Text((std::string("LOD Params ") + std::to_string(level)).c_str());
        ImGui::SliderFloat((std::string("min LOD ") + std::to_string(level)).c_str(),
                           &engine.lod_params_[level][0], 0.05, 1.0);
        AddTooltip(std::string("min LOD level (near distance)") + std::to_string(level));
        ImGui::SliderFloat((std::string("max LOD ") + std::to_string(level)).c_str(),
                           &engine.lod_params_[level][1], 0.05, 1.0);
        AddTooltip(std::string("max LOD level (far distance)") + std::to_string(level));
        ImGui::SliderFloat((std::string("min dist ") + std::to_string(level)).c_str(),
                           &engine.lod_params_[level][2], 0.05, 10.0);
        AddTooltip(std::string("min distance (meters)") + std::to_string(level));
        ImGui::SliderFloat((std::string("max dist ") + std::to_string(level)).c_str(),
                           &engine.lod_params_[level][3], 0.05, 10.0);
        AddTooltip(std::string("max distance (meters)") + std::to_string(level));
      }
    }
    ImGui::Text("Resolution");
    for (int level = 0; level < engine.config_.num_levels(); level++) {
      auto res_label = std::string("Res scale ") + std::to_string(level);
      ImGui::SliderFloat(res_label.c_str(), &engine.res_scales_[level], 0.01, 1.0);
    }

    ImGui::Text("Translation");
    ImGui::PushID("Translation");
    ImGui::DragFloat3("local", glm::value_ptr(lt), 0.01f);
    if (ImGui::IsItemDeactivated()) {
      engine.translation_ += glm::toMat3(engine.rotation_) * engine.scale_ * lt;
      lt = glm::vec3(0.f);
    }

    ImGui::DragFloat3("global", glm::value_ptr(gt), 0.01f);
    if (ImGui::IsItemDeactivated()) {
      engine.translation_ += gt;
      gt = glm::vec3(0.f);
    }
    ImGui::PopID();

    ImGui::Text("Rotation");
    ImGui::PushID("Rotation");
    ImGui::DragFloat3("local", glm::value_ptr(lr), 0.1f);
    lq = glm::quat(glm::radians(lr));
    if (ImGui::IsItemDeactivated()) {
      engine.rotation_ = engine.rotation_ * lq;
      lr = glm::vec3(0.f);
      lq = glm::quat(1.f, 0.f, 0.f, 0.f);
    }

    ImGui::DragFloat3("global", glm::value_ptr(gr), 0.1f);
    gq = glm::quat(glm::radians(gr));
    if (ImGui::IsItemDeactivated()) {
      engine.translation_ = gq * engine.translation_;
      engine.rotation_ = gq * engine.rotation_;
      gr = glm::vec3(0.f);
      gq = glm::quat(1.f, 0.f, 0.f, 0.f);
    }
    ImGui::PopID();

    ImGui::Text("Scale");
    ImGui::PushID("Scale");
    ImGui::DragFloat("local", &scale, 0.01f, 0.1f, 10.f, "%.3f",
                     ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemDeactivated()) {
      engine.scale_ *= scale;
      scale = 1.f;
    }
    ImGui::PopID();
    ImGui::Checkbox("Demo", &demo_mode_);
    AddTooltip(std::string("rotate camera around the center"));
    ImGui::SameLine();
    ImGui::Checkbox("Blending", &blending_mode_);
    AddTooltip(std::string("turns off layer blending for foveated rendering"));

    if (engine.config_.debug()) {
      const char* items[] = {"normal", "overdraw", "overdraw_alpha"};
      auto num_items = sizeof(items) / sizeof(*items);
      int selected_vis_mode = engine.vis_mode_;
      ImGui::Combo("Vis mode", &selected_vis_mode, items, num_items);
      engine.vis_mode_ = selected_vis_mode;
      ImGui::DragFloat("Vis scale", &engine.vis_mode_scale_, 1.0f, 1.0f, 450.0f,
                       "%.3f", ImGuiSliderFlags_Logarithmic);
    }

    ImGui::Separator();

    if (engine.benchmark_.isRecordingDone()) {
      engine.benchmark_mode_ = false;
    }
    ImGui::BeginDisabled(strlen(engine.recorder_path_) == 0);
    if (ImGui::Button("Benchmark")) {
      engine.benchmark_mode_ = true;
      engine.benchmark_.resetRecording();
      camera->reset();
    }
    AddTooltip(std::string("benchmark scene render performance and save to .csv"));
    ImGui::EndDisabled();

    if (engine.recorder_.isRenderingDone()) {
      engine.recorder_mode_ = false;
    }
    if (engine.benchmark_.isRecordingDone()) {
      ImGui::SameLine();
      if (ImGui::Button("Record")) {
        if (engine.recorder_.isSavingDone()) {
          engine.recorder_mode_ = true;
          engine.recorder_.resetSaving(engine.swapchain_->width(),
                                       engine.swapchain_->height());
          camera->reset();
        }
      }
    }
    AddTooltip(std::string("record scene renderings and save each frame as image"));
    ImGui::InputTextWithHint("file path", "file path", &engine.recorder_path_[0],
                             128);

    if (ImGui::Button("Save Transform")) {
      auto camera_filename =
          vk::utils_io::get_camera_filename(engine.last_loaded_ply_filepath_);
      vk::utils_io::save_camera_parameters(camera_filename, engine.translation_,
                                           engine.rotation_, engine.scale_);
    }
    AddTooltip(std::string("save scene transform (loads next startup)"));
  }
    
  if (ImGui::Button("+")) {
    font_size_ += 0.25;
  }
  ImGui::SameLine();
  if (ImGui::Button("-")) {
    font_size_ -= 0.25;
  }
  font_size_ = glm::clamp(font_size_, 0.5f, 2.0f);
  ImGui::SameLine();
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << font_size_;
  ImGui::Text((std::string("Font scale: ") + stream.str()).c_str());

  ImGui::End();

  performance_graph_.Insert(engine.frame_infos_, frame_index);
  if (performance_window_) {
    ImGui::Begin("Render Time Breakdown");
    ImGui::SetWindowFontScale(font_size_);
    performance_graph_.Render();
    ImGui::End();
  }
  
  // create visualization color map
  if (engine.vis_mode_ == (uint32_t)VisMode::Overdraw ||
      engine.vis_mode_ == (uint32_t)VisMode::OverdrawAlpha) {
    auto draw_list = ImGui::GetBackgroundDrawList();
    int font_size = 20;
    glm::ivec4 padding = glm::ivec4(60, 30, 30, 30);
    glm::ivec2 top_left = glm::ivec2(engine.width_ - 100, engine.height_ / 2 + 150);
    glm::ivec2 bottom_right = glm::ivec2(engine.width_ - 50, engine.height_ / 2 - 150);
    draw_list->AddRectFilled(
      ImVec2(top_left.x - padding.x, top_left.y + padding.y),
      ImVec2(bottom_right.x + padding.z, bottom_right.y - padding.w),
      ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.1f, 0.1f, 0.7f)));
    draw_list->AddRectFilledMultiColor(
      ImVec2(top_left.x, top_left.y),
      ImVec2(bottom_right.x, bottom_right.y),
      ImGui::ColorConvertFloat4ToU32(ImVec4(0.0f, 0.0f, 0.0f, 1.0f)),
      ImGui::ColorConvertFloat4ToU32(ImVec4(0.0f, 0.0f, 0.0f, 1.0f)),
      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)),
      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)));
    draw_list->AddRect(
      ImVec2(top_left.x, top_left.y),
      ImVec2(bottom_right.x, bottom_right.y),
      ImGui::ColorConvertFloat4ToU32(ImVec4(0.0f, 0.0f, 0.0f, 1.0f)));
    draw_list->AddText(
      ImGui::GetFont(), font_size,
      ImVec2(top_left.x - 40, bottom_right.y - 10),
      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)),
      std::to_string((int)engine.vis_mode_scale_).c_str());
    draw_list->AddText(
      ImGui::GetFont(), font_size,
      ImVec2(top_left.x - 40, top_left.y - 10),
      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)),
      std::to_string((int)0).c_str());

    char * color_map_title;
    if (engine.vis_mode_ == (uint32_t)VisMode::Overdraw) {
      color_map_title = "# splats";
    } else {
      color_map_title = "opacity sum";
    }

    draw_list->AddText(
      ImGui::GetFont(),
      font_size,
      ImVec2(top_left.x + 5 - padding.x, bottom_right.y - 25),
      ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)),
      color_map_title);
  }

  // draw FOV
  if (show_center_) {
    auto mouse_pos = ImGui::GetMousePos();
    auto draw_list = ImGui::GetForegroundDrawList();
    auto color = ImVec4(1.0, 0.0, 0.0, 1.0);
    draw_list->AddCircleFilled(mouse_pos, 5, ImGui::GetColorU32(color));
  }
  ImGui::Render();

  GUIInfo info;
  info.model_ = vkgs::math::ToScaleMatrix4(engine.scale_ * scale) * glm::toMat4(gq) *
                vkgs::math::ToTranslationMatrix4(engine.translation_ + gt) *
                glm::toMat4(engine.rotation_ * lq) * vkgs::math::ToTranslationMatrix4(lt);
  return info;
}

void GUI::destroy() {
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
}

PerformanceGraph::PerformanceGraph(uint32_t num_samples) : num_samples_(num_samples) {
  // initialize deque
  for (uint32_t i = 0; i < num_samples; i++) {
    samples_.push_front(PerformanceGraphFrame{});
  }
}

void PerformanceGraph::Insert(
    std::vector<std::vector<std::vector<vk::FrameInfo>>>& frame_info,
    uint32_t frame_index) {
  auto current_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  auto period = 1e6 / freq_;

  if (current_time - last_time_ >= period) {
    samples_.pop_front();
    samples_.push_back(PerformanceGraph::ConvertFrameInfo(frame_info, frame_index));
    if (current_time - last_time_ >= period * 2) {
      last_time_ = current_time;
    } else {
      last_time_ = last_time_ + period;
    }
  }
}

std::map<std::string, std::vector<float>> PerformanceGraph::data() {
  std::map<std::string, std::vector<float>> output;
  for (PerformanceGraphFrame sample : samples_) {
    output[std::string("sum_total_rank")].push_back(sample.sum_total_rank);
    output[std::string("sum_total_sort")].push_back(sample.sum_total_sort);
    output[std::string("sum_total_inverse")].push_back(sample.sum_total_inverse);
    output[std::string("sum_total_projection")].push_back(sample.sum_total_projection);
    output[std::string("sum_total_rendering")].push_back(sample.sum_total_rendering);
    output[std::string("sum_total_e2e")].push_back(sample.sum_total_e2e);
  }
  return output;
}

PerformanceGraphFrame PerformanceGraph::ConvertFrameInfo(
  std::vector<std::vector<std::vector<vk::FrameInfo>>>& frame_info,
  uint32_t frame_index
) {
  float total_rank = 0;
  float total_sort = 0;
  float total_inverse = 0;
  float total_projection = 0;
  float total_rendering = 0;

  for (uint32_t i = 0; i < frame_info.size(); i++) {
    for (uint32_t j = 0; j < frame_info[i].size(); j++) {
      total_rank += frame_info[i][j][frame_index].rank_time / 1e6;
      total_sort += frame_info[i][j][frame_index].sort_time / 1e6;
      total_inverse += frame_info[i][j][frame_index].inverse_time / 1e6;
      total_projection += frame_info[i][j][frame_index].projection_time / 1e6;
      total_rendering += frame_info[i][j][frame_index].rendering_time / 1e6;
    }
  }

  PerformanceGraphFrame output{};
  output.sum_total_rank = total_rank;
  output.sum_total_sort = total_sort + output.sum_total_rank;
  output.sum_total_inverse = total_inverse + output.sum_total_sort;
  output.sum_total_projection = total_projection + output.sum_total_inverse;
  output.sum_total_rendering = total_rendering + output.sum_total_projection;
  output.sum_total_e2e = frame_info[0][0][frame_index].end_to_end_time / 1e6;
  return output;
}

void PerformanceGraph::Render() {
  auto data = this->data();
  std::vector<float> x_values(num_samples_);
  std::iota(x_values.begin(), x_values.end(), 0);

  if (ImPlot::BeginPlot("End-to-End Frame Time", ImVec2(-1, 0),
                        ImPlotFlags_::ImPlotFlags_NoInputs)) {
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 5.0);
    ImPlot::SetupAxis(ImAxis_X1, "frame #");
    ImPlot::SetupAxis(ImAxis_Y1, "frame time (ms)");
    
    // plot in reverse order
    ImPlot::PlotShaded("e2e", &x_values[0],
                       &(data[std::string("sum_total_e2e")][0]), num_samples_);
    ImPlot::PlotShaded("rendering", &x_values[0],
                       &(data[std::string("sum_total_rendering")][0]),
                       num_samples_);
    ImPlot::PlotShaded("projection", &x_values[0],
                       &(data[std::string("sum_total_projection")][0]),
                       num_samples_);
    ImPlot::PlotShaded("inverse", &x_values[0],
                       &(data[std::string("sum_total_inverse")][0]),
                       num_samples_);
    ImPlot::PlotShaded("sort", &x_values[0],
                       &(data[std::string("sum_total_sort")][0]), num_samples_);
    ImPlot::PlotShaded("rank", &x_values[0],
                       &(data[std::string("sum_total_rank")][0]), num_samples_);

    ImPlot::EndPlot();
  }
}

};  // namespace vkgs
