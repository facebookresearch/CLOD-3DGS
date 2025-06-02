// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_GUI_H
#define VKGS_ENGINE_GUI_H

#include <chrono>
#include <deque>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>

#include "vkgs/engine/utils/math.h"
#include "vkgs/engine/vulkan/structs.h"
#include "vkgs/engine/vulkan/utils_io.h"


namespace vkgs {

class Context;
class Engine;
class GLFWWindow;


/**
 * @brief Performance data for a single frame
 */
struct PerformanceGraphFrame {
  float sum_total_rank;
  float sum_total_sort;
  float sum_total_inverse;
  float sum_total_projection;
  float sum_total_rendering;
  float sum_total_e2e;
};


/**
 * @brief Performance graph GUI element
 */
class PerformanceGraph {
 public:
  PerformanceGraph(uint32_t num_samples = 180);
  ~PerformanceGraph() = default;

  void Insert(std::vector<std::vector<std::vector<vk::FrameInfo>>>& frame_info,
              uint32_t frame_index);
  std::map<std::string, std::vector<float>> data();
  void Render();

  static PerformanceGraphFrame ConvertFrameInfo(
    std::vector<std::vector<std::vector<vk::FrameInfo>>>& frame_info,
    uint32_t frame_index);

 private:
  uint32_t num_samples_;
  uint32_t freq_ = 60;
  long long last_time_ = 0;

  std::deque<PerformanceGraphFrame> samples_;
};


/**
 * @brief Model transformation matrix in GUI
 */
struct GUIInfo {
  glm::mat4 model_;
};


/**
 * @brief GUI for viewer
 */
class GUI {
 friend class Engine;
 public:
  GUI();
  ~GUI();

  void prepare();

  void initialize(Engine& engine);
  
  GUIInfo update(Engine& engine, uint32_t frame_index);

  void destroy();

 protected:
  bool show_axis_ = false;
  bool show_grid_ = false;
  bool show_views_ = false;
  bool show_center_ = false;
  int budget_mode_ = 0;

  bool demo_mode_ = false;
  bool blending_mode_ = true;
  bool performance_window_ = false;
  PerformanceGraph performance_graph_;

  float font_size_ = 1.0f;
};

};  // namespace vkgs

#endif  // VKGS_ENGINE_GUI_H