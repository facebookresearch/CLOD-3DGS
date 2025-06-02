// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_CONFIG_H
#define VKGS_ENGINE_CONFIG_H

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

/// \file

namespace vkgs {

/**
 * @brief Engine rendering mode
 */
enum class Mode {
  /**Desktop rendering mode for viewer*/
  Desktop,
  /**VR rendering mode*/
  VR,
  /**Immediate mode used in Python API*/
  Immediate,
};


/**
 * @brief Visualization mode for debugging
 */
enum class VisMode {
  /**Default RGB rendering for splatting*/
  Normal = 0,
  /**Grayscale rendering showing quad overdraw*/
  Overdraw = 1,
  /**Grayscale rendering showing opacity accumulation*/
  OverdrawAlpha = 2,
};
const static uint32_t VisModeCount = 3;


/**
 * @brief Configuration settings for Engine
 */
class Config {
 public:
  Config();
  Config(std::string& filename, std::string& mode, bool dynamic_res = true,
         bool debug = false, std::string& color_mode = std::string("sfloat16"),
         bool dither = false,
         uint32_t max_splats = (std::numeric_limits<uint32_t>::max)(),
         std::string& view_filename = std::string(""), float time_budget = -1.0f);
  ~Config();

  uint32_t num_levels();
  bool lod();
  std::vector<float>& fov_res();
  const std::string& color_mode();
  const bool dither();
  const uint32_t max_splats();
  const std::string& view_filename();
  const float time_budget();

  std::vector<float>& Config::lod_params();
  std::vector<float>& Config::lod_params(float min_lod, float max_lod,
                                         float min_dist, float max_dist);
  
  const bool debug();
  const bool debug(const bool debug);

  const VisMode vis_mode();
  const VisMode vis_mode(std::string& vis_mode);
  const VisMode vis_mode(VisMode vis_mode);

  const float vis_scale();
  const float vis_scale(float vis_scale);

  const std::vector<float>& radii_levels();
  const std::vector<float>& radii_levels(std::vector<float>& radii_levels);

  bool dynamic_res();
  bool dynamic_res(bool enable);

  std::vector<uint32_t>& res();
  std::vector<uint32_t>& res(uint32_t width, uint32_t height);

  uint32_t num_frames_recorder();
  uint32_t num_frames_recorder(uint32_t num_frames);
  uint32_t num_frames_benchmark();
  uint32_t num_frames_benchmark(uint32_t num_frames);

  Mode mode();

private:
  // parameters
  std::vector<uint32_t> res_ = {1920, 1080};
  std::vector<float> lod_params_ = {1.0, 1.0, 1.0, 1.0};
  uint32_t num_levels_ = 1;
  bool debug_ = false;
  bool dynamic_res_ = false;
  bool lod_ = false;
  std::vector<float> fov_res_;
  std::string color_mode_;
  bool dither_ = false;
  uint32_t max_splats_ = (std::numeric_limits<uint32_t>::max)();
  std::string view_filename_ = "";
  std::vector<float> radii_levels_;
  float time_budget_ = -1.0f;
  VisMode vis_mode_ = VisMode::Normal;
  float vis_scale_ = 1.0f;
  
  // benchmark
  uint32_t num_frames_recorder_ = 1200;
  uint32_t num_frames_benchmark_ = 1200;

  Mode mode_ = Mode::Desktop;
};

};  // namespace vkgs

#endif  // VKGS_ENGINE_CONFIG_H