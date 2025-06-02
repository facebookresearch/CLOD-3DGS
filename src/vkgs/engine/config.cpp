// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/config.h"

#include "yaml-cpp/yaml.h"

#include "core/string.h"

namespace vkgs {

Config::Config() = default;
    
Config::Config(
  std::string& filename,
  std::string& mode,
  bool dynamic_res,
  bool debug,
  std::string& color_mode,
  bool dither,
  uint32_t max_splats,
  std::string& view_filename,
  float time_budget
) {
  YAML::Node config;
  try {
    config = YAML::LoadFile(filename);
  } catch (const YAML::BadFile& e) {
    printf("Incorrect config file [.yaml] path.\n");
  }
  
  if (config["num_levels"]) {
    num_levels_ = config["num_levels"].as<uint32_t>();
  }

  if (config["lod"]) {
    lod_ = config["lod"].as<bool>();
  }

  if (config["fov_res"]) {
    for (int i = 0; i < config["fov_res"].size(); i++) {
      fov_res_.push_back(config["fov_res"][i].as<float>());
    }  
  }

  std::string mode_lower = str::to_lower(mode);

  if (mode_lower == "desktop") {
    mode_ = Mode::Desktop;
  } else if (mode_lower == "vr") {
    mode_ = Mode::VR;
  } else if (mode_lower == "immediate") {
    mode_ = Mode::Immediate;
  } else {
    throw std::invalid_argument("invalid mode");
  }

  dynamic_res_ = dynamic_res;
  debug_ = debug;
  color_mode_ = color_mode;
  dither_ = dither;
  max_splats_ = max_splats;
  view_filename_ = view_filename;
  time_budget_ = time_budget;
};

Config::~Config() = default;

std::vector<uint32_t>& Config::res() { return res_; };

std::vector<uint32_t>& Config::res(uint32_t width, uint32_t height) {
  res_[0] = width;
  res_[1] = height;
  return res_;
};

std::vector<float>& Config::lod_params() { return lod_params_; };

std::vector<float>& Config::lod_params(float min_lod, float max_lod, float min_dist, float max_dist) {
  lod_params_[0] = min_lod;
  lod_params_[1] = max_lod;
  lod_params_[2] = min_dist;
  lod_params_[3] = max_dist;
  return lod_params_;
};

uint32_t Config::num_levels() { return num_levels_; };
bool Config::lod() { return lod_; };
std::vector<float>& Config::fov_res() { return fov_res_; };
Mode Config::mode() { return mode_; };
const std::string& Config::color_mode() { return color_mode_; };
const bool Config::dither() { return dither_; };
const uint32_t Config::max_splats() { return max_splats_; }
const std::string& Config::view_filename() { return view_filename_; }
const float Config::time_budget() { return time_budget_; }

const bool Config::debug() { return debug_; };
const bool Config::debug(const bool debug) {
  debug_ = debug;
  return debug_;
};

const VisMode Config::vis_mode() {return vis_mode_;}
const VisMode Config::vis_mode(std::string& vis_mode) {
  if (vis_mode == "normal") {
    vis_mode_ = VisMode::Normal;
  } else if (vis_mode == "overdraw") {
    vis_mode_ = VisMode::Overdraw;
  } else if (vis_mode == "overdraw_alpha") {
    vis_mode_ = VisMode::OverdrawAlpha;
  }
  return vis_mode_;
};
const VisMode Config::vis_mode(VisMode vis_mode) {
  vis_mode_ = vis_mode;
  return vis_mode_;
};

const float Config::vis_scale() { return vis_scale_; }
const float Config::vis_scale(float vis_scale) {
  vis_scale_ = vis_scale;
  return vis_scale_;
};

const std::vector<float>& Config::radii_levels() { return radii_levels_; };
const std::vector<float>& Config::radii_levels(std::vector<float>& radii_levels) {
  radii_levels_ = radii_levels;
  return radii_levels_;
};

bool Config::dynamic_res() { return dynamic_res_; };
bool Config::dynamic_res(bool enable_dynamic_res) {
  dynamic_res_ = enable_dynamic_res;
  return dynamic_res_;
};

uint32_t Config::num_frames_recorder() { return num_frames_recorder_; };
uint32_t Config::num_frames_recorder(uint32_t num_frames) {
  num_frames_recorder_ = num_frames;
  return num_frames_recorder_;
};

uint32_t Config::num_frames_benchmark() { return num_frames_benchmark_; };
uint32_t Config::num_frames_benchmark(uint32_t num_frames) {
  num_frames_benchmark_ = num_frames;
  return num_frames_benchmark_;
};

};  // namespace vkgs
