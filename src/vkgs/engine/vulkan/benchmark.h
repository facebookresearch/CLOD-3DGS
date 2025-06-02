// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_BENCHMARK_H
#define VKGS_ENGINE_VULKAN_BENCHMARK_H

#include <filesystem>
#include <fstream>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/structs.h"

namespace vkgs {
namespace vk {

/**
 * @brief Storing and processing benchmark (timing) data
 */
class Benchmark { 
 public:
  Benchmark(uint32_t num_levels);

  ~Benchmark();

  void incrementFrame();
  uint32_t frame();

  void recordTimingData(FrameInfo& frame_info, uint32_t level);
  void recordLODData(uint32_t level, float lod_data);
  float getLODLevel(uint32_t level, uint32_t frame);
  void saveTimingData(std::string dir_name);

  void resetRecording();
  bool isRecordingDone();

  float mean_end_to_end_time();
  std::vector<float> end_to_end_time();
  const uint32_t num_frames() const;
  const uint32_t num_frames(uint32_t num);

 private:
  uint32_t frame_ = 0;
  uint32_t num_frames_ = 0;
  uint32_t num_levels_ = 0;

  std::vector<std::vector<float>> lod_data_;

  std::vector<std::vector<uint64_t>> rank_time_;
  std::vector<std::vector<uint64_t>> sort_time_;
  std::vector<std::vector<uint64_t>> inverse_time_;
  std::vector<std::vector<uint64_t>> projection_time_;
  std::vector<std::vector<uint64_t>> rendering_time_;
  std::vector<std::vector<uint64_t>> end_to_end_time_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_BENCHMARK_H