// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/benchmark.h"

namespace vkgs {
namespace vk {

/**
 * @brief constructor
 * @param num_levels number of foveated layers
 */
Benchmark::Benchmark(uint32_t num_levels) {
  num_levels_ = num_levels;
  
  lod_data_.resize(num_levels);

  rank_time_.resize(num_levels);
  sort_time_.resize(num_levels);
  inverse_time_.resize(num_levels);
  projection_time_.resize(num_levels);
  rendering_time_.resize(num_levels);
  end_to_end_time_.resize(num_levels);
};

/**
 * @brief destructor
 */
Benchmark::~Benchmark(){};

/**
 * @brief incremement frame number
 */
void Benchmark::incrementFrame() { frame_++; }

/**
 * @brief get frame number
 */
uint32_t Benchmark::frame() { return frame_; }

/**
 * @brief record timing data
 * @param frame_info frame info data structure
 * @param level foveated layer level
 */
void Benchmark::recordTimingData(FrameInfo& frame_info, uint32_t level) {
  if (frame_ >= num_frames_) {
    return;
  }

  // record timing data
  rank_time_[level].push_back(frame_info.rank_time);
  sort_time_[level].push_back(frame_info.sort_time);
  inverse_time_[level].push_back(frame_info.inverse_time);
  projection_time_[level].push_back(frame_info.projection_time);
  rendering_time_[level].push_back(frame_info.rendering_time);
  end_to_end_time_[level].push_back(frame_info.end_to_end_time);
};

/**
 * @brief record LOD data
 * @param level foveated layer level
 * @param lod_data LOD level
 */
void Benchmark::recordLODData(uint32_t level, float lod_data) {
  lod_data_[level].push_back(lod_data);
}

/**
 * @brief get LOD level
 * @param level foveated layer level
 * @param frame frame number
 * @return LOD level
 */
float Benchmark::getLODLevel(uint32_t level, uint32_t frame) {
  return lod_data_[level][frame];
}

void Benchmark::saveTimingData(std::string dir_name) {
  if (!std::filesystem::exists(dir_name)) {
    std::filesystem::create_directory(dir_name);
  }

  std::ofstream csv_file;
  csv_file.open(std::filesystem::path(dir_name) /
                std::filesystem::path("benchmark.csv"));

  // write header
  for (uint32_t level = 0; level < num_levels_; level++) {
    csv_file << "rank_" + std::to_string(level) + ",";
    csv_file << "sort_" + std::to_string(level) + ",";
    csv_file << "inverse_" + std::to_string(level) + ",";
    csv_file << "projection_" + std::to_string(level) + ",";
    csv_file << "rendering_" + std::to_string(level) + ",";
    csv_file << "end_to_end_" + std::to_string(level) + ",";
  }
  csv_file << "\n";

  // write data
  for (uint32_t frame = 0; frame < num_frames_; frame++) {
    for (uint32_t level = 0; level < num_levels_; level++) {
      double rank_time = static_cast<double>(rank_time_[level][frame]) / 1e6;
      double sort_time = static_cast<double>(sort_time_[level][frame]) / 1e6;
      double inverse_time =
          static_cast<double>(inverse_time_[level][frame]) / 1e6;
      double projection_time =
          static_cast<double>(projection_time_[level][frame]) / 1e6;
      double rendering_time =
          static_cast<double>(rendering_time_[level][frame]) / 1e6;
      double end_to_end_time =
          static_cast<double>(end_to_end_time_[level][frame]) / 1e6;

      csv_file << std::to_string(rank_time) + ",";
      csv_file << std::to_string(sort_time) + ",";
      csv_file << std::to_string(inverse_time) + ",";
      csv_file << std::to_string(projection_time) + ",";
      csv_file << std::to_string(rendering_time) + ",";
      csv_file << std::to_string(end_to_end_time) + ",";
    }
    csv_file << "\n";
  }
  csv_file.close();
}

/**
 * @brief reset recording data
 * Run this function before recording again
 */
void Benchmark::resetRecording() {
  // clear all recorded times
  for (uint32_t level = 0; level < num_levels_; level++) {
    lod_data_[level].clear();

    rank_time_[level].clear();
    sort_time_[level].clear();
    inverse_time_[level].clear();
    projection_time_[level].clear();
    rendering_time_[level].clear();
    end_to_end_time_[level].clear();
  }

  frame_ = 0;
};

/**
 * @brief tests if recording is done
 * @return true if recording is done
 */
bool Benchmark::isRecordingDone() { return (frame_ >= num_frames_); }

/**
 * @brief get mean end-to-end time
 * @return mean end-to-end time (in milliseconds)
 */
float Benchmark::mean_end_to_end_time() {
  float output = 0.0;
  uint32_t num_samples = 0;
  
  for (uint32_t frame = 0; frame < num_frames_; frame++) {
    output += end_to_end_time_[0][frame] / 1e6;
  }
  return output / num_frames_;
}

/**
 * @brief get end-to-end time vector
 * @return end-to-end time vector (in milliseconds)
 */
std::vector<float> Benchmark::end_to_end_time() {
  std::vector<float> output;
  for (uint32_t frame = 0; frame < num_frames_; frame++) {
    output.push_back(end_to_end_time_[0][frame] / 1e6);
  }
  return output;
}

/**
 * @brief get maximum number of frames
 * @return maximum number of frames
 */
const uint32_t Benchmark::num_frames() const { return num_frames_; }
const uint32_t Benchmark::num_frames(uint32_t num) {
  num_frames_ = num;
  resetRecording();
  return num_frames_;
};


};  // namespace vk
};  // namespace vkgs
