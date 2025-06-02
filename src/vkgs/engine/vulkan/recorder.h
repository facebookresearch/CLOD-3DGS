// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_RECORDER_H
#define VKGS_ENGINE_VULKAN_RECORDER_H

#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/barrier.h"
#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {


/**
 * @brief Saving renders from camera path
 */
class Recorder {
 public:
  Recorder();

  ~Recorder();

  void saveImage(std::string dir_name, Context& context, const VkImage& image,
                 std::string& save_type);
  void saveImageThread(std::string dir_name, const unsigned char* image_data,
                       uint32_t frame, uint32_t row_pitch);
  void saveArrayThread(const unsigned char* image_data,
                       uint32_t frame, uint32_t row_pitch);
  
  uint32_t frame();

  void resetSaving(uint32_t width, uint32_t height);
    
  bool isSavingDone();
  bool isRenderingDone();

  std::vector<std::vector<unsigned char>>& image_data();
  const uint32_t num_frames() const;
  const uint32_t num_frames(uint32_t num);

 private:
  uint32_t frame_ = 0;
  uint32_t num_frames_ = 0;
  bool ready_to_save_ = true;

  std::string save_type_ = "image";
  
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  std::atomic<uint32_t> num_saved_images_{0};

  std::vector<std::vector<unsigned char>> image_data_;
  std::vector<std::vector<unsigned char>> image_data_buffer_;
  std::vector<std::thread> image_save_threads_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_RECORDER_H
