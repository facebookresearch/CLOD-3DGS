// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_UTILS_IO_H
#define VKGS_ENGINE_VULKAN_UTILS_IO_H

#include <filesystem>
#include <fstream>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace vkgs {
namespace vk {
namespace utils_io {

/**
 * @brief Get model transform filename
 * @param input_filename filename of .ply file
 * @return corresponding .transform.bin file
 */
std::string get_camera_filename(std::string input_filename);

/**
 * @brief Save model transformation
 * @param filename camera transformation matrix name
 * @param translation translation
 * @param rotation rotation
 * @param scale scale
 */
void save_camera_parameters(
  std::string filename,
  glm::vec3 translation,
  glm::quat rotation,
  float scale
);

/**
 * @brief Load model transformation
 * @param filename camera transformation matrix name
 * @param translation translation
 * @param rotation rotation
 * @param scale scale
 */
void load_camera_parameters(std::string filename, glm::vec3& translation,
                            glm::quat& rotation, float& scale);

void write_float(std::ofstream& file, float& scalar);
void write_vec3(std::ofstream& file, glm::vec3& vec3);
void write_quat(std::ofstream& file, glm::quat& quat);

void read_float(std::ifstream& file, float& scalar);
void read_vec3(std::ifstream& file, glm::vec3& vec3);
void read_quat(std::ifstream& file, glm::quat& quat);

}  // namespace utils_io
}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_UTILS_IO_H
