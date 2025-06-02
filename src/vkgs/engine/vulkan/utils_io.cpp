// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/utils_io.h"


namespace vkgs {
namespace vk {
namespace utils_io
{

std::string get_camera_filename(std::string input_filename) {
  auto basename = input_filename.substr(0, input_filename.find_last_of("."));
  return basename + ".transform.bin";
}

void save_camera_parameters(std::string filename, glm::vec3 translation,
                            glm::quat rotation, float scale) {
  std::ofstream file(filename.c_str(), std::ios::binary);
  write_vec3(file, translation);
  write_quat(file, rotation);
  write_float(file, scale);
  file.close();
}

void load_camera_parameters(std::string filename, glm::vec3& translation, glm::quat& rotation, float& scale) { 
  if (std::filesystem::exists(filename.c_str())) {
    std::ifstream file(filename.c_str(), std::ios::binary);
    read_vec3(file, translation);
    read_quat(file, rotation);
    read_float(file, scale);
    file.close();
  }
}

void write_float(std::ofstream& file, float& scalar) {
  file.write(reinterpret_cast<char*>(&scalar), sizeof(float));
}

void write_vec3(std::ofstream& file, glm::vec3& vec3) {
  file.write(reinterpret_cast<char*>(&vec3.x), sizeof(float));
  file.write(reinterpret_cast<char*>(&vec3.y), sizeof(float));
  file.write(reinterpret_cast<char*>(&vec3.z), sizeof(float));
}

void write_quat(std::ofstream& file, glm::quat& quat) {
  file.write(reinterpret_cast<char*>(&quat.w), sizeof(float));
  file.write(reinterpret_cast<char*>(&quat.x), sizeof(float));
  file.write(reinterpret_cast<char*>(&quat.y), sizeof(float));
  file.write(reinterpret_cast<char*>(&quat.z), sizeof(float));
}


void read_float(std::ifstream& file, float& scalar) {
  file.read(reinterpret_cast<char*>(&scalar), sizeof(float));
}

void read_vec3(std::ifstream& file, glm::vec3& vec3) {
  file.read(reinterpret_cast<char*>(&vec3.x), sizeof(float));
  file.read(reinterpret_cast<char*>(&vec3.y), sizeof(float));
  file.read(reinterpret_cast<char*>(&vec3.z), sizeof(float));
}

void read_quat(std::ifstream& file, glm::quat& quat) {
  file.read(reinterpret_cast<char*>(&quat.w), sizeof(float));
  file.read(reinterpret_cast<char*>(&quat.x), sizeof(float));
  file.read(reinterpret_cast<char*>(&quat.y), sizeof(float));
  file.read(reinterpret_cast<char*>(&quat.z), sizeof(float));
}

}  // namespace utils_io
}  // namespace vk
}  // namespace vkgs
