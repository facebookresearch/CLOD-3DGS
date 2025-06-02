// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "core/string.h"

namespace vkgs {
namespace str {

/**
 * @brief Split string by spaces
 * @param text input string
 * @return vector of split strings
 */
std::vector<std::string> split(std::string text) {
  std::stringstream stream(text);
  std::vector<std::string> output;
  std::string value;

  while (stream >> value) {
    output.push_back(value);
  }
  
  return output;
}

/**
 * @brief Convert vector of character arrays to vector of strings
 * @param list vector of character arrays
 * @return vector of strings
 */
std::vector<std::string> const_char_to_string(std::vector<const char*> list) {
  std::vector<std::string> output;
  for (int i = 0; i < list.size(); i++) {
    output.push_back(std::string(list[i]));
  }
  return output;
}

/**
 * @brief Combine vector of strings (removes duplicates)
 * @param list0 vector of strings
 * @param list1 vector of strings
 * @return vector of strings
 */
std::vector<std::string> combine_vectors_of_string(
  std::vector<std::string> list0,
  std::vector<std::string> list1)
{
  std::vector<std::string> output_vector;

  // add strings from list0
  for (int i = 0; i < list0.size(); i++) {
    bool found = false;
    for (int j = 0; j < output_vector.size(); j++) {
      if (output_vector[j] == list0[i]) {
        found = true;
      }
    }
    if (!found) {
      output_vector.push_back(list0[i]);
    }
  }

  // add strings from list1
  for (int i = 0; i < list1.size(); i++) {
    bool found = false;
    for (int j = 0; j < output_vector.size(); j++) {
      if (output_vector[j] == list1[i]) {
        found = true;
      }
    }
    if (!found) {
      output_vector.push_back(list1[i]);
    }
  }

  return output_vector;
}

/**
 * @brief Convert string to lowercase
 * @param text input strings
 * @return lowercase string
 */
std::string to_lower(std::string text) {
  std::string output = "";
  for (char& c : text) {
    output += std::tolower(c);
  }
  return output;
}

};
};  // namespace
