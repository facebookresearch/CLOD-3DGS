// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_STRING_H
#define VKGS_ENGINE_STRING_H

#include <set>
#include <string>
#include <sstream>
#include <vector>

namespace vkgs {
namespace str {

std::vector<std::string> split(std::string text);
std::vector<std::string> const_char_to_string(std::vector<const char*> list);
std::vector<std::string> combine_vectors_of_string(
    std::vector<std::string> list0, std::vector<std::string> list1);
std::string to_lower(std::string text);

};
};  // namespace

#endif  // VKGS_ENGINE_STRING_H