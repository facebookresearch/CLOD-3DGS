// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FOVEATION_MATH_H
#define FOVEATION_MATH_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace fov {
namespace math {

glm::mat4 perspective_symmetric(float fov, float width, float height,
                                float res_x, float res_y, float near, float far,
                                float cx = 0.0, float cy = 0.0);

glm::mat4 perspective_asymmetric(float angle_right, float angle_left,
                                 float angle_down, float angle_up, float width,
                                 float height, float res_x, float res_y,
                                 float near, float far, float cx = 0.5f,
                                 float cy = 0.5f);
}
}

#endif