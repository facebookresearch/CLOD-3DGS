// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "foveation/foveated_layers.h"

namespace fov {

/**
 * @brief Convert camera vertical field-of-view to view frustum angles
 * @param fovy vertical field-of-view (radians)
 * @param width render width (pixels)
 * @param height render height (pixels)
 * @return view frustum angles
 */
core::ViewFrustumAngles cameraFovyToAngles(float fovy, float width, float height) {
  float aspect = width / height;
  float fovx = 2 * atan(tan(fovy*0.5) * aspect);

  core::ViewFrustumAngles output{};
  output.angle_right = fovx * 0.5;
  output.angle_left = -fovx * 0.5;
  output.angle_down = -fovy * 0.5;
  output.angle_up = fovy * 0.5;
  return output;
}

}