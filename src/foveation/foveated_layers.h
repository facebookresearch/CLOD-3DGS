// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FOVEATION_FOVEATED_LAYERS_H
#define FOVEATION_FOVEATED_LAYERS_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <core/structs.h>

namespace fov {
  
/**
 * @brief Foveated layer data structure
 * Each layer corresponds to the foveation layer (e.g., foveal, peripheral).
 */
struct FoveatedLayer {
  int level_num;  // 0 is full screen (lowest res)
  float scale;
  // eccentricity (in radians) for half-angle (corresponds to radius)
  float eccentricity;
  uint32_t render_res[2];      // render resolution
  uint32_t projection_res[2];  // original resolution
  uint32_t radius;
};


core::ViewFrustumAngles cameraFovyToAngles(float fovy, float width,
                               float height);


/**
 * @brief Collection of FoveatedLayer objects
 */
class FoveatedLayers {
 public:
  FoveatedLayers() = default;

protected:
  std::vector<FoveatedLayer> foveated_layers_;
};

}

#endif