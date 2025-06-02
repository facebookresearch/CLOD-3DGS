// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef FOVEATION_FOVEATED_LAYERS_DESKTOP_H
#define FOVEATION_FOVEATED_LAYERS_DESKTOP_H

#include "foveation/foveated_layers.h"

namespace fov {

/**
 * @brief Default parameters for foveated rendering layers
 */
struct FoveatedLayersDesktopInfo {
  int num_levels;
  int res[2];
  float viewer_distance = 59;  // distance from viewer (cm)
  float monitor_width = 51;  // monitor width (cm)
  float w_0 = (1.0 / 48.0) * (glm::pi<float>() / 180.0);
  float m = 0.0275;
};

/**
 * @brief Foveated rendering layer generator
 * Foveation logic based on "Foveated 3D Graphics" by Guenter et al. [2012]
 */
class FoveatedLayersDesktop : FoveatedLayers {
public:
  FoveatedLayersDesktop() = default;
 
  void initialize(FoveatedLayersDesktopInfo& info,
                  const std::vector<float>* override_res_levels = nullptr,
                  const std::vector<float>* override_radii_levels = nullptr);
 
  std::vector<FoveatedLayer>& getLayers();

 private:
  float computeAngularDisplayRadius(float V_s, float W_s);
  float computeAngularDisplaySharpness(uint32_t D_s, float V_s, float W_s);
  float computeAngleForScale(float m, float s_i, float w_0, float w_s);
  uint32_t computeRadiusFromAngle(uint32_t D_s, float e_i, float s_i,
                                float V_s, float W_s);
  float computeAngleFromRadius(uint32_t D_s, float r_i, float s_i, float V_s,
                                  float W_s);
};

}

#endif