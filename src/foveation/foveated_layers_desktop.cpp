// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "foveation/foveated_layers_desktop.h"

namespace fov {


/**
 * @brief Initialize the foveation layers
 * @param info foveated layers desktop
 * @param override_res_levels specify scale resolution scale for each layer
 * [0.0, 1.0]
 * @param override_radii_levels specify radii (ratio of half of horizontal resolution)
 * for each layer [0.0, 1.0]
 */
void FoveatedLayersDesktop::initialize(FoveatedLayersDesktopInfo& info,
                                       const std::vector<float>* override_res_levels,
                                       const std::vector<float>* override_radii_levels) {

  foveated_layers_.clear();

  float D_s = info.res[0];
  float V_s = info.viewer_distance;
  float W_s = info.monitor_width;

  float w_s = computeAngularDisplaySharpness(D_s, V_s, W_s);

  for (int level = 0; level < info.num_levels; level++) {
    float s_i = pow(2, level);
    float s_i_1 = pow(2, level + 1);
    if (info.num_levels == 2) {
      s_i_1 = pow(2, level + 2);
    }

    float e_i = computeAngleForScale(info.m, s_i_1, info.w_0, w_s);
    float r_i = computeRadiusFromAngle(D_s, e_i, s_i, V_s, W_s);

    FoveatedLayer layer{};
    // override the resolutions for each layer
    if (override_res_levels == nullptr || override_res_levels->empty()) {
      layer.scale = s_i;
    // override the resolutions for each layer
    } else {
      layer.scale = 1.0 / (*override_res_levels)[level];
    }

    // override radii for each layer
    if (override_radii_levels != nullptr && !override_radii_levels->empty()) {
      if (level < info.num_levels - 1) {
        r_i = ((*override_radii_levels)[level] * (info.res[1] / 2.0)) / s_i;
        e_i = computeAngleFromRadius(D_s, r_i, s_i, V_s, W_s);
      }
    }

    layer.eccentricity = e_i;
    layer.radius = r_i;
    layer.level_num = level;

    // render resolution (resolution for rendering)
    layer.render_res[0] = (info.res[0] / layer.scale);
    layer.render_res[1] = (info.res[1] / layer.scale);
    if (level < info.num_levels - 1) {
      layer.render_res[0] = r_i * 2;
      layer.render_res[1] = r_i * 2;
    }

    // projection resolution (resolution for projection matrix)
    layer.projection_res[0] = (info.res[0]);
    layer.projection_res[1] = (info.res[1]);
    if (level < info.num_levels - 1) {
      layer.projection_res[0] = r_i * 2 * s_i;
      layer.projection_res[1] = r_i * 2 * s_i;
    }

    // compute FOV for Y-axis
    float height = ((layer.projection_res[1] / s_i) / 2) * s_i;

    foveated_layers_.push_back(layer);
  }
}

/**
 * @brief Get foveated layers
 * @return foveated layers
 */
std::vector <FoveatedLayer>& FoveatedLayersDesktop::getLayers(){
  return foveated_layers_;
}

/**
 * @brief Compute angular display radius
 * @param V_s distance from viewer (cm)
 * @param W_s monitor width (cm)
 * @return e_s angle of display (radians)
 */
float FoveatedLayersDesktop::computeAngularDisplayRadius(float V_s, float W_s) {
  return atan(W_s / (2 * V_s));
}

/**
 * @brief 
 * @param D_s horizontal resolution (pixels)
 * @param V_s distance from viewer (cm)
 * @param W_s monitor width (cm)
 * @return e_s angle of display (radians)
 */
float FoveatedLayersDesktop::computeAngularDisplaySharpness(uint32_t D_s,
                                                           float V_s,
                                                           float W_s) {
  return atan((2 * W_s) / (V_s * D_s));
}

/**
 * @brief 
 * @param m falloff slope
 * @param s_i_1 scale of next level
 * @param w_0 smallest perceivable angle of fovea (radians)
 * @param w_s 
 * @return e_i angle for scale (radians)
 */
float FoveatedLayersDesktop::computeAngleForScale(float m, float s_i_1, float w_0,
                                         float w_s) {
  return ((s_i_1 * w_s) - w_0) / m;
}

/**
 * @brief Compute radius (pixels) from angle (radians)
 * @param D_s horizontal resolution (pixels)
 * @param e_i angle of level (radians)
 * @param s_i scale of level
 * @param V_i distance from viewer (cm)
 * @param W_s monitor width (cm)
 * @return radius (pixels)
 */
uint32_t FoveatedLayersDesktop::computeRadiusFromAngle(uint32_t D_s, float e_i,
                                                     float s_i, float V_s,
                                                     float W_s) {
  auto diameter = 2 * (D_s / s_i) * atan(e_i) * (V_s / W_s);
  return diameter / 2;
}

/**
 * @brief Compute radius (pixels) from angle (radians)
 * @param D_s horizontal resolution (pixels)
 * @param r_i radius of level (pixels)
 * @param s_i scale of level
 * @param V_i distance from viewer (cm)
 * @param W_s monitor width (cm)
 * @return angle (radians)
 */
float FoveatedLayersDesktop::computeAngleFromRadius(uint32_t D_s, float r_i,
                                                    float s_i, float V_s,
                                                    float W_s) {
  
  auto eccentricity = tan((r_i * s_i * W_s) / (D_s * V_s));
  return eccentricity;
}

}