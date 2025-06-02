// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CORE_STRUCTS_H
#define CORE_STRUCTS_H


namespace core {

/**
 * @brief View frustum angles
 * The angles are view-frustum angles (in radians)
 */
struct ViewFrustumAngles {
  float angle_right;
  float angle_left;
  float angle_down;
  float angle_up;
};

}

#endif