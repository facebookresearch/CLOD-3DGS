// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#version 460

layout (constant_id = 0) const bool DEBUG = false;
layout (constant_id = 1) const bool DITHER = false;

layout (location = 0) in vec4 color;
layout (location = 1) in vec2 position;
layout (location = 2) in flat uint index;

layout (location = 0) out vec4 out_color;

layout (push_constant, std430) uniform PushConstants {
  mat4 model;
  uint time;
  uint vis_mode;
  float vis_mode_scale;
};

// LCG pseudo-random number generator
vec4 rand(uint seed) {
  uint m = 2147483647;  // 2**31-1
  uint a = 48271;
  uint c = 0;
  uint output_uint = (a * seed + c) % m;
  return vec4(output_uint) / float(m);
}

void main() {
  float gaussian_alpha = exp(-0.5f * dot(position, position));
  float alpha = color.a * gaussian_alpha;
  out_color = color;

  // add dithering
  if (DITHER) {
    uint seed = (time % 2147483647) + uint(gl_FragCoord.x * 7919) + uint(gl_FragCoord.y * 7909) + (index * 7901);
    vec4 noise = -(rand(seed) * (1.0f / 256.0f)) + (0.5f / 256.0f);
    alpha = alpha + noise.a;
    out_color.rgb = out_color.rgb + noise.rgb;
  }

  // premultiplied alpha
  out_color = vec4(out_color.rgb * alpha, alpha);

  if (DEBUG) {
    if (vis_mode == 1) {
      out_color = vec4(1.0f) / vis_mode_scale;
    }
    else if (vis_mode == 2) {
      out_color = vec4(alpha) / vis_mode_scale;
    }
  }
}
