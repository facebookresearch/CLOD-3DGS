// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#version 460

layout (std430, set = 1, binding = 1) readonly buffer Instances {
  float instances[];  // (N, 10). 3 for ndc position, 3 for scale rot, 4 for color
};

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_position;
layout (location = 2) out flat uint out_id;

layout (push_constant, std430) uniform PushConstants {
  mat4 model;
  uint time;
  uint vis_mode;
  float vis_mode_scale;
};

void main() {
  // index [0,1,2,2,1,3], 4 vertices for a splat.
  int index = gl_VertexIndex / 4;
  out_id = index;
  vec3 ndc_position = vec3(instances[index * 10 + 0], instances[index * 10 + 1], instances[index * 10 + 2]);
  vec2 scale = vec2(instances[index * 10 + 3], instances[index * 10 + 4]);
  float theta = instances[index * 10 + 5];
  vec4 color = vec4(instances[index * 10 + 6], instances[index * 10 + 7], instances[index * 10 + 8], instances[index * 10 + 9]);

  // quad positions (-1, -1), (-1, 1), (1, -1), (1, 1), ccw in screen space.
  int vert_index = gl_VertexIndex % 4;
  vec2 position = vec2(vert_index / 2, vert_index % 2) * 2.f - 1.f;

  mat2 rot = mat2(cos(theta), sin(theta), -sin(theta), cos(theta));

  float confidence_radius = 3.f;

  gl_Position = vec4(ndc_position + vec3(rot * (scale * position) * confidence_radius, 0.f), 1.f);
  out_color = color;
  out_position = position * confidence_radius;
}
