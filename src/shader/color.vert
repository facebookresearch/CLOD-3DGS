// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;

layout (set = 0, binding = 0) uniform Camera {
  mat4 projection;
  mat4 view;
  mat4 eye;
  vec3 camera_position;
  float pad0;
  uvec2 screen_size;  // (width, height)
  float z_near;
  float z_far;
};

layout (push_constant, std430) uniform PushConstants {
  mat4 model;
  uint time;
};

layout (location = 0) out vec4 out_color;

void main() {
  gl_Position = projection * view * model * vec4(position, 1.f);
  out_color = color;
}
