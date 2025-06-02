// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv_vertex;

layout (location = 3) out vertex_data{
	vec2 uv;
}vertex;

void main(void) {
  vertex.uv = uv_vertex;
  gl_Position = vec4(position, 1.0);
}
