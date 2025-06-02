// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#version 460

layout (constant_id = 0) const int NUM_LEVELS = 1;
layout (constant_id = 1) const bool DEBUG = true;

layout (set = 0, binding = 0) uniform sampler samp;
layout (set = 0, binding = 1) uniform texture2D textures[NUM_LEVELS];

layout (location = 0) out vec4 final_color;
layout (location = 3) in vertex_data{
	vec2 uv;
}vertex;

layout (push_constant, std430) uniform PushConstants {
  vec2 center;
  vec2 levels[4];
  float res_scales[4];
  ivec2 full_res;
  int blending;
  int eye;
} push_constants;


// scale coordinates, taking into account rounding offset
vec2 scaled_coords(vec2 coords, float scale) {
	vec2 scaled_coords = coords * scale;
	return scaled_coords;
}


void main(void) {
	// base layer
	float scale = push_constants.res_scales[NUM_LEVELS-1];
	final_color = texture(sampler2D(textures[NUM_LEVELS-1], samp), scaled_coords(vertex.uv, scale));
	
	// other (higher resolution) layers
	for (int level = NUM_LEVELS-2; level > -1; level--)
	{
		float s_i = pow(2.0, level);
		
		float min_x = push_constants.center.x - (push_constants.levels[level].x / 2);
		float min_y = push_constants.center.y - (push_constants.levels[level].y / 2);
		float max_x = push_constants.center.x + (push_constants.levels[level].x / 2);
		float max_y = push_constants.center.y + (push_constants.levels[level].y / 2);
		
		if (vertex.uv.x > min_x && vertex.uv.y > min_y && vertex.uv.x < max_x && vertex.uv.y < max_y)
		{
			vec2 coords = vec2(0, 0);
			coords.x = (vertex.uv.x - min_x) / (max_x - min_x);
			coords.y = (vertex.uv.y - min_y) / (max_y - min_y);
			scale = push_constants.res_scales[level];
			vec4 layer_color = texture(sampler2D(textures[level], samp), scaled_coords(coords, scale));

			if (push_constants.blending == 1) {
				float radius = distance((coords - 0.5) * 2.0, vec2(0, 0));

				float alpha = smoothstep(0.6, 1, radius);
				layer_color = (layer_color * (1.0 - alpha)) + (final_color * alpha);
				
				// for debug
				if (DEBUG) {
				  if (radius < 0.1) {
					layer_color = vec4(1.0, 0.0, 0.0, 1.0);
				  }
				}
			}
			
			final_color = layer_color;
		}
	}
}
