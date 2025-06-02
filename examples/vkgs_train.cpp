// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

/*
This file is meant for debugging the training process
*/

#include <iostream>

#include <argparse/argparse.hpp>

#include <vkgs/engine/engine_api.h>
#include "../src/vkgs/engine/config.h"
#include "../src/vkgs/engine/sample.h"

class Config;

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("vkgs");
  parser.add_argument("-i", "--input").help("input ply file [.ply].");
  parser.add_argument("--config").help("config [.yaml].");
  parser.add_argument("--mode").help("mode.").default_value("immediate");
  parser.parse_args(argc, argv);

  vkgs::Config config(
    parser.get<std::string>("config"),
    parser.get<std::string>("mode")
  );

  uint32_t num_frames = 5;
  config.num_frames_benchmark(num_frames);
  config.num_frames_recorder(num_frames);
  
  vkgs::EngineAPI engine(config);

  auto ply_filepath = parser.get<std::string>("input");
  engine.LoadSplats(ply_filepath);

  engine.Start();

  uint32_t iterations = 0;
  uint32_t max_iterations = 10;

  // warm start
  vkgs::SampleParams sample_params;
  sample_params.num_frames_benchmark = num_frames;
  sample_params.num_frames_recorder = num_frames;
  for (uint32_t i = 0; i < num_frames; i++) {
    sample_params.lod.push_back(std::vector<float>(num_frames, 1.0));
    sample_params.res.push_back(std::vector<float>(num_frames, 1.0));
    sample_params.lod_params.push_back(std::vector<glm::vec4>(num_frames, glm::vec4(1.0)));
  }

  vkgs::SampleState sample_state;
  sample_state.pos = std::vector<glm::vec3>();
  sample_state.quat = std::vector<glm::quat>();
  sample_state.center = std::vector<glm::vec2>();
  sample_state.view_angles = std::vector<core::ViewFrustumAngles>();
  for (uint32_t i = 0; i < num_frames; i++) {
    sample_state.pos.push_back(glm::vec3(0.0, 0.0, 0.0));
    sample_state.quat.push_back(glm::quat(1.0, 0.0, 0.0, 0.0));
    sample_state.center.push_back(glm::vec2(0.5, 0.5));
    core::ViewFrustumAngles view_angles;
    view_angles.angle_right = 20.0f;
    view_angles.angle_left = -20.0f;
    view_angles.angle_up = 20.0f;
    view_angles.angle_down = -20.0f;
    sample_state.view_angles.push_back(view_angles);
  }

  while (iterations < max_iterations) {
    auto result = engine.Sample(sample_params, sample_state);
    iterations++;
  }

  std::cout << "Finished rendering" << std::endl;
  
  engine.End();

  return 0;
}
