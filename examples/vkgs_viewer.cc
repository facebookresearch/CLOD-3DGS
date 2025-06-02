// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <limits>

#include <argparse/argparse.hpp>

#include <vkgs/engine/engine_api.h>
#include "../src/vkgs/engine/config.h"

class Config;

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("vkgs");
  parser.add_argument("-i", "--input")
      .help("input ply file [.ply].");
  parser.add_argument("--config").help("config [.yaml].");
  parser.add_argument("--view_file")
      .help("view file [.yaml].")
      .default_value("");
  parser.add_argument("--dynamic_res")
      .help("dynamic resolution.")
      .default_value(false)
      .implicit_value(true);
  parser.add_argument("--debug")
      .help("debug mode.")
      .default_value(false)
      .implicit_value(true);
  parser.add_argument("--time_budget")
      .help("time budget in milliseconds.")
      .scan<'g', float>()
      .default_value(-1.0f);
  parser.add_argument("--mode").help("mode.").default_value("desktop");
  parser.add_argument("--color_mode").help("color mode.").default_value("sfloat16");
  parser.add_argument("--dither")
      .help("dither.")
      .default_value(false)
      .implicit_value(true);
  parser.add_argument("--max_splats")
      .help("maximum number of splats.")
      .scan<'u', uint32_t>()
      .default_value((std::numeric_limits<uint32_t>::max)());
  parser.add_argument("--radii_levels")
      .help("radii (ratio of horizontal resolution) for each level [0.0, 1.0].")
      .scan<'g', float>()
      .nargs(argparse::nargs_pattern::at_least_one);
  parser.add_argument("--vis_mode")
      .help("visualization mode [normal, overdraw, overdraw_alpha].")
      .default_value("normal");
  parser.add_argument("--vis_scale")
      .help("visualization scale.")
      .scan<'g', float>()  
      .default_value(1.0f);
  parser.add_argument("--res")
      .help("render resolution.")
      .scan<'u', uint32_t>()
      .nargs(2);
  parser.add_argument("--lod_params")
      .help("LOD params.")
      .scan<'g', float>()
      .nargs(4);

  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  vkgs::Config config(
    parser.get<std::string>("config"),
    parser.get<std::string>("mode"),
    parser.get<bool>("dynamic_res"),
    parser.get<bool>("debug"),
    parser.get<std::string>("color_mode"),
    parser.get<bool>("dither"),
    parser.get<uint32_t>("max_splats"),
    parser.get<std::string>("view_file"),
    parser.get<float>("time_budget")
  );

  auto vis_mode = parser.get<std::string>("vis_mode");
  if (vis_mode == "normal") {
    config.vis_mode(vkgs::VisMode::Normal);
  } else if (vis_mode == "overdraw") {
    config.vis_mode(vkgs::VisMode::Overdraw);
  } else if (vis_mode == "overdraw_alpha") {
    config.vis_mode(vkgs::VisMode::OverdrawAlpha);
  }

  config.vis_scale(parser.get<float>("vis_scale"));

  if (parser.is_used("radii_levels")) {
    config.radii_levels(parser.get<std::vector<float>>("radii_levels"));
  }

  if (parser.is_used("res")) {
    auto res = parser.get<std::vector<uint32_t>>("res");
    config.res(res[0], res[1]);
  }

  if (parser.is_used("lod_params")) {
    auto lod_params = parser.get<std::vector<float>>("lod_params");
    config.lod_params(lod_params[0], lod_params[1], lod_params[2], lod_params[3]);
  }
  
  vkgs::EngineAPI engine(config, false);

  if (parser.is_used("input")) {
    auto ply_filepath = parser.get<std::string>("input");
    engine.LoadSplats(ply_filepath);
  }

  engine.Run();

  return 0;
}
