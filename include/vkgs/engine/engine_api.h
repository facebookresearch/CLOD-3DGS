// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_ENGINE_API_H
#define VKGS_ENGINE_ENGINE_API_H

#include <memory>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace vkgs {
class Config;
class Engine;
class SampleParams;
class SampleState;
class SampleResult;

class EngineAPI {
 public:
  EngineAPI(Config config, bool enable_validation=false);
  ~EngineAPI();

  void LoadSplats(std::string);
  void SplatsLoaded();
  void Start();
  void End();

  void WaitSplatsLoaded();
  
  SampleResult Sample(SampleParams& params, SampleState& state);
  void Run();
  glm::mat4 GetModelMatrix();

 private:
  Engine* engine_;
};

}  // namespace vkgs

#endif