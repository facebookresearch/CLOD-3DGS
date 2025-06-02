// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <vkgs/engine/engine_api.h>
#include <vkgs/engine/engine.h>

namespace vkgs {

EngineAPI::EngineAPI(Config config, bool enable_validation) {
  engine_ = new Engine(config, enable_validation);
}

EngineAPI::~EngineAPI() {
  delete engine_;
}

void EngineAPI::LoadSplats(std::string filepath) {
  engine_->LoadSplats(filepath);
}

void EngineAPI::Start() {
  engine_->Start();
}

void EngineAPI::End() {
  engine_->End();
}

SampleResult EngineAPI::Sample(SampleParams& params, SampleState& state) {
  return engine_->Sample(params, state);
}

void EngineAPI::Run() {
  engine_->Run();
}

glm::mat4 EngineAPI::GetModelMatrix() {
  return engine_->GetModelMatrix();
}

}  // namespace vkgs