// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vkgs/engine/engine_api.h>
#include "../src/vkgs/engine/config.h"
#include "../src/vkgs/engine/sample.h"
#include "../src/core/structs.h"


namespace nb = nanobind;

nb::ndarray<nb::numpy, uint8_t> get_sample_result_numpy(vkgs::SampleResult& sample_result) {
  auto shape = sample_result.shape;
  auto output = nb::ndarray<nb::numpy, uint8_t>(
      (uint8_t*)sample_result.data.data(),
      {shape[0], shape[1], shape[2], shape[3]},
    nb::handle()
  );
  return output;
}


nb::ndarray<nb::numpy, float> mat4_to_numpy(glm::mat4& mat) {
  auto output = nb::ndarray<nb::numpy, float>(
      (float*)glm::value_ptr(mat), {4, 4}, nb::handle());
  return output;
}


NB_MODULE(vkgs_py, m) {
  nb::class_<core::ViewFrustumAngles>(m, "ViewFrustumAngles")
      .def(nb::init<>())
      .def(nb::init<float, float, float, float>())
      .def_rw("angle_right", &core::ViewFrustumAngles::angle_right)
      .def_rw("angle_left", &core::ViewFrustumAngles::angle_left)
      .def_rw("angle_down", &core::ViewFrustumAngles::angle_down)
      .def_rw("angle_up", &core::ViewFrustumAngles::angle_up);

  nb::class_<glm::vec2>(m, "vec2")
      .def(nb::init<float>())
      .def(nb::init<float, float>())
      .def_rw("x", &glm::vec2::x)
      .def_rw("y", &glm::vec2::y);

  nb::class_<glm::vec3>(m, "vec3")
      .def(nb::init<float>())  
      .def(nb::init<float, float, float>())
      .def_rw("x", &glm::vec3::x)
      .def_rw("y", &glm::vec3::y)
      .def_rw("z", &glm::vec3::z);

  nb::class_<glm::vec4>(m, "vec4")
      .def(nb::init<float>())
      .def(nb::init<float, float, float, float>())
      .def_rw("x", &glm::vec4::x)
      .def_rw("y", &glm::vec4::y)
      .def_rw("z", &glm::vec4::z)
      .def_rw("w", &glm::vec4::w);

  nb::class_<glm::quat>(m, "quat")
      .def(nb::init<float, float, float, float>())
      .def_rw("x", &glm::quat::x)
      .def_rw("y", &glm::quat::y)
      .def_rw("z", &glm::quat::z)
      .def_rw("w", &glm::quat::w);

  nb::class_<glm::mat4>(m, "mat4")
    .def(nb::init<float>())
    .def(nb::init<
      float, float, float, float,
      float, float, float, float,
      float, float, float, float,
      float, float, float, float
    >());

  m.def("lookAt", [](glm::vec3& eye, glm::vec3& center, glm::vec3& up) {
    return glm::mat4(glm::lookAt(eye, center, up));
  });

  m.def("inverse", [](glm::mat4& mat) {
    return glm::inverse(mat);
  });

  m.def("quat_cast", [](glm::mat4& matrix) {
    return glm::quat_cast(matrix);
  });

  nb::class_<vkgs::SampleParams>(m, "SampleParams")
      .def(nb::init<>())
      .def_rw("num_frames_benchmark", &vkgs::SampleParams::num_frames_benchmark)
      .def_rw("num_frames_recorder", &vkgs::SampleParams::num_frames_recorder)
      .def_rw("lod", &vkgs::SampleParams::lod)
      .def_rw("res", &vkgs::SampleParams::res)
      .def_rw("lod_params", &vkgs::SampleParams::lod_params);

  nb::class_<vkgs::SampleState>(m, "SampleState")
      .def(nb::init<>())
      .def_rw("pos", &vkgs::SampleState::pos)
      .def_rw("quat", &vkgs::SampleState::quat)
      .def_rw("center", &vkgs::SampleState::center)
      .def_rw("gaze_dir", &vkgs::SampleState::gaze_dir)
      .def_rw("view_angles", &vkgs::SampleState::view_angles);

  nb::class_<vkgs::SampleResult>(m, "SampleResult")
      .def(nb::init<>())
      .def_rw("time", &vkgs::SampleResult::time)
      .def_rw("shape", &vkgs::SampleResult::shape);

  m.def("get_sample_result_numpy", &get_sample_result_numpy);

  m.def("mat4_to_numpy", &mat4_to_numpy);

  nb::class_<vkgs::Config>(m, "Config")
      .def(nb::init<std::string&, std::string&>())
      .def(nb::init<std::string&, std::string&, bool>())
      .def(nb::init<std::string&, std::string&, bool, bool>())
      .def("num_levels", &vkgs::Config::num_levels)
      .def("dynamic_res", nb::overload_cast<>(&vkgs::Config::dynamic_res))
      .def("dynamic_res", nb::overload_cast<bool>(&vkgs::Config::dynamic_res))
      .def("debug", nb::overload_cast<>(&vkgs::Config::debug))
      .def("debug", nb::overload_cast<const bool>(&vkgs::Config::debug))
      .def("res", nb::overload_cast<>(&vkgs::Config::res))
      .def("res", nb::overload_cast<uint32_t, uint32_t>(&vkgs::Config::res))
      .def("num_frames_benchmark", nb::overload_cast<>(&vkgs::Config::num_frames_benchmark))
      .def("num_frames_benchmark", nb::overload_cast<uint32_t>(& vkgs::Config::num_frames_benchmark))
      .def("num_frames_recorder", nb::overload_cast<>(&vkgs::Config::num_frames_recorder))
      .def("num_frames_recorder", nb::overload_cast<uint32_t>(&vkgs::Config::num_frames_recorder))
      .def("vis_mode", nb::overload_cast<>(&vkgs::Config::vis_mode))
      .def("vis_mode", nb::overload_cast<std::string&>(&vkgs::Config::vis_mode))
      .def("vis_scale", nb::overload_cast<>(&vkgs::Config::vis_scale))
      .def("vis_scale", nb::overload_cast<float>(&vkgs::Config::vis_scale))
      .def("radii_levels", nb::overload_cast<>(&vkgs::Config::radii_levels))
      .def("radii_levels", nb::overload_cast<std::vector<float>&>(&vkgs::Config::radii_levels));

  nb::class_<vkgs::EngineAPI>(m, "Engine")
      .def(nb::init<vkgs::Config>())
      .def(nb::init<vkgs::Config, bool>())
      .def("load_splats", &vkgs::EngineAPI::LoadSplats)
      .def("start", &vkgs::EngineAPI::Start)
      .def("end", &vkgs::EngineAPI::End)
      .def("sample", &vkgs::EngineAPI::Sample)
      .def("run", &vkgs::EngineAPI::Run)
      .def("get_model_matrix", &vkgs::EngineAPI::GetModelMatrix);

  nb::enum_<vkgs::VisMode>(m, "VisMode")
      .value("Normal", vkgs::VisMode::Normal)
      .value("Overdraw", vkgs::VisMode::Overdraw)
      .value("OverdrawAlpha", vkgs::VisMode::OverdrawAlpha)
      .export_values();
}