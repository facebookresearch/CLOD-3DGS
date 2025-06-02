cmake_minimum_required(VERSION 3.28)
include(FetchContent)

project(vkgs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)
add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
add_subdirectory(third_party/vulkan_radix_sort EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glm EXCLUDE_FROM_ALL)
add_subdirectory(third_party/argparse EXCLUDE_FROM_ALL)
add_subdirectory(third_party/yaml-cpp EXCLUDE_FROM_ALL)
include_directories(third_party/stb)

# OpenXR
add_library(OpenXR UNKNOWN IMPORTED)
set_target_properties(OpenXR PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/OpenXR/include"
  IMPORTED_LOCATION_RELEASE "${CMAKE_SOURCE_DIR}/third_party/OpenXR/bin/Release/openxr_loader.lib"
  IMPORTED_LOCATION_DEBUG "${CMAKE_SOURCE_DIR}/third_party/OpenXR/bin/Debug/openxr_loaderd.lib"
)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# add_shader(TARGET SHADER OUTPUT DEFINE...)
function(add_shader)
  list(POP_FRONT ARGV TARGET SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
    COMMAND
      ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
      --target-env spirv1.5
      -V
      --vn ${OUTPUT}
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    DEPENDS ${SHADER}
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
  )

  add_custom_target(${OUTPUT} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h)
  add_dependencies(${TARGET} ${OUTPUT})
endfunction()

# vkgs
add_library(vkgs SHARED
  src/vkgs/scene/camera.cpp
  src/vkgs/scene/camera_global.cpp
  src/vkgs/scene/camera_look_at.cpp
  src/vkgs/engine/config.cpp
  src/vkgs/engine/engine.cc
  src/vkgs/engine/engine_api.cpp
  src/vkgs/engine/gui.cpp
  src/vkgs/engine/sample.cpp
  src/vkgs/engine/splat_load_thread.cc
  src/vkgs/engine/view_frustum_dataset.cpp
  src/vkgs/engine/utils/math.cpp
  src/vkgs/engine/vulkan/attachment.cc
  src/vkgs/engine/vulkan/barrier.cpp
  src/vkgs/engine/vulkan/benchmark.cpp
  src/vkgs/engine/vulkan/buffer.cc
  src/vkgs/engine/vulkan/compositor.cpp
  src/vkgs/engine/vulkan/compute_pipeline.cc
  src/vkgs/engine/vulkan/context.cc
  src/vkgs/engine/vulkan/cpu_buffer.cc
  src/vkgs/engine/vulkan/debug.cpp
  src/vkgs/engine/vulkan/descriptor_layout.cc
  src/vkgs/engine/vulkan/descriptor.cc
  src/vkgs/engine/vulkan/framebuffer.cc
  src/vkgs/engine/vulkan/graphics_pipeline.cc
  src/vkgs/engine/vulkan/pipeline_layout.cc
  src/vkgs/engine/vulkan/recorder.cpp
  src/vkgs/engine/vulkan/render_pass.cc
  src/vkgs/engine/vulkan/swapchain.cpp
  src/vkgs/engine/vulkan/swapchain_desktop.cpp
  src/vkgs/engine/vulkan/swapchain_vr.cpp
  src/vkgs/engine/vulkan/uniform_buffer.cc
  src/vkgs/engine/vulkan/utils_io.cpp
  src/vkgs/engine/vulkan/vma_impl.cc
  src/vkgs/engine/vulkan/xr_input_manager.cpp
  src/vkgs/engine/vulkan/xr_manager.cpp
  src/vkgs/engine/vulkan/mesh/axis_mesh.cpp
  src/vkgs/engine/vulkan/mesh/frustum_mesh.cpp
  src/vkgs/engine/vulkan/mesh/grid_mesh.cpp
  src/vkgs/engine/vulkan/mesh/mesh.cpp

  # imgui
  third_party/imgui/backends/imgui_impl_glfw.cpp
  third_party/imgui/backends/imgui_impl_vulkan.cpp
  third_party/imgui/imgui.cpp
  third_party/imgui/imgui_draw.cpp
  third_party/imgui/imgui_demo.cpp
  third_party/imgui/imgui_tables.cpp
  third_party/imgui/imgui_widgets.cpp

  # implot
  third_party/implot/implot.cpp
  third_party/implot/implot_items.cpp
)

target_include_directories(vkgs
  PUBLIC
    include
  PRIVATE
    src
    third_party/imgui
    third_party/imgui/backends
    third_party/implot
    third_party/yaml-cpp/include
)

# core
add_library(core SHARED
  src/core/structs.cpp
  src/core/string.cpp
)

target_include_directories(core
  PUBLIC
    include
    src
)

target_link_libraries(core
  PUBLIC glm
)

# foveation
add_library(foveation SHARED
  src/foveation/foveated_layers.cpp
  src/foveation/foveated_layers_desktop.cpp
  src/foveation/math.cpp
)

target_include_directories(foveation
  PUBLIC include
  PRIVATE
    src
)

target_link_libraries(foveation
  PUBLIC glm
)

# eye tracker
add_library(eye_tracker SHARED
  src/eye_tracker/eye_tracker.cpp
  src/eye_tracker/eye_tracker_openxr.cpp
)

target_include_directories(eye_tracker
  PUBLIC include
  PRIVATE
    src
)

target_link_libraries(eye_tracker
  PUBLIC
    glm
    core
  PRIVATE
    OpenXR
)

# vkgs
target_link_libraries(vkgs
  PUBLIC
    glm
    core
  PRIVATE
    Vulkan::Vulkan
    VulkanMemoryAllocator
    vk_radix_sort
    glfw
    foveation
    eye_tracker
    yaml-cpp
    OpenXR
)

make_directory("${CMAKE_BINARY_DIR}/Release/")
make_directory("${CMAKE_BINARY_DIR}/Debug/")

add_custom_target(OpenXR_binary
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/third_party/OpenXR/bin/Release/openxr_loader.dll" ${CMAKE_BINARY_DIR}/Release
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/third_party/OpenXR/bin/Debug/openxr_loaderd.dll" ${CMAKE_BINARY_DIR}/Debug
)

add_shader(vkgs src/shader/color.vert color_vert)
add_shader(vkgs src/shader/color.frag color_frag)
add_shader(vkgs src/shader/inverse_index.comp inverse_index_comp)
add_shader(vkgs src/shader/parse_ply.comp parse_ply_comp)
add_shader(vkgs src/shader/projection.comp projection_comp)
add_shader(vkgs src/shader/rank.comp rank_comp)
add_shader(vkgs src/shader/splat.vert splat_vert)
add_shader(vkgs src/shader/splat.frag splat_frag)
add_shader(vkgs src/shader/compositor.vert compositor_vert)
add_shader(vkgs src/shader/compositor.frag compositor_frag)

# executable
add_executable(vkgs_viewer examples/vkgs_viewer.cc)
set_target_properties(vkgs_viewer PROPERTIES CXX_STANDARD 17)
target_link_libraries(vkgs_viewer
  PUBLIC vkgs
  PRIVATE argparse
)

add_dependencies(vkgs_viewer OpenXR_binary)

# executable
add_executable(vkgs_train examples/vkgs_train.cpp)
target_link_libraries(vkgs_train
  PUBLIC
    vkgs
    core
  PRIVATE
   argparse
)

add_dependencies(vkgs_train OpenXR_binary)

# vkgs Python binding
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(third_party/nanobind)

nanobind_build_library(nanobind SHARED)
add_library(vkgs_py MODULE src/vkgs/optimization/vkgs_py.cpp)
target_link_libraries(vkgs_py
  PUBLIC
    vkgs
    core
  PRIVATE
    nanobind
)
nanobind_opt_size(vkgs_py)
nanobind_lto(vkgs_py)
nanobind_set_visibility(vkgs_py)
nanobind_strip(vkgs_py)
nanobind_disable_stack_protector(vkgs_py)
nanobind_extension(vkgs_py)
nanobind_compile_options(vkgs_py)
nanobind_link_options(vkgs_py)
nanobind_musl_static_libcpp(vkgs_py)
