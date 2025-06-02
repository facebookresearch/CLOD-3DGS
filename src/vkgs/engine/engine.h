// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_ENGINE_H
#define VKGS_ENGINE_ENGINE_H

#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vk_radix_sort.h>

#include "vkgs/engine/config.h"
#include "vkgs/engine/gui.h"
#include "vkgs/engine/sample.h"
#include "vkgs/engine/splat_load_thread.h"
#include "vkgs/engine/view_frustum_dataset.h"
#include "vkgs/engine/vulkan/attachment.h"
#include "vkgs/engine/vulkan/benchmark.h"
#include "vkgs/engine/vulkan/buffer.h"
#include "vkgs/engine/vulkan/compositor.h"
#include "vkgs/engine/vulkan/compute_pipeline.h"
#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/cpu_buffer.h"
#include "vkgs/engine/vulkan/framebuffer.h"
#include "vkgs/engine/vulkan/recorder.h"
#include "vkgs/engine/vulkan/render_pass.h"
#include "vkgs/engine/vulkan/shader/uniforms.h"
#include "vkgs/engine/vulkan/structs.h"
#include "vkgs/engine/vulkan/swapchain.h"
#include "vkgs/engine/vulkan/swapchain_desktop.h"
#include "vkgs/engine/vulkan/swapchain_vr.h"
#include "vkgs/engine/vulkan/uniform_buffer.h"
#include "vkgs/engine/vulkan/xr_manager.h"
#include "vkgs/engine/vulkan/mesh/axis_mesh.h"
#include "vkgs/engine/vulkan/mesh/frustum_mesh.h"
#include "vkgs/engine/vulkan/mesh/grid_mesh.h"
#include "vkgs/scene/camera.h"
#include "vkgs/scene/camera_look_at.h"
#include "vkgs/scene/camera_global.h"

#include "foveation/foveated_layers_desktop.h"

namespace vkgs {

class Config;
class Splats;


/**
 * @brief Rendering engine
 */
class Engine {
 friend class GUI;
 friend class vk::XRManager;
 public:
  Engine(Config config, bool enable_validation=false);
  ~Engine();

  void LoadSplats(const std::string& ply_filepath);
  void LoadSplatsAsync(const std::string& ply_filepath);

  void PreparePrimitives();
  void RecreateCompositorFramebuffer();
  void RecreateFramebuffer(uint32_t view_index, uint32_t level, fov::FoveatedLayer& layer);

  void Draw();
  void DrawView(uint32_t view_index, uint32_t frame_index,
                uint32_t acquire_index, uint32_t image_index,
                VkCommandBuffer cb);
  void DrawCompositorPass(VkCommandBuffer cb, uint32_t view_index,
                          uint32_t frame_index, uint32_t image_index,
                          uint32_t width, uint32_t height,
                          VkImageView target_image_view);
  void DrawNormalPass(VkCommandBuffer cb, uint32_t view_index,
                      uint32_t frame_index, uint32_t width, uint32_t height,
                      VkImageView target_image_view, int level);

  void Start();
  void End();

  void Run();
  void Close();

  SampleResult Sample(SampleParams& sample_params, SampleState& state);

  bool SplatsLoaded(uint32_t view_index);

  glm::mat4 GetModelMatrix();
  
  /**
   * @brief Splat render mode (only triangle list supported)
   */
  enum class SplatRenderMode {
    TriangleList,
  };


protected:
  std::atomic_bool terminate_ = false;
  bool enable_validation_ = true;

  std::mutex mutex_;
  std::string pending_ply_filepath_;
  std::string last_loaded_ply_filepath_;

  GLFWwindow* window_ = nullptr;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t num_frames_ = 3;
  uint32_t num_cb_ = 3;
  bool vsync_ = false;

  // budget-based
  bool lazy_budget_mode_ = false;
  float time_budget_ = 1.0;
  std::deque<float> lazy_budget_frame_times_;
  std::vector<float> last_lod_levels_;
  
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkFormat color_format_ = VK_FORMAT_B8G8R8A8_UNORM;
  VkFormat depth_format_ = VK_FORMAT_D32_SFLOAT;
  SplatRenderMode splat_render_mode_ = SplatRenderMode::TriangleList;

  std::shared_ptr<Camera> camera_ = nullptr;

  vk::Context context_;
  std::shared_ptr<vk::Swapchain> swapchain_;

  std::vector<VkCommandBuffer> draw_command_buffers_;    // per frame
  std::vector<VkSemaphore> image_acquired_semaphores_;   // per frame
  std::vector<VkSemaphore> render_finished_semaphores_;  // per frame
  std::vector<VkFence> render_finished_fences_;          // per frame

  vk::DescriptorLayout camera_descriptor_layout_;
  vk::DescriptorLayout gaussian_descriptor_layout_;
  vk::DescriptorLayout instance_layout_;
  vk::DescriptorLayout ply_descriptor_layout_;
  vk::PipelineLayout compute_pipeline_layout_;
  vk::PipelineLayout graphics_pipeline_layout_;

  // preprocess
  vk::ComputePipeline parse_ply_pipeline_;
  vk::ComputePipeline rank_pipeline_;
  vk::ComputePipeline inverse_index_pipeline_;
  vk::ComputePipeline projection_pipeline_;

  // sorter
  std::vector<std::vector<vk::Buffer>> sort_storage_;
  std::vector<std::vector<VrdxSorter>> sorter_;               // per view, per level

  // normal pass
  std::vector<std::vector<vk::Framebuffer>> framebuffer_;      // per view, per level
  std::vector<std::vector<vk::Attachment>> color_attachment_;  // per view, per level
  std::vector<std::vector<vk::Attachment>> depth_attachment_;  // per view, per level
  std::vector<std::vector<vk::RenderPass>> render_passes_;     // per view, per level
  vk::GraphicsPipeline color_line_pipeline_;
  std::vector<vk::GraphicsPipeline> splat_pipelines_;

  // compositor pass
  std::vector<std::vector<vk::Framebuffer>> framebuffer_compositor_;      // per view, per frame
  std::vector<std::vector<vk::Attachment>> depth_compositor_attachment_;  // per view, per frame
  vk::RenderPass render_pass_compositor_;

  std::vector<std::vector<vk::UniformBuffer<vk::shader::Camera>>>
      camera_buffer_;  // per view, per level

  std::shared_ptr<vk::AxisMesh> axis_;
  std::shared_ptr<vk::GridMesh> grid_;
  std::vector<std::shared_ptr<vk::FrustumMesh>> view_frustum_meshes_;

  /**
   * @brief Vulkan descriptor for frames
   */
  struct FrameDescriptor {
    vk::Descriptor camera;
    vk::Descriptor gaussian;
    vk::Descriptor splat_instance;
    vk::Descriptor ply;
  };
  std::vector<std::vector<std::vector<FrameDescriptor>>>
      descriptors_;  // per view, per level, per frame
  std::vector<std::vector<std::vector<vk::FrameInfo>>>
      frame_infos_;  // per view, per level, per frame

  /**
   * @brief Splat storage buffer
   */
  struct SplatStorage {
    vk::Buffer position;  // (N, 3)
    vk::Buffer cov3d;     // (N, 6)
    vk::Buffer opacity;   // (N)
    vk::Buffer sh;        // (N, 3, 16)

    vk::Buffer key;            // (N)
    vk::Buffer index;          // (N)
    vk::Buffer inverse_index;  // (N)

    vk::Buffer instance;  // (N, 10)
  };  // per view, per level
  std::vector<std::vector<SplatStorage>> splat_storage_;  // per view, per level
  static constexpr uint32_t MAX_SPLAT_COUNT = 1 << 23;  // 2^23
  // 2^23 * 3 * 16 * sizeof(float) is already 1.6GB.

  std::vector<std::vector<vk::UniformBuffer<vk::shader::SplatInfo>>>
      splat_info_buffer_;                              // (2), per view, per level
  std::vector<std::vector<vk::Buffer>> splat_visible_point_count_;  // (2), per view, per level
  std::vector<std::vector<vk::Buffer>> splat_draw_indirect_;        // (5), per view, per level

  // config
  Config config_;

  // view frustum data
  std::shared_ptr<ViewFrustumDataset> view_frustum_dataset_ = nullptr;
  
  // create Compositor
  vk::Compositor compositor_;

  // create Benchmark
  bool benchmark_mode_ = false;
  vk::Benchmark benchmark_;

  // create Recorder
  bool recorder_mode_ = false;
  char recorder_path_[128] = {0};
  vk::Recorder recorder_;

  // foveated rendering
  fov::FoveatedLayersDesktopInfo fov_info_;
  fov::FoveatedLayersDesktop fov_layers_;
  std::vector<glm::vec2> center_;

  // current
  glm::vec3 translation_{0.f, 0.f, 0.f};
  glm::quat rotation_{1.f, 0.f, 0.f, 0.f};
  float scale_{1.f};
  
  std::vector<float> lod_levels_;    // per level
  std::vector<float> res_scales_;    // per level

  // [min LOD, max LOD, min distance, max distance]
  std::vector<glm::vec4> lod_params_;  // per level

  GUI gui_;

  std::vector<std::vector<vk::CpuBuffer>> visible_point_count_cpu_buffer_;  // (2) for debug, per view, per level

  std::vector<std::vector<vk::Buffer>> splat_index_buffer_;  // gaussian2d quads, per view, per level

  VkSemaphore transfer_semaphore_ = VK_NULL_HANDLE;
  uint64_t transfer_timeline_ = 0;

  std::vector<std::vector<SplatLoadThread>> splat_load_thread_;  // per view, per level
  std::vector<std::vector<uint32_t>> loaded_point_count_;  // per view, per level
  std::vector<std::vector<bool>> loaded_point_;  // per view, per frame

  // timestamp queries
  static constexpr uint32_t timestamp_count_ = 12;
  std::vector<std::vector<std::vector<VkQueryPool>>>
      timestamp_query_pools_;  // per view, per level, per timestamp

  uint64_t frame_counter_ = 0;

  uint32_t num_views_ = 1;
  uint32_t vis_mode_ = static_cast<int>(VisMode::Normal);
  float vis_mode_scale_ = 1.0f;
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_ENGINE_H
