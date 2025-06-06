// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/splat_load_thread.h"

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstring>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

namespace vkgs {

/**
  * @brief Implementation of SplatLoadThread
  */
class SplatLoadThread::Impl {
 public:
  Impl() = delete;

  Impl(vk::Context context) : context_(context) {
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(context_.device(), &fence_info, NULL, &fence_);

    thread_ = std::thread([this] {
      // thread-local command buffer
      VkCommandPoolCreateInfo command_pool_info = {
          VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      command_pool_info.queueFamilyIndex =
          context_.transfer_queue_family_index();
      VkCommandPool command_pool = VK_NULL_HANDLE;
      vkCreateCommandPool(context_.device(), &command_pool_info, NULL,
                          &command_pool);

      VkCommandBufferAllocateInfo command_buffer_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      command_buffer_info.commandPool = command_pool;
      command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_buffer_info.commandBufferCount = 1;
      VkCommandBuffer cb = VK_NULL_HANDLE;
      vkAllocateCommandBuffers(context_.device(), &command_buffer_info, &cb);

      while (true) {
        std::string ply_filepath;
        {
          std::unique_lock<std::mutex> guard{mutex_};
          cv_.wait(guard,
                   [this] { return terminate_ || !ply_filepath_.empty(); });
          if (terminate_) break;
          ply_filepath = std::move(ply_filepath_);
        }

        cancel_ = false;

        std::ifstream in(ply_filepath, std::ios::binary);

        // parse header
        std::unordered_map<std::string, int> offsets;
        int offset = 0;
        uint32_t point_count = 0;
        std::string line;
        while (std::getline(in, line)) {
          if (line == "end_header") break;

          std::istringstream iss(line);
          std::string word;
          iss >> word;
          if (word == "property") {
            int size = 0;
            std::string type, property;
            iss >> type >> property;
            if (type == "float") {
              size = 4;
            }
            offsets[property] = offset;
            offset += size;
          } else if (word == "element") {
            std::string type;
            size_t count;
            iss >> type >> count;
            if (type == "vertex") {
              point_count = count;
            }
          }
        }

        // update total point count
        {
          std::unique_lock<std::mutex> guard{mutex_};
          point_count = std::min(point_count, max_splats_);
          total_point_count_ = point_count;
        }

        // assuming all properties are float
        VkDeviceSize size = 60 * sizeof(uint32_t) +
                            static_cast<VkDeviceSize>(offset) * point_count;
        if (staging_size_ < size) {
          if (staging_)
            vmaDestroyBuffer(context_.allocator(), staging_, allocation_);

          VkBufferCreateInfo buffer_info = {
              VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
          buffer_info.size = size;
          buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
          VmaAllocationCreateInfo allocation_create_info = {};
          allocation_create_info.flags =
              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
              VMA_ALLOCATION_CREATE_MAPPED_BIT;
          allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
          VmaAllocationInfo allocation_info;
          vmaCreateBuffer(context_.allocator(), &buffer_info,
                          &allocation_create_info, &staging_, &allocation_,
                          &allocation_info);
          staging_map_ =
              reinterpret_cast<uint8_t*>(allocation_info.pMappedData);
          staging_size_ = size;
        }

        // TODO: make GPU buffer persist. Buffer creation is expensive.
        auto ply_buffer = vk::Buffer(context_, size,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        // ply offsets
        std::vector<uint32_t> ply_offsets(60);
        ply_offsets[0] = offsets["x"] / 4;
        ply_offsets[1] = offsets["y"] / 4;
        ply_offsets[2] = offsets["z"] / 4;
        ply_offsets[3] = offsets["scale_0"] / 4;
        ply_offsets[4] = offsets["scale_1"] / 4;
        ply_offsets[5] = offsets["scale_2"] / 4;
        ply_offsets[6] = offsets["rot_1"] / 4;  // qx
        ply_offsets[7] = offsets["rot_2"] / 4;  // qy
        ply_offsets[8] = offsets["rot_3"] / 4;  // qz
        ply_offsets[9] = offsets["rot_0"] / 4;  // qw
        ply_offsets[10 + 0] = offsets["f_dc_0"] / 4;
        ply_offsets[10 + 16] = offsets["f_dc_1"] / 4;
        ply_offsets[10 + 32] = offsets["f_dc_2"] / 4;
        for (int i = 0; i < 15; ++i) {
          ply_offsets[10 + 1 + i] = offsets["f_rest_" + std::to_string(i)] / 4;
          ply_offsets[10 + 17 + i] =
              offsets["f_rest_" + std::to_string(15 + i)] / 4;
          ply_offsets[10 + 33 + i] =
              offsets["f_rest_" + std::to_string(30 + i)] / 4;
        }
        ply_offsets[58] = offsets["opacity"] / 4;
        ply_offsets[59] = offset / 4;

        // read all binary data
        buffer_.resize(offset * point_count);

        constexpr uint32_t chunk_size = 65536;
        for (uint32_t start = 0; start < point_count; start += chunk_size) {
          if (terminate_ || cancel_) break;

          auto chunk_point_count =
              std::min<uint32_t>(chunk_size, point_count - start);
          in.read(buffer_.data() + offset * start, offset * chunk_point_count);

          {
            std::unique_lock<std::mutex> guard{mutex_};
            loaded_point_count_ = start + chunk_point_count;
          }
        }

        if (terminate_) break;

        // copy to staging buffer
        {
          std::memcpy(staging_map_, ply_offsets.data(),
                      ply_offsets.size() * sizeof(uint32_t));
          std::memcpy(staging_map_ + 60 * sizeof(uint32_t), buffer_.data(),
                      buffer_.size() * sizeof(char));
        }

        // transfer command
        VkCommandBufferBeginInfo begin_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &begin_info);

        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size = size;
        vkCmdCopyBuffer(cb, staging_, ply_buffer, 1, &region);

        // transfer ownership
        std::vector<VkBufferMemoryBarrier> buffer_barriers(1);
        buffer_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        buffer_barriers[0].srcQueueFamilyIndex =
            context_.transfer_queue_family_index();
        buffer_barriers[0].dstQueueFamilyIndex =
            context_.graphics_queue_family_index();
        buffer_barriers[0].buffer = ply_buffer;
        buffer_barriers[0].offset = 0;
        buffer_barriers[0].size = size;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL,
                             buffer_barriers.size(), buffer_barriers.data(), 0,
                             NULL);

        vkEndCommandBuffer(cb);

        // submit
        {
          VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
          submit_info.commandBufferCount = 1;
          submit_info.pCommandBuffers = &cb;
          vkQueueSubmit(context_.transfer_queue(), 1, &submit_info, fence_);

          vkWaitForFences(context_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
          vkResetFences(context_.device(), 1, &fence_);
        }

        // update loaded point count
        {
          std::unique_lock<std::mutex> guard{mutex_};
          buffer_barriers_.insert(buffer_barriers_.end(),
                                  buffer_barriers.begin(),
                                  buffer_barriers.end());
          buffer_barriers.clear();
          ply_buffer_ = ply_buffer;
        }
      }

      vkDestroyCommandPool(context_.device(), command_pool, NULL);
    });
  }

  ~Impl() {
    if (thread_.joinable()) {
      cancel_ = true;
      terminate_ = true;
      cv_.notify_one();
      thread_.join();
    }

    vkDestroyFence(context_.device(), fence_, NULL);
    if (staging_) vmaDestroyBuffer(context_.allocator(), staging_, allocation_);
  }

  void Start(const std::string& ply_filepath,
             uint32_t max_splats = (std::numeric_limits<uint32_t>::max)()) {
    max_splats_ = max_splats;
    {
      std::unique_lock<std::mutex> guard{mutex_};
      total_point_count_ = 0;
      loaded_point_count_ = 0;
      ply_filepath_ = ply_filepath;
    }
    cv_.notify_one();
  }

  Progress GetProgress() {
    Progress result;
    std::unique_lock<std::mutex> guard{mutex_};
    result.total_point_count = total_point_count_;
    result.loaded_point_count = loaded_point_count_;
    result.ply_buffer = ply_buffer_;
    result.buffer_barriers = std::move(buffer_barriers_);

    //ply_buffer_ = {};

    return result;
  }

  void Cancel() { cancel_ = true; }

 private:
  vk::Context context_;

  std::thread thread_;
  std::mutex mutex_;
  std::atomic_bool terminate_ = false;
  std::atomic_bool cancel_ = false;
  std::condition_variable cv_;

  std::string ply_filepath_;

  uint32_t total_point_count_ = 0;
  uint32_t loaded_point_count_ = 0;
  std::vector<VkBufferMemoryBarrier> buffer_barriers_;

  VkFence fence_ = VK_NULL_HANDLE;

  // position: (N, 3), cov3d: (N, 6), sh: (N, 48), opacity: (N).
  // staging: (N, 58)
  std::vector<char> buffer_;
  VkDeviceSize staging_size_ = 0;
  VkBuffer staging_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  uint8_t* staging_map_ = nullptr;

  vk::Buffer ply_buffer_;
  uint32_t max_splats_ = 0;
};

SplatLoadThread::SplatLoadThread() = default;

SplatLoadThread::SplatLoadThread(vk::Context context)
    : impl_(std::make_shared<Impl>(context)) {}

SplatLoadThread::~SplatLoadThread() = default;

void SplatLoadThread::Start(
    const std::string& ply_filepath,
    uint32_t max_splats) {
  impl_->Start(ply_filepath, max_splats);
}

SplatLoadThread::Progress SplatLoadThread::GetProgress() {
  return impl_->GetProgress();
}

void SplatLoadThread::Cancel() { impl_->Cancel(); }

}  // namespace vkgs
