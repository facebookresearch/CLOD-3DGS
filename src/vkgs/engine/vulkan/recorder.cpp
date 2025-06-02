// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/recorder.h"

#include "vk_mem_alloc.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace vkgs {
namespace vk {

Recorder::Recorder() {}

Recorder::~Recorder() {}

void Recorder::saveImage(std::string dir_name, Context& context,
                         const VkImage& image, std::string& save_type) {
  // Adapted from
  // https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp
  if (frame_ >= num_frames_) {
    return;
  }

  vkQueueWaitIdle(context.graphics_queue());
  
  std::string filename;
  if (save_type == "image") {
    if (!std::filesystem::exists(dir_name)) {
      std::filesystem::create_directory(dir_name);
    }
    std::filesystem::path filename_path = dir_name;
    filename_path /= "frame_" + std::to_string(frame_) + ".png";
    filename = filename_path.string();
  }

  // create command buffer
  VkCommandBufferAllocateInfo command_buffer_info = {};
  command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_info.pNext = nullptr;
  command_buffer_info.commandPool = context.command_pool();
  command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_info.commandBufferCount = 1;

  VkCommandBuffer cb;
  vkAllocateCommandBuffers(context.device(), &command_buffer_info, &cb);

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  begin_info.pInheritanceInfo = nullptr;
  vkBeginCommandBuffer(cb, &begin_info);

  // move image from GPU to CPU
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  image_info.flags = 0;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = VK_FORMAT_B8G8R8A8_UNORM;
  image_info.extent = {width_, height_, 1};
  image_info.arrayLayers = 1;
  image_info.mipLevels = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_LINEAR;
  image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.queueFamilyIndexCount = 0;
  image_info.pQueueFamilyIndices = nullptr;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  // create image
  VkImage cpu_image;

  // allocation memory
  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

  VmaAllocation allocation;
  vmaCreateImage(context.allocator(), &image_info, &alloc_info, &cpu_image,
                 &allocation, nullptr);

  // copy images
  VkImageCopy image_copy = {};
  image_copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_copy.srcSubresource.layerCount = 1;
  
  image_copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_copy.dstSubresource.layerCount = 1;
  
  image_copy.extent = {width_, height_, 1};

  barrier::changeImageLayout(
      cb, context, cpu_image, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  barrier::changeImageLayout(
      cb, context, image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_READ_BIT,
      VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT);

  vkCmdCopyImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cpu_image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);

  barrier::changeImageLayout(
      cb, context, cpu_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  barrier::changeImageLayout(
      cb, context, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT,
      VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT);

  vkEndCommandBuffer(cb);

  // submit command buffer
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.pNext = 0;
  VkFence fence;
  vkCreateFence(context.device(), &fence_info, nullptr, &fence);
  vkQueueSubmit(context.graphics_queue(), 1, &submit_info, fence);
  vkWaitForFences(context.device(), 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(context.device(), fence, nullptr);

  vkFreeCommandBuffers(context.device(), context.command_pool(), 1, &cb);

  // read subresources
  VkImageSubresource subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout subresource_layout;
  vkGetImageSubresourceLayout(context.device(), cpu_image, &subresource,
                              &subresource_layout);

  const char* image_data;
  vmaMapMemory(context.allocator(), allocation, (void**)&image_data);
  image_data += subresource_layout.offset;
  
  image_data_buffer_[frame_].resize(subresource_layout.size);
  image_data_[frame_].resize(width_ * height_ * 3);
  memcpy(image_data_buffer_[frame_].data(), image_data,
         subresource_layout.size);

  vmaUnmapMemory(context.allocator(), allocation);
  vmaDestroyImage(context.allocator(), cpu_image, allocation);

  if (save_type == "image") {
    image_save_threads_.push_back(std::thread(
        &Recorder::saveImageThread, this, filename,
        &image_data_buffer_[frame_][0], frame_, subresource_layout.rowPitch));

    image_save_threads_[image_save_threads_.size() - 1].detach();
  } else if (save_type == "array") {
    image_save_threads_.push_back(std::thread(
        &Recorder::saveArrayThread, this,
        &image_data_buffer_[frame_][0], frame_, subresource_layout.rowPitch));

    image_save_threads_[image_save_threads_.size() - 1].detach();
  } else {throw std::invalid_argument("invalid save type");}

  frame_++;
}

void Recorder::saveImageThread(std::string filename, const unsigned char* image_data,
                               uint32_t frame, uint32_t row_pitch) {
  const unsigned char* image_data_addr = image_data;

  // parse memory and save image
  for (uint32_t y = 0; y < height_; y++) {
    unsigned int* color = (unsigned int*)image_data_addr;
    for (uint32_t x = 0; x < width_; x++) {
      int pixel_index = y * width_ + x;
      image_data_[frame][(pixel_index * 3) + 0] = (*color >> 16) & 0xFF;
      image_data_[frame][(pixel_index * 3) + 1] = (*color >> 8) & 0xFF;
      image_data_[frame][(pixel_index * 3) + 2] = (*color >> 0) & 0xFF;
      color++;
    }
    image_data_addr += row_pitch;
  }

  stbi_write_png(filename.c_str(), (int)width_, (int)height_, 3,
                 (const void*)&image_data_[frame].data()[0], (int)(width_ * 3));
  num_saved_images_++;
}

void Recorder::saveArrayThread(const unsigned char* image_data, uint32_t frame,
                               uint32_t row_pitch) {
  const unsigned char* image_data_addr = image_data;

  // parse memory and save image
  for (uint32_t y = 0; y < height_; y++) {
    unsigned int* color = (unsigned int*)image_data_addr;
    for (uint32_t x = 0; x < width_; x++) {
      int pixel_index = y * width_ + x;
      image_data_[frame][(pixel_index * 3) + 0] = (*color >> 16) & 0xFF;
      image_data_[frame][(pixel_index * 3) + 1] = (*color >> 8) & 0xFF;
      image_data_[frame][(pixel_index * 3) + 2] = (*color >> 0) & 0xFF;
      color++;
    }
    image_data_addr += row_pitch;
  }

  num_saved_images_++;
}

uint32_t Recorder::frame() { return frame_; }

void Recorder::resetSaving(uint32_t width, uint32_t height) {
  frame_ = 0;
  width_ = width;
  height_ = height;

  image_data_.clear();
  image_data_buffer_.clear();
  image_data_.resize(num_frames_);
  image_data_buffer_.resize(num_frames_);

  image_save_threads_.clear();
  num_saved_images_.exchange(0);
  num_saved_images_.exchange(0);
  ready_to_save_ = false;
}
bool Recorder::isRenderingDone() { return (frame_ >= num_frames_); }
bool Recorder::isSavingDone() {
  if (num_saved_images_ >= num_frames_) {
    ready_to_save_ = true;
  }
  return ready_to_save_;
}

std::vector<std::vector<unsigned char>>& Recorder::image_data() {
  return image_data_;
}

const uint32_t Recorder::num_frames() const { return num_frames_; }
const uint32_t Recorder::num_frames(uint32_t num) {
  num_frames_ = num;
  
  image_data_.clear();
  image_data_buffer_.clear();
  image_data_.resize(num_frames_);
  image_data_buffer_.resize(num_frames_);

  return num_frames_;
};

}  // namespace vk
}  // namespace vkgs
