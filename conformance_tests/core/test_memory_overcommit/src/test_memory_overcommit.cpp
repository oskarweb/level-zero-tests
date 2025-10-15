/*
 *
 * Copyright (C) 2019-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "gtest/gtest.h"

#include "utils/utils.hpp"
#include "test_harness/test_harness.hpp"
#include "logging/logging.hpp"
#include "random/random.hpp"

namespace lzt = level_zero_tests;

#include <level_zero/ze_api.h>

namespace {

struct DriverInfo {
  ze_driver_handle_t driver_handle;
  std::vector<ze_device_handle_t> device_handles{};
  std::vector<ze_device_properties_t> device_properties{};
  std::vector<ze_device_compute_properties_t> device_compute_properties{};
  std::vector<std::vector<ze_device_memory_properties_t>>
      device_memory_properties{};
  /*
   * sharedSystemAllocCapabilities
   * 	ZE_MEMORY_ACCESS_NONE, ZE_MEMORY_ACCESS, ZE_MEMORY_ATOMIC_ACCESS
   * 	ZE_MEMORY_CONCURENT_ACCESS, ZE_MEMORY_CONCURRENT_ATOMIC_ACCESS
   *
   * 	There is one ze_device_memory_access_properties_t per device handle
   */
  std::vector<ze_device_memory_access_properties_t>
      device_memory_access_properties{};

  DriverInfo(ze_driver_handle_t driver_handle) : driver_handle(driver_handle) {}
};

static int count = 0;

struct UsmDeleter {
  ze_context_handle_t ctx{};
  bool system_alloc{}; 
  void operator()(uint64_t *p) const noexcept {
    if (!p)
      return;
    lzt::free_memory_with_allocator_selector(ctx, p, system_alloc);
    //count--;
    //LOG_INFO << count;
  }
};

inline std::unique_ptr<uint64_t, UsmDeleter>
alloc_device_usm(ze_context_handle_t ctx, uint32_t ordinal,
                                        ze_device_handle_t device, size_t size) {
  return std::unique_ptr<uint64_t, UsmDeleter>(
      static_cast<uint64_t *>(
          lzt::allocate_device_memory(size, 8, 0, ordinal, device, ctx)),
      UsmDeleter{ctx, false});
}

inline std::unique_ptr<uint64_t, UsmDeleter>
alloc_shared_usm(ze_context_handle_t ctx, ze_device_handle_t device, size_t size) {
  //count++;
  //LOG_INFO << count;
  return std::unique_ptr<uint64_t, UsmDeleter>(
    static_cast<uint64_t *>(
          lzt::allocate_shared_memory(size, 8, 0, 0, device, ctx)),
      UsmDeleter{ctx, false});
}

inline std::unique_ptr<uint64_t, UsmDeleter>
alloc_system_usm (ze_context_handle_t ctx, size_t size) {
  return std::unique_ptr<uint64_t, UsmDeleter>(new uint64_t[size / sizeof(uint64_t)],
      UsmDeleter{ctx, true});
}

enum class BufferType { Device, Shared, SharedSystem };

struct GpuBuffer {
  uint64_t *data;
};

struct MemoryOvercommitData {
  BufferType buffer_type = BufferType::Shared;

  std::vector<std::unique_ptr<uint64_t, UsmDeleter>> gpu_stress_buffers;
  std::vector<std::unique_ptr<uint64_t, UsmDeleter>> gpu_verify_buffers;
  
  std::unique_ptr<uint64_t, UsmDeleter> gpu_stress_ptrs_buffer;
  std::unique_ptr<uint64_t, UsmDeleter> gpu_verify_ptrs_buffer;
  
  std::vector<GpuBuffer> host_stress_buffers;
  std::vector<GpuBuffer> host_verify_buffers;

  std::unique_ptr<uint64_t[]> host_found_output_buffer;

  uint32_t output_count = 64u;
  size_t output_size = output_count * sizeof(uint64_t);

  uint64_t num_stress_allocs = 0;
  uint64_t stress_alloc_size = 0;

  uint64_t num_verify_allocs = 0;
  uint64_t verify_alloc_size = 0;

  void initialize() {
    LOG_INFO << num_stress_allocs << " " << stress_alloc_size;
    host_found_output_buffer =
        std::make_unique<uint64_t[]>(num_verify_allocs * 16);
      switch (buffer_type) {
      case BufferType::Device:
        for (uint64_t i = 0; i < num_stress_allocs; ++i) {
          gpu_stress_buffers.push_back(std::move(alloc_device_usm(
              context_, device_ordinal_,
                                          device_handle_,
                                 stress_alloc_size)));
        }
        for (uint64_t i = 0; i < num_verify_allocs; ++i) {
          gpu_verify_buffers.push_back(std::move(alloc_device_usm(
              context_, device_ordinal_, device_handle_, verify_alloc_size)));
        }
        gpu_stress_ptrs_buffer =
            alloc_device_usm(context_, device_ordinal_, device_handle_,
                             num_stress_allocs * sizeof(GpuBuffer));
        gpu_verify_ptrs_buffer =
            alloc_device_usm(context_, device_ordinal_, device_handle_,
                             num_verify_allocs * sizeof(GpuBuffer));
        break;
      case BufferType::Shared:
        for (uint64_t i = 0; i < num_stress_allocs; ++i) {
          gpu_stress_buffers.push_back(std::move(
              alloc_shared_usm(context_, device_handle_, stress_alloc_size)));
          std::memset(gpu_stress_buffers.back().get(), 0, stress_alloc_size);
        }
        for (uint64_t i = 0; i < num_verify_allocs; ++i) {
          gpu_verify_buffers.push_back(std::move(
              alloc_shared_usm(context_, device_handle_, verify_alloc_size)));
          std::memset(gpu_verify_buffers.back().get(), 0, verify_alloc_size);
        }
        gpu_stress_ptrs_buffer = alloc_shared_usm(
            context_, device_handle_, num_stress_allocs * sizeof(GpuBuffer));
        gpu_verify_ptrs_buffer = alloc_shared_usm(
            context_, device_handle_, num_verify_allocs * sizeof(GpuBuffer));
        break;
      case BufferType::SharedSystem:
        for (uint64_t i = 0; i < num_stress_allocs; ++i) {
          gpu_stress_buffers.push_back(
              std::move(alloc_system_usm(context_, stress_alloc_size)));
        }
        for (uint64_t i = 0; i < num_verify_allocs; ++i) {
          gpu_verify_buffers.push_back(
              std::move(alloc_system_usm(context_, verify_alloc_size)));
        }
        gpu_stress_ptrs_buffer =
            alloc_system_usm(context_, num_stress_allocs * sizeof(GpuBuffer));
        gpu_verify_ptrs_buffer =
            alloc_system_usm(context_, num_verify_allocs * sizeof(GpuBuffer));
        break;
      }
      host_stress_buffers.resize(num_stress_allocs);
      host_verify_buffers.resize(num_verify_allocs);
      for (uint64_t i = 0; i < num_stress_allocs; ++i) {
        host_stress_buffers[i].data = gpu_stress_buffers[i].get();
      }
      for (uint64_t i = 0; i < num_verify_allocs; ++i) {
        host_verify_buffers[i].data = gpu_verify_buffers[i].get();
      }
      memcpy(gpu_stress_ptrs_buffer.get(), host_stress_buffers.data(),
             num_stress_allocs * sizeof(GpuBuffer));
      memcpy(gpu_verify_ptrs_buffer.get(), host_verify_buffers.data(),
             num_verify_allocs * sizeof(GpuBuffer));
  }

  void cleanup() {
    gpu_stress_buffers.clear();
    gpu_verify_buffers.clear();
    gpu_stress_ptrs_buffer.reset();
    gpu_verify_ptrs_buffer.reset();
    host_stress_buffers.clear();
    host_verify_buffers.clear();
    host_found_output_buffer.reset();
  }

  ze_device_memory_access_properties_t device_memory_access_cap_{};
  ze_context_handle_t context_ = nullptr;
  ze_device_handle_t device_handle_ = nullptr;
  uint32_t device_ordinal_ = 0u;
};

struct MemoryOvercommitWorkload {
  MemoryOvercommitWorkload(
                           uint32_t device_ordinal,
                           std::unique_ptr<MemoryOvercommitData> data)
      : 
        device_ordinal_(device_ordinal),
        data(std::move(data)) {}
  std::unique_ptr<MemoryOvercommitData> data;

  virtual const char *get_module_name() {
    return "test_fill_device_memory_overcommit_indirect.spv";
  }

  void initialize(const DriverInfo &driver_info, size_t memory_size_multiple) {
    driver_handle_ = driver_info.driver_handle;
    context_ = lzt::create_context(driver_handle_);
    device_handle_ = driver_info.device_handles[device_ordinal_];
    module_handle_ =
        lzt::create_module(context_, device_handle_, get_module_name(),
                           ZE_MODULE_FORMAT_IL_SPIRV, nullptr, nullptr);
    data->context_ = context_;
    data->device_handle_ = device_handle_;
    data->device_ordinal_ = device_ordinal_;
    data->device_memory_access_cap_ =
        driver_info.device_memory_access_properties[device_ordinal_];

    // totalSize / 128 stress buffers
    uint64_t total_stress_alloc_size = driver_info.device_memory_properties[device_ordinal_][0].totalSize;
    uint64_t num_allocs = 128;
    uint64_t stress_alloc_size = total_stress_alloc_size / num_allocs;
    LOG_INFO << "Total allocatons size: " << total_stress_alloc_size;
    LOG_INFO << "Number of allocations: " << num_allocs;
    LOG_INFO << "Single allocaton size: " << stress_alloc_size;
    
    data->num_stress_allocs = num_allocs;
    data->stress_alloc_size = stress_alloc_size;

    // 1024 verify buffers of 1MB each
    data->num_verify_allocs = false ? 8 : 1024;
    data->verify_alloc_size = false ? 1024 : 1024ull * 1024ull;
  }

  virtual void inject_stress_function() {}
  virtual void inject_verify_function() {}

  void run(bool immediate) {
    data->initialize();
    LOG_INFO << "Buffers initialized";

        cmd_bundle_ = lzt::create_command_bundle(
        context_, device_handle_, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0, 0, 0, immediate);

    stress_function_ =
        lzt::create_function(module_handle_, "fill_device_memory");
        inject_stress_function();
    verify_function_ =
        lzt::create_function(module_handle_, "fill_device_memory");
        inject_verify_function();

    uint32_t group_size_x = 256;
    ze_group_count_t group_count_stress = {
        (data->stress_alloc_size / sizeof(uint64_t) + group_size_x * 16 - 1) /
            (group_size_x * 16),
                                    data->num_stress_allocs, 1};
    lzt::set_group_size(stress_function_, group_size_x, 1,
                        1);

    auto gpu_stress_ptrs_buffer_ptr = data->gpu_stress_ptrs_buffer.get();
    lzt::set_argument_value(stress_function_, 0,
                            sizeof(gpu_stress_ptrs_buffer_ptr),
                            &gpu_stress_ptrs_buffer_ptr);
    lzt::set_argument_value(stress_function_, 1,
                            sizeof(data->stress_alloc_size),
                            &data->stress_alloc_size);

    // Not used yet
    uint64_t stress_value = 0xDEADBEEF;
    lzt::set_argument_value(stress_function_, 2, sizeof(stress_value),
                            &stress_value);

   ze_group_count_t group_count_verify = {
        (data->verify_alloc_size / sizeof(uint64_t) + group_size_x * 16 - 1) /
            (group_size_x * 16),
       data->num_verify_allocs, 1};
    lzt::set_group_size(verify_function_, group_size_x, 1, 1);

   auto gpu_verify_ptr_buffer_ptr = data->gpu_verify_ptrs_buffer.get();
   lzt::set_argument_value(verify_function_, 0,
                           sizeof(gpu_verify_ptr_buffer_ptr),
                           &gpu_verify_ptr_buffer_ptr);
   lzt::set_argument_value(verify_function_, 1, sizeof(data->verify_alloc_size),
                           &data->verify_alloc_size);
   
   // Not used yet
   uint64_t verify_value = 0xDEADBEEFDEADBEEF;
   lzt::set_argument_value(verify_function_, 2, sizeof(verify_value),
                           &verify_value);


   // Launch first op on verify buffers (increment by 1, expected element value after = 1);
   lzt::append_launch_function(cmd_bundle_.list, verify_function_,
                               &group_count_verify, nullptr, 0, nullptr);
   lzt::append_barrier(cmd_bundle_.list);
   if (!immediate) {
     lzt::close_command_list(cmd_bundle_.list);
   }
   lzt::execute_and_sync_command_bundle(cmd_bundle_, UINT64_MAX);
   if (!immediate) {
     lzt::reset_command_list(cmd_bundle_.list);
   }

   // Launch stress function (Based on totalSize);
   for (int i = 0; i < 3; ++i) {
     lzt::append_launch_function(cmd_bundle_.list, stress_function_,
                                 &group_count_stress, nullptr, 0, nullptr);
     lzt::append_barrier(cmd_bundle_.list);
   }
   if (!immediate) {
     lzt::close_command_list(cmd_bundle_.list);
   }
   lzt::execute_and_sync_command_bundle(cmd_bundle_, UINT64_MAX);
   if (!immediate) {
     lzt::reset_command_list(cmd_bundle_.list);
   }

   // Launch second op on verify buffers (increment by 1, expected element value after = 2);
   lzt::append_launch_function(cmd_bundle_.list, verify_function_,
                               &group_count_verify, nullptr, 0, nullptr);
   lzt::append_barrier(cmd_bundle_.list);

   // Read 16 elements of each verify buffer to host
   for (int i = 0; i < data->num_verify_allocs; ++i) {
     lzt::append_memory_copy(cmd_bundle_.list,
                             data->host_found_output_buffer.get() + i * 16,
                             data->gpu_verify_buffers[i].get(), 128, nullptr);
     lzt::append_barrier(cmd_bundle_.list);
   }
   if (!immediate) {
     lzt::close_command_list(cmd_bundle_.list);
   }
   lzt::execute_and_sync_command_bundle(cmd_bundle_, UINT64_MAX);

    verify();

    data->cleanup();
  }

  void verify() {
    LOG_INFO << "verify output";

    // Print first element of all verify buffers
    for (uint32_t i = 0; i < data->num_verify_allocs; i++) {
      for (uint32_t j = 0; j < 1; j++) {
        std::cout << std::hex << data->host_found_output_buffer[i * 16 + j]
                  << " ";
      }
    }
  }

  ~MemoryOvercommitWorkload() {
    lzt::destroy_command_bundle(cmd_bundle_);
    lzt::destroy_function(stress_function_);
    lzt::destroy_function(verify_function_);
    lzt::destroy_module(module_handle_);
    lzt::destroy_context(context_);
  }

public:
  uint32_t device_ordinal_ = 0u;

  ze_kernel_handle_t stress_function_ = nullptr;
  ze_kernel_handle_t verify_function_ = nullptr;
  lzt::zeCommandBundle cmd_bundle_{};

  ze_driver_handle_t driver_handle_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_device_handle_t device_handle_ = nullptr;
  ze_module_handle_t module_handle_ = nullptr;
};

struct MemoryOvercommitIndirectAccessWorkload : public MemoryOvercommitWorkload {
  using MemoryOvercommitWorkload::MemoryOvercommitWorkload;
  void inject_stress_function() override {
    lzt::kernel_set_indirect_access(stress_function_,
                                    ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE);
  }
  void inject_verify_function() override {
    lzt::kernel_set_indirect_access(verify_function_,
                                    ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE);
  }
};


struct MemoryOvercommitArgs {
  uint32_t memory_size_multiple;
  /* Whether to use immediate command list in the test */
  bool is_immediate;
};

class zeMemoryOvercommitTests
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<uint32_t, bool>> {
public:
  MemoryOvercommitArgs args{};
  std::unique_ptr<MemoryOvercommitWorkload> workload;

  void SetUp() override {
    args.memory_size_multiple = std::get<0>(GetParam());
    args.is_immediate = std::get<1>(GetParam());

    LOG_INFO << "TEST args "
             << "memory_size_multiple=" << args.memory_size_multiple
             << " is_immediate=" << args.is_immediate;

    collect_drivers_info();

    DriverInfo &driver_info = drivers_info_[driver_ordinal_];
    
    LOG_INFO << "driver ordinal " << driver_ordinal_;
    LOG_INFO << "device ordinal " << device_ordinal_;

    bool loop = false;
    while (loop)
      ;

    LOG_INFO
        << "totalSize "
        << driver_info.device_memory_properties[device_ordinal_][0].totalSize;
    
    LOG_INFO << "maxSharedLocalMemory "
             << driver_info.device_compute_properties[device_ordinal_]
                    .maxSharedLocalMemory;
  }
protected:
  void collect_drivers_info() {
    LOG_INFO << "collect driver information";

    auto driver_handles = lzt::get_all_driver_handles();

    for (auto &handle : driver_handles) {
      drivers_info_.push_back(handle);
    }

    for (uint32_t i = 0; i < drivers_info_.size(); ++i) {
      auto &driver_info = drivers_info_[i];

      driver_info.device_handles = lzt::get_devices(driver_info.driver_handle);

      for (uint32_t j = 0; j < driver_info.device_handles.size(); ++j) {
        auto device_handle = driver_info.device_handles[j];

        driver_info.device_properties.emplace_back(
            lzt::get_device_properties(device_handle));

        driver_info.device_compute_properties.emplace_back(
            lzt::get_compute_properties(device_handle));

        driver_info.device_memory_properties.emplace_back(
            lzt::get_memory_properties(device_handle));

        driver_info.device_memory_access_properties.emplace_back(
            lzt::get_memory_access_properties(device_handle));
      }
    }
  }

  std::vector<DriverInfo> drivers_info_;

  uint32_t device_ordinal_ = 0u;
  uint32_t driver_ordinal_ = 1u;

  uint32_t workload_runs = 1u;
};

LZT_TEST_P(
    zeMemoryOvercommitTests,
    GivenDeviceMemoryWhenAllocationSizeLargerThenDeviceMaxMemoryThenMemoryIsPagedOffAndOnTheDevice) {
  if ((drivers_info_[driver_ordinal_]
           .device_memory_access_properties[device_ordinal_]
           .deviceAllocCapabilities &
       ZE_MEMORY_ACCESS_CAP_FLAG_RW) == 0) {
    GTEST_SKIP() << "Unable to allocate device memory";
  }
  workload = std::make_unique<MemoryOvercommitWorkload>(
      device_ordinal_, std::make_unique<MemoryOvercommitData>());
  for (uint32_t i = 0; i < workload_runs; ++i) {
    workload->initialize(drivers_info_[driver_ordinal_],
                         args.memory_size_multiple);
    workload->run(args.is_immediate);
  }
}

LZT_TEST_P(
    zeMemoryOvercommitTests,
    GivenDeviceMemoryWithIndirectAccessWhenAllocationSizeLargerThenDeviceMaxMemoryThenMemoryIsPagedOffAndOnTheDevice) {
  if ((drivers_info_[driver_ordinal_]
           .device_memory_access_properties[device_ordinal_]
           .deviceAllocCapabilities &
       ZE_MEMORY_ACCESS_CAP_FLAG_RW) == 0) {
    GTEST_SKIP() << "Unable to allocate device memory";
  }
  workload = std::make_unique<MemoryOvercommitIndirectAccessWorkload>(
      device_ordinal_, std::make_unique<MemoryOvercommitData>());
  for (uint32_t i = 0; i < workload_runs; ++i) {
    workload->initialize(drivers_info_[driver_ordinal_],
                         args.memory_size_multiple);
    workload->run(args.is_immediate);
  }
}

LZT_TEST_P(
    zeMemoryOvercommitTests,
    GivenSharedSingleDeviceMemoryWhenAllocationSizeLargerThenDeviceMaxMemoryThenMemoryIsPagedOffAndOnTheDevice) {
  if ((drivers_info_[driver_ordinal_]
           .device_memory_access_properties[device_ordinal_]
           .sharedSingleDeviceAllocCapabilities &
       ZE_MEMORY_ACCESS_CAP_FLAG_RW) == 0) {
    GTEST_SKIP() << "Unable to allocate shared single device memory";
  }
  workload = std::make_unique<MemoryOvercommitIndirectAccessWorkload>(
      device_ordinal_, std::make_unique<MemoryOvercommitData>());
  for (uint32_t i = 0; i < workload_runs; ++i) {
    workload->initialize(drivers_info_[driver_ordinal_],
                         args.memory_size_multiple);
    workload->run(args.is_immediate);
  }
}

LZT_TEST_P(
    zeMemoryOvercommitTests,
    GivenSharedCrossDeviceMemoryWhenAllocationSizeLargerThenDeviceMaxMemoryThenMemoryIsPagedOffAndOnTheDevice) {
  std::vector<ze_device_handle_t> &device_handles =
      drivers_info_[driver_ordinal_].device_handles;
  if (device_handles.size() <
      2) {
    GTEST_SKIP() << "Test requires at least 2 devices";
  }
  uint32_t peer_device_ordinal = 0;
  for (auto i = 0u; i < device_handles.size();
       i++) {
    if (i != device_ordinal_) {
      peer_device_ordinal = i;
      break;
    }
  }
  if (lzt::can_access_peer(
          device_handles[device_ordinal_],
                           device_handles[peer_device_ordinal])) {
    GTEST_SKIP() << "Devices do not have p2p access";
  }
  LOG_INFO << "Allocation on device " << device_ordinal_ << " launch on device "
           << peer_device_ordinal;
  workload = std::make_unique<MemoryOvercommitWorkload>(
      device_ordinal_, std::make_unique<MemoryOvercommitData>());
  for (uint32_t i = 0; i < workload_runs; ++i) {
    workload->initialize(drivers_info_[driver_ordinal_],
                         args.memory_size_multiple);
    workload->device_ordinal_ = peer_device_ordinal;
    workload->device_handle_ = device_handles[peer_device_ordinal];
    workload->run(args.is_immediate);
  }
}

LZT_TEST_P(
    zeMemoryOvercommitTests,
    GivenSharedSystemMemoryWhenAllocationSizeLargerThenDeviceMaxMemoryThenMemoryIsPagedOffAndOnTheDevice) {
  if (!lzt::supports_shared_system_alloc(
          drivers_info_[driver_ordinal_]
              .device_memory_access_properties[device_ordinal_])) {
    GTEST_SKIP() << "Unable to allocate shared system memory";
  }
  workload = std::make_unique<MemoryOvercommitWorkload>(
      device_ordinal_, std::make_unique<MemoryOvercommitData>());
  for (uint32_t i = 0; i < workload_runs; ++i) {
    workload->initialize(drivers_info_[driver_ordinal_],
                         args.memory_size_multiple);
    workload->run(args.is_immediate);
  }
}

INSTANTIATE_TEST_SUITE_P(TestAllInputPermuntations, zeMemoryOvercommitTests,
    ::testing::Combine(::testing::Values(1), ::testing::Bool()));

} // namespace
