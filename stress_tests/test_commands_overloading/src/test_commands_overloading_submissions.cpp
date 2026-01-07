/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "gtest/gtest.h"
#include "utils/utils.hpp"
#include "test_harness/test_harness.hpp"
#include "logging/logging.hpp"
#include "stress_common_func.hpp"
#include <level_zero/ze_api.h>
#include <unistd.h>
#include <fcntl.h>
#include <mutex>
#include <cstdio>

namespace lzt = level_zero_tests;

namespace {

static int saved_stdout_fd = -1;
static int saved_stderr_fd = -1;
static int null_fd = -1;
static std::mutex redirect_mutex;
static int redirect_refcount = 0;

void disable_stdout_stderr() {
  std::lock_guard<std::mutex> lock(redirect_mutex);
  if (redirect_refcount++ > 0) {
    return;
  }
  fflush(stdout);
  fflush(stderr);
  null_fd = open("/dev/null", O_WRONLY);
  if (null_fd == -1) {
    redirect_refcount--;
    return;
  }
  saved_stdout_fd = dup(STDOUT_FILENO);
  saved_stderr_fd = dup(STDERR_FILENO);
  if (saved_stdout_fd == -1 || saved_stderr_fd == -1) {
    if (saved_stdout_fd != -1) {
      close(saved_stdout_fd);
      saved_stdout_fd = -1;
    }
    if (saved_stderr_fd != -1) {
      close(saved_stderr_fd);
      saved_stderr_fd = -1;
    }
    close(null_fd);
    null_fd = -1;
    redirect_refcount--;
    return;
  }
  dup2(null_fd, STDOUT_FILENO);
  dup2(null_fd, STDERR_FILENO);
}

void enable_stdout_stderr() {
  std::lock_guard<std::mutex> lock(redirect_mutex);
  if (redirect_refcount == 0) {
    return;
  }
  if (--redirect_refcount > 0) {
    return;
  }
  fflush(stdout);
  fflush(stderr);
  if (saved_stdout_fd != -1) {
    dup2(saved_stdout_fd, STDOUT_FILENO);
    close(saved_stdout_fd);
    saved_stdout_fd = -1;
  }
  if (saved_stderr_fd != -1) {
    dup2(saved_stderr_fd, STDERR_FILENO);
    close(saved_stderr_fd);
    saved_stderr_fd = -1;
  }
  if (null_fd != -1) {
    close(null_fd);
    null_fd = -1;
  }
}

enum class zeCmdListMode { Regular = 0, Immediate, ImmediateAppendRegular };

std::string to_string(zeCmdListMode mode) {
  switch (mode) {
  case zeCmdListMode::Regular:
    return "Regular";
  case zeCmdListMode::Immediate:
    return "Immediate";
  case zeCmdListMode::ImmediateAppendRegular:
    return "zeCommandListAppendCommandListsImmediateExp";
  default:
    return "Unknown";
  }
}

class zeCommandListSubmissionsStressTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          std::tuple<zeCmdListMode, uint64_t>> {
protected:
  void SetUp() override {
    cmd_list_mode = std::get<0>(GetParam());
    submission_count = std::get<1>(GetParam());
    is_immediate = (cmd_list_mode == zeCmdListMode::Immediate ||
                    cmd_list_mode == zeCmdListMode::ImmediateAppendRegular);
    context = lzt::get_default_context();
    device = lzt::get_default_device(lzt::get_default_driver());

    execute_list = lzt::create_immediate_command_list(
        device,
        0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0);
  }

  void TearDown() override {
    lzt::destroy_command_list(execute_list);
  }

  ze_context_handle_t context = nullptr;
  ze_device_handle_t device = nullptr;
  zeCmdListMode cmd_list_mode;
  uint64_t submission_count;
  bool is_immediate;
  std::vector<ze_command_list_handle_t> cmd_lists;
  ze_command_list_handle_t execute_list = nullptr;
  ze_command_queue_handle_t cmd_queue = nullptr;
};

LZT_TEST_P(
    zeCommandListSubmissionsStressTest,
    GivenMultipleCommandListSubmissionsWhenExecutingThenSuccessIsReturned) {
      uint32_t num_threads = 1; // std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
      LOG_INFO << "Using " << num_threads << " threads for submissions";
LOG_INFO << "Cmd list mode: " << to_string(cmd_list_mode);
  if (cmd_list_mode == zeCmdListMode::Regular) {
    cmd_queue = lzt::create_command_queue(device);
  }
  ze_module_handle_t module_handle =
      lzt::create_module(context, device, "test_commands_overloading.spv",
                         ZE_MODULE_FORMAT_IL_SPIRV, nullptr, nullptr);
  ze_kernel_handle_t kernel_handle =
      lzt::create_function(module_handle, "test_submissions");
  lzt::set_group_size(kernel_handle, 1, 1, 1);
  ze_group_count_t group_count = {1, 1, 1};

uint8_t zero = 0;
  void *verify_value = 
      lzt::allocate_device_memory(sizeof(uint64_t), 8, 0, 0, device, context);
  lzt::append_memory_set(execute_list, verify_value, &zero, sizeof(uint64_t));
  lzt::synchronize_command_list_host(execute_list, UINT64_MAX);

  lzt::set_argument_value(kernel_handle, 0, sizeof(verify_value),
                         &verify_value);

    for (uint32_t i = 0; i < num_threads; ++i) {
    if (cmd_list_mode == zeCmdListMode::Immediate) {
      cmd_lists.push_back(
      lzt::create_immediate_command_list(
        device,
        0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0));
    } else {
      cmd_lists.push_back(
        lzt::create_command_list(device, 0));
    }
  } 

  std::vector<std::thread> threads;
  // Submit workload
  LOG_INFO << "Submitting " << submission_count << " kernel launches";
  uint64_t submissions_per_thread = submission_count / num_threads;
  uint64_t remainder = submission_count % num_threads;

  disable_stdout_stderr();

  std::atomic<bool> enabled = false;
  for (uint32_t i = 0; i < num_threads; i++) {
    uint64_t count = submissions_per_thread + ((i == num_threads - 1) ? remainder : 0);
    threads.push_back(std::thread([&, i, count]() {
      for (uint64_t j = 0; j < count; j++) {
        lzt::append_launch_function(
            cmd_lists[i], kernel_handle, &group_count, nullptr, 0, nullptr);
        if (j % 1'000'000 == 0 && j != 0) {
          LOG_INFO << "Thread " << i << " submitted " << j << " submissions";
          LOG_INFO << lzt::total_available_host_memory() << " bytes of host memory available";
        }
        
        if (j > 4'294'967'200 && !enabled.load()) {
          enabled.store(true);
          enable_stdout_stderr();
        }
      }
    }));
  }

  for (auto &t : threads) {
    t.join();
  }

  enable_stdout_stderr();

  // Try overflowing submission count
  for (uint64_t i = 0; i < UINT8_MAX; i++) {
    lzt::append_launch_function(
        cmd_lists[0], kernel_handle, &group_count, nullptr, 0, nullptr);
  }
  LOG_INFO << "Submissions done";

  if (cmd_list_mode == zeCmdListMode::Regular ||
      cmd_list_mode == zeCmdListMode::ImmediateAppendRegular) {
    for (uint32_t i = 0; i < num_threads; ++i) {
      lzt::close_command_list(cmd_lists[i]);
    }
  }

  if (cmd_list_mode == zeCmdListMode::ImmediateAppendRegular) {
    lzt::append_command_lists_immediate_exp(execute_list, num_threads, cmd_lists.data());
    lzt::synchronize_command_list_host(execute_list, UINT64_MAX);
  }
  if (cmd_list_mode == zeCmdListMode::Regular) {
    lzt::execute_command_lists(cmd_queue, num_threads, cmd_lists.data(),
                               nullptr);
    lzt::synchronize(cmd_queue, UINT64_MAX);
  }
  if (cmd_list_mode == zeCmdListMode::Immediate) {
    for (uint32_t i = 0; i < num_threads; ++i) {
      lzt::synchronize_command_list_host(cmd_lists[i], UINT64_MAX);
    }
  }

  // Verify
  uint64_t read_value = 0;
  lzt::append_memory_copy(execute_list, &read_value, verify_value,
                          sizeof(uint64_t), nullptr);
  lzt::synchronize_command_list_host(execute_list, UINT64_MAX);
  EXPECT_EQ(read_value, submission_count + UINT8_MAX);

  // Cleanup
  for (uint32_t i = 0; i < num_threads; ++i) {
    lzt::destroy_command_list(cmd_lists[i]);
  }
  lzt::destroy_function(kernel_handle);
  lzt::destroy_module(module_handle);
  lzt::free_memory(context, verify_value);
}

INSTANTIATE_TEST_SUITE_P(
    TestCommandListSubmissionsTypeMax, zeCommandListSubmissionsStressTest,
    ::testing::Combine(::testing::Values(zeCmdListMode::Immediate),
                       ::testing::Values(UINT8_MAX, UINT32_MAX)));

} // namespace