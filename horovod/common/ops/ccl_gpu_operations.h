// Copyright (C) 2022 Intel CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef HOROVOD_CCL_GPU_OPERATIONS_H_
#define HOROVOD_CCL_GPU_OPERATIONS_H_

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "oneapi/ccl.hpp"

#include "../logging.h"
#include "gpu_operations.h"

using cclComm_t = ccl::communicator;

namespace horovod {
namespace common {
// TODO(Maozhou): combine w/ CCL CPU
ccl::datatype GetCCLDataType(const std::shared_ptr<Tensor>& tensor);

struct ccl4hvd {
  ccl::stream ccl_stream_;
  cclComm_t ccl_comm_;
};

struct ccl_data {
  std::shared_ptr<ccl::event>& ccl_event;
};

class CCLGPUContext {
public:
  CCLGPUContext() = default;
  ~CCLGPUContext() = default;

  void Initialize(HorovodGlobalState& state);
  void Finalize(HorovodGlobalState& state);

  // TODO(Maozhou): use the ones from GPUContext
  int GetGpuEvent(Event* event, gpuStream_t& stream);

  gpuError_t ReleaseGpuEvent(Event event);

  void RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue,
                   std::string name, gpuStream_t& stream);

  Event RecordEvent(gpuStream_t& stream);

  void ReleaseEvent(Event event);

  void
  WaitForEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                const std::vector<TensorTableEntry>& entries,
                Timeline& timeline,
                const std::function<void()>& error_check_callback = nullptr);

  void ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                   const std::vector<TensorTableEntry>& entries,
                   Timeline& timeline,
                   const std::function<void()>& error_check_callback = nullptr,
                   bool elastic = false);
  void StreamCreate(const TensorTableEntry& e, gpuStream_t& stream);
  void StreamSynchronize(gpuStream_t stream);

  template <typename ccl_fn_type>
  static decltype(auto) CallWithLock(std::mutex& lock, ccl_fn_type fn) {
    std::unique_lock<std::mutex> GlobalMutex(lock);
    return fn();
  }

  // Thread pool for finalizer threads
  ThreadPool finalizer_thread_pool;

  // base primitives
  std::vector<
      std::unordered_map<std::tuple<int32_t, std::vector<int32_t>>, ccl4hvd>>
      ccl_comms;
  std::vector<std::unordered_map<int, gpuStream_t>> streams;
  // TODO(Maozhou): sycl_events
  std::unordered_map<sycl::queue, std::queue<Event>> sycl_events;
  // TODO(Maozhou): remove it when CCL_SYCL_OUTPUT_EVENT=1 is ready
  std::unordered_map<sycl::event, std::shared_ptr<ccl::event>>
      sycl_2_ccl_event_map;

  // ccl helpers knobs
  bool enable_fin_threads;
  bool enable_cache;

  static std::mutex GlobalMutex;
};

class CCLGPUOpContext {
public:
  CCLGPUOpContext(CCLGPUContext* context, HorovodGlobalState* global_state,
                  Communicator communicator_type)
      : kvs_(nullptr), ccl_context_(context), global_state_(global_state),
        communicator_type_(communicator_type){};
  ~CCLGPUOpContext();

  // TODO(Maozhou): use the ones from GPUOpContext
  void InitGPU(const std::vector<TensorTableEntry>& entries);
  void InitGPUQueue(const std::vector<TensorTableEntry>& entries,
                    const Response& response);
  Status
  FinalizeGPUQueue(std::vector<TensorTableEntry>& entries,
                   ccl_data ccl_util_data, bool free_host_buffer = true,
                   const std::function<void()>& error_check_callback = nullptr);
  void InitCCLComm(const std::vector<TensorTableEntry>& entries,
                   const std::vector<int32_t>& ccl_device_map);

  // helpers
  bool IsEnabled(const std::vector<TensorTableEntry>& entries) const;
  ccl::communicator& GetCCLComm(const TensorTableEntry& entry,
                                const std::vector<int32_t>& devices);
  ccl::stream& GetCCLStream(const TensorTableEntry& entry,
                            const std::vector<int32_t>& devices);

  std::queue<std::pair<std::string, Event>> event_queue;
  gpuStream_t stream;

  void* host_buffer = nullptr;

  std::shared_ptr<ccl::kvs> kvs_;

  std::function<void()> error_check_callback_;

private:
  void PopulateCCLCommStrategy(int& ccl_rank, int& ccl_size,
                               Communicator& ccl_id_bcast_comm,
                               const ProcessSet& process_set);

  CCLGPUContext* ccl_context_;
  HorovodGlobalState* global_state_;
  Communicator communicator_type_;

  std::vector<ccl::stream> ccl_streams;
  std::vector<ccl::device> ccl_devices;
  std::vector<ccl::context> ccl_contexts;
};

class CCLGPUAllreduce : public GPUAllreduce {
public:
  CCLGPUAllreduce(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state,
                  Communicator communicator_type = Communicator::GLOBAL)
      : GPUAllreduce(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state){};

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e,
                                 void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset,
                                  TensorTableEntry& e) override;

  void ScaleMemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const void*& fused_input_data,
                                 void*& buffer_data, size_t& buffer_len,
                                 double scale_factor);
  void ScaleMemcpyOutFusionBuffer(void* buffer_data, size_t buffer_len,
                                  double scale_factor,
                                  std::vector<TensorTableEntry>& entries);

  void ScaleBuffer(double scale_factor,
                   const std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data,
                   int64_t num_elements) override;

  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

class CCLGPUBroadcast : public GPUBroadcast {
public:
  CCLGPUBroadcast(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state,
                  Communicator communicator_type = Communicator::GLOBAL)
      : GPUBroadcast(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state){};

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

class CCLGPUAllgather : public GPUAllgather {
public:
  CCLGPUAllgather(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state,
                  Communicator communicator_type = Communicator::GLOBAL)
      : GPUAllgather(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state){};

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e,
                                 void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset,
                                  TensorTableEntry& e, int64_t entry_offset,
                                  size_t entry_size) override;

  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

class CCLGPUAlltoall : public GPUAlltoall {
public:
  CCLGPUAlltoall(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                 HorovodGlobalState* global_state,
                 Communicator communicator_type = Communicator::GLOBAL)
      : GPUAlltoall(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state){};

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CCL_GPU_OPERATIONS_H_
