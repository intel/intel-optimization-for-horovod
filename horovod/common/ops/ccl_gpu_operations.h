// Copyright (C) 2022-2023 Intel CORPORATION. All rights reserved.
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

namespace horovod {
namespace common {
// TODO(Maozhou): combine w/ CCL CPU
ccl::datatype GetCCLDataType(const std::shared_ptr<Tensor>& tensor);

struct ccl4hvd {
  ccl::stream ccl_stream_;
  ccl::communicator ccl_comm_;
};

class CCLGPUContext {
public:
  CCLGPUContext() = default;
  ~CCLGPUContext() = default;

  void Initialize(HorovodGlobalState& state);
  void Finalize();

  // base primitives
  std::vector<
      std::unordered_map<std::tuple<int32_t, std::vector<int32_t>>, ccl4hvd>>
      ccl_comms;

  // ccl helpers knobs
  // TODO(Maozhou): move to global_state when unified with CCL CPU?
  bool enable_cache;
};

class CCLGPUOpContext {
public:
  CCLGPUOpContext(CCLGPUContext* context, HorovodGlobalState* global_state,
                  Communicator communicator_type)
      : kvs_(nullptr), ccl_context_(context), global_state_(global_state),
        communicator_type_(communicator_type){};
  ~CCLGPUOpContext();

  void InitCCLComm(const gpuStream_t& stream,
                   const std::vector<TensorTableEntry>& entries,
                   const std::vector<int32_t>& ccl_device_map);

  // helpers
  bool IsEnabled(const std::vector<TensorTableEntry>& entries) const;
  ccl::communicator& GetCCLComm(const TensorTableEntry& entry,
                                const std::vector<int32_t>& devices);
  ccl::stream& GetCCLStream(const TensorTableEntry& entry,
                            const std::vector<int32_t>& devices);

  std::shared_ptr<ccl::kvs> kvs_;
  ccl4hvd* ccl4hvd_;
  // oneCCL does not support async error check
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
        global_state_(global_state) {}

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

class CCLGPUBroadcast : public GPUBroadcast {
public:
  CCLGPUBroadcast(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state,
                  Communicator communicator_type = Communicator::GLOBAL)
      : GPUBroadcast(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state) {}

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
        global_state_(global_state) {}

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  Status AllocateOutput(std::vector<TensorTableEntry>& entries,
                        const Response& response,
                        int64_t**& entry_component_sizes) override;

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
        global_state_(global_state) {}

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  template <typename T>
  Status PrepareOutputAndParams(TensorTableEntry& e, std::vector<T>& sdispls,
                                std::vector<T>& rdispls,
                                std::vector<T>& sendcounts,
                                std::vector<T>& recvcounts) {
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    auto world_size = process_set.controller->GetSize();

    const auto& splits = e.splits;
    std::vector<int32_t> recvsplits;

    process_set.controller->AlltoallGetRecvSplits(splits, recvsplits);

    // Every tensor participating in Alltoall operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }
    int64_t slice_num_elements = slice_shape.num_elements();

    // Prepare send/recvcounts and displacements for Alltoallv
    sdispls.resize(world_size);
    rdispls.resize(world_size);
    sendcounts.resize(world_size);
    recvcounts.resize(world_size);

    size_t output_first_dim = 0;
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = splits[i] * slice_num_elements;
      recvcounts[i] = recvsplits[i] * slice_num_elements;
      output_first_dim += recvsplits[i];
    }

    for (int i = 1; i < world_size; ++i) {
      sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
      rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }

    // Allocate output
    TensorShape output_shape;
    output_shape.AddDim(output_first_dim);
    output_shape.AppendShape(slice_shape);

    std::shared_ptr<ReadyEvent> event;
    Status status = e.context->AllocateOutput(output_shape, &e.output, &event);
    if (!status.ok()) {
      LOG(WARNING) << "CCLGPUAlltoall::PrepareOutputAndParams failed to "
                      "allocate output: "
                   << status.reason();
      return status;
    }

    // Add event dependency for output allocation to stream
    if (event) {
      (*gpu_op_context_.stream)->ext_oneapi_submit_barrier({event->event()});
    }

    // Allocate and fill received_splits output
    TensorShape received_splits_shape;
    received_splits_shape.AddDim(recvsplits.size());

    std::shared_ptr<ReadyEvent> revent;
    Status rstatus = e.context->AllocateOutput(1, received_splits_shape,
                                               &e.received_splits, &revent);
    if (!rstatus.ok()) {
      LOG(WARNING)
          << "CCLGPUAlltoall::PrepareOutputAndParams failed to allocate "
             "received_splits: "
          << status.reason();
      return rstatus;
    }

    // Add event dependency for received_splits allocation to stream
    if (revent) {
      (*gpu_op_context_.stream)->ext_oneapi_submit_barrier({revent->event()});
    }

    auto* target_pointer = reinterpret_cast<int32_t*>(
        const_cast<void*>(e.received_splits->data()));
    std::copy(recvsplits.cbegin(), recvsplits.cend(), target_pointer);

    return Status::OK();
  }

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

class CCLGPUReducescatter : public GPUReducescatter {
public:
  CCLGPUReducescatter(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                      HorovodGlobalState* global_state,
                      Communicator communicator_type = Communicator::GLOBAL)
      : GPUReducescatter(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state) {}

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

class CCLGPUTorusAllreduce : public GPUAllreduce {
public:
  CCLGPUTorusAllreduce(CCLGPUContext* local_ccl_context,
                       CCLGPUContext* cross_ccl_context,
                       GPUContext* gpu_context,
                       HorovodGlobalState* global_state)
      : GPUAllreduce(gpu_context, global_state),
        local_ccl_context_(local_ccl_context),
        cross_ccl_context_(cross_ccl_context),
        local_ccl_op_context_(local_ccl_context, global_state,
                              Communicator::LOCAL),
        cross_ccl_op_context_(cross_ccl_context, global_state,
                              Communicator::CROSS),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* local_ccl_context_;
  CCLGPUContext* cross_ccl_context_;
  CCLGPUOpContext local_ccl_op_context_;
  CCLGPUOpContext cross_ccl_op_context_;
  HorovodGlobalState* global_state_;
};
} // namespace common
} // namespace horovod

#endif // HOROVOD_CCL_GPU_OPERATIONS_H_
