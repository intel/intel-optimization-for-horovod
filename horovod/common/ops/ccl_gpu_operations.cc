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

#include "ccl_gpu_operations.h"

// TODO(IOH): move to sycl_operations.cc
#include "sycl/sycl_kernels.h"

namespace horovod {
namespace common {

std::mutex CCLGPUContext::GlobalMutex;

ccl::datatype GetCCLDataType(const std::shared_ptr<Tensor>& tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return ccl::datatype::uint8;
  case HOROVOD_INT8:
    return ccl::datatype::int8;
  case HOROVOD_UINT16:
    return ccl::datatype::uint16;
  case HOROVOD_INT16:
    return ccl::datatype::int16;
  case HOROVOD_FLOAT16:
    return ccl::datatype::float16;
  case HOROVOD_BFLOAT16:
    return ccl::datatype::bfloat16;
  case HOROVOD_FLOAT32:
    return ccl::datatype::float32;
  case HOROVOD_FLOAT64:
    return ccl::datatype::float64;
  case HOROVOD_INT32:
    return ccl::datatype::int32;
  case HOROVOD_INT64:
    return ccl::datatype::int64;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in CCL.");
  }
}

inline void
CheckTensorTableEntry(const std::vector<TensorTableEntry>& entries) {
  if (entries.empty()) {
    throw std::runtime_error("TensorTableEntry is empty!");
  }
}

void CCLGPUContext::Initialize(HorovodGlobalState& state) {
  ccl::init();

  enable_cache = GetBoolEnvOrDefault(HOROVOD_CCL_CACHE, false);
  LOG(INFO) << "CCLGPUContext initialized: \n"
            << "enable_cache " << enable_cache << "\n";

  ccl_comms.resize(state.num_nccl_streams);
}

void CCLGPUContext::Finalize() { ccl_comms.clear(); }

CCLGPUOpContext::~CCLGPUOpContext() {
  ccl_streams.clear();
  ccl_devices.clear();
  ccl_contexts.clear();
}

void CCLGPUOpContext::InitCCLComm(const gpuStream_t& stream,
                                  const std::vector<TensorTableEntry>& entries,
                                  const std::vector<int32_t>& ccl_device_map) {
  CheckTensorTableEntry(entries);

  auto process_set_id = entries[0].process_set_id;
  auto& process_set = global_state_->process_set_table.Get(process_set_id);

  if (ccl_context_->ccl_comms[global_state_->current_nccl_stream].empty()) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_CCL);

    int ccl_rank, ccl_size;
    Communicator ccl_id_bcast_comm;
    PopulateCCLCommStrategy(ccl_rank, ccl_size, ccl_id_bcast_comm, process_set);

    if (ccl_rank == 0) {
      if (!kvs_) {
        kvs_ = ccl::create_main_kvs();
        auto main_addr = kvs_->get_address();
        global_state_->global_controller->Bcast(
            (void*)main_addr.data(), main_addr.size(), 0, ccl_id_bcast_comm);
      }
    } else {
      ccl::kvs::address_type main_addr;
      global_state_->global_controller->Bcast(
          (void*)main_addr.data(), main_addr.size(), 0, Communicator::GLOBAL);
      kvs_ = ccl::create_kvs(main_addr);
    }

    auto queue = stream;
    {
      std::lock_guard<std::mutex> lock(CCLGPUContext::GlobalMutex);
      ccl_streams.push_back(ccl::create_stream(*queue));
      ccl_devices.push_back(ccl::create_device(queue->get_device()));
      ccl_contexts.push_back(ccl::create_context(queue->get_context()));
      // fill ccl_comms via creating communicator
      ccl_context_->ccl_comms[global_state_->current_nccl_stream].insert(
          {std::make_tuple(process_set_id, ccl_device_map),
           ccl4hvd{ccl_streams[0],
                   ccl::create_communicator(ccl_size, ccl_rank, ccl_devices[0],
                                            ccl_contexts[0], kvs_)}});
    }

    process_set.controller->Barrier(Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);
  }
}

// helpers
void CCLGPUOpContext::PopulateCCLCommStrategy(int& ccl_rank, int& ccl_size,
                                              Communicator& ccl_id_bcast_comm,
                                              const ProcessSet& process_set) {
  if (communicator_type_ == Communicator::GLOBAL) {
    ccl_rank = process_set.controller->GetRank();
    ccl_size = process_set.controller->GetSize();
  } else if (communicator_type_ == Communicator::LOCAL) {
    ccl_rank = process_set.controller->GetLocalRank();
    ccl_size = process_set.controller->GetLocalSize();
  } else {
    throw std::logic_error("Communicator type " +
                           std::to_string(communicator_type_) +
                           " is not supported in CCL mode.");
  }
  ccl_id_bcast_comm = communicator_type_;
}

bool CCLGPUOpContext::IsEnabled(
    const std::vector<TensorTableEntry>& entries) const {
  return entries[0].device != CPU_DEVICE_ID;
}

ccl::communicator&
CCLGPUOpContext::GetCCLComm(const TensorTableEntry& entry,
                            const std::vector<int32_t>& devices) {
  return ccl_context_->ccl_comms[global_state_->current_nccl_stream]
      .at(std::make_tuple(entry.process_set_id, devices))
      .ccl_comm_;
}

ccl::stream&
CCLGPUOpContext::GetCCLStream(const TensorTableEntry& entry,
                              const std::vector<int32_t>& devices) {
  return ccl_context_->ccl_comms[global_state_->current_nccl_stream]
      .at(std::make_tuple(entry.process_set_id, devices))
      .ccl_stream_;
}

// Allreduce
Status CCLGPUAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  CheckTensorTableEntry(entries);
  auto& first_entry = entries[0];

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  auto ccl_reduction_op = ccl::reduction::sum;
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
    ccl_reduction_op = ccl::reduction::sum;
    auto process_set_id = first_entry.process_set_id;
    auto& process_set = global_state_->process_set_table.Get(process_set_id);
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
  } else if (response.reduce_op() == ReduceOp::SUM) {
    ccl_reduction_op = ccl::reduction::sum;
  } else if (response.reduce_op() == ReduceOp::MIN) {
    ccl_reduction_op = ccl::reduction::min;
  } else if (response.reduce_op() == ReduceOp::MAX) {
    ccl_reduction_op = ccl::reduction::max;
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    ccl_reduction_op = ccl::reduction::prod;
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  LOG(DEBUG) << "CCLGPUAllreduce::Execute"
             << " final prescale_factor: " << prescale_factor
             << " final postscale_factor: " << postscale_factor;

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  if (entries.size() > 1) {
    ScaleMemcpyInFusionBuffer(entries, fused_input_data, buffer_data,
                              buffer_len, prescale_factor);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = const_cast<void*>(first_entry.output->data());
    buffer_len = (size_t)first_entry.output->size();
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (prescale_factor != 1.0) {
      // Execute prescaling op
      ScaleBuffer(prescale_factor, entries, fused_input_data, buffer_data,
                  num_elements);
      fused_input_data = buffer_data; // for unfused, scale is done out of place
    }
  }

  // cache
  auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
  if (ccl_context_->enable_cache) {
    std::string match_id = "dt_" + DataType_Name(first_entry.tensor->dtype()) +
                           "_len_" + std::to_string(buffer_len);

    if (prescale_factor != 1.0) {
      match_id += "_prescale_" + std::to_string(prescale_factor);
    }
    if (postscale_factor != 1.0) {
      match_id += "_postscale_" + std::to_string(postscale_factor);
    }
    for (size_t idx = 0; idx < entries.size(); idx++) {
      match_id += "_" + entries[idx].tensor_name;
    }

    attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));
    attr.set<ccl::operation_attr_id::to_cache>(true);

    LOG(DEBUG) << "CCLGPUAllreduce::Execute enable_cache"
               << " buffer_len: " << buffer_len << " recv_buf: " << buffer_data
               << " match_id: " << match_id;
  } else {
    attr.set<ccl::operation_attr_id::to_cache>(false);
  }

  // Do allreduce
  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());
  LOG(DEBUG) << "Do CCLGPUAllreduce, number of elements: " << num_elements
             << ", dtype: " << DataType_Name(first_entry.tensor->dtype());

  // TODO(Maozhou): we assume oneCCL honors SYCL in-order queue semantics, and
  // can use the sycl::event from ccl::event.get_native() when oneCCL is ready
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl::allreduce(
        fused_input_data, buffer_data, (size_t)num_elements,
        GetCCLDataType(first_entry.tensor), ccl_reduction_op,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()), attr);
  });

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_ALLREDUCE,
                              *gpu_op_context_.stream);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, buffer_len, postscale_factor,
                               entries);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    if (postscale_factor != 1.0) {
      // Execute postscaling op
      ScaleBuffer(postscale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

void CCLGPUAllreduce::ScaleMemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
    void*& buffer_data, size_t& buffer_len, double scale_factor) {
  auto& first_entry = entries[0];
  // Access the fusion buffer.
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));
  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];

  if (enableBatchedScaledD2DMemcpy(global_state_, entries)) {
    buffer_len = 0;
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      auto tensor_elements =
          e.tensor->size() / DataType_Size(e.tensor->dtype());
      if (count == 0) {
        offset = tensor_elements;
      } else {
        offset = d2d_params.offsets[count - 1] + tensor_elements;
      }

      if (offset >= UNSIGNED_INT32_MAX) {
        BatchedScaledD2DMemcpyInImpl(d2d_params, buffer_data, count,
                                     scale_factor, first_entry.tensor->dtype(),
                                     stream);
        count = 0;
        offset = tensor_elements;
      }

      // Set small buffers and offsets
      d2d_params.buffers[count] = const_cast<void*>(e.tensor->data());
      d2d_params.offsets[count] = offset;
      buffer_len += e.tensor->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == entries.size()) {
        BatchedScaledD2DMemcpyInImpl(d2d_params, buffer_data, count,
                                     scale_factor, first_entry.tensor->dtype(),
                                     stream);
        count = 0;
        offset = 0;
      }
    }
  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
      offset += e.tensor->size();
    }
    buffer_len = (size_t)offset;
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (scale_factor != 1.0) {
      ScaleBuffer(scale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }
  }
  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void CCLGPUAllreduce::ScaleMemcpyOutFusionBuffer(
    void* buffer_data, size_t buffer_len, double scale_factor,
    std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];

  if (enableBatchedScaledD2DMemcpy(global_state_, entries)) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      auto tensor_elements =
          e.output->size() / DataType_Size(e.output->dtype());
      if (count == 0) {
        offset = tensor_elements;
      } else {
        offset = d2d_params.offsets[count - 1] + tensor_elements;
      }

      if (tensor_elements >= UNSIGNED_INT32_MAX) {
        BatchedScaledD2DMemcpyOutImpl(d2d_params, buffer_data, count,
                                      scale_factor, first_entry.tensor->dtype(),
                                      stream);
        count = 0;
        offset = tensor_elements;
      }

      // Set input/output pointers and sizes
      d2d_params.buffers[count] = const_cast<void*>(e.output->data());
      d2d_params.offsets[count] = offset;
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == entries.size()) {
        // Perform batched d2d memcpy
        BatchedScaledD2DMemcpyOutImpl(d2d_params, buffer_data, count,
                                      scale_factor, first_entry.tensor->dtype(),
                                      stream);
        count = 0;
        offset = 0;
      }
    }
  } else {
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (scale_factor != 1.0) {
      ScaleBuffer(scale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
      offset += e.tensor->size();
    }
  }
}

void CCLGPUAllreduce::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      buffer_data_at_offset, e.tensor->data(), (size_t)e.tensor->size(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void CCLGPUAllreduce::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      (void*)e.output->data(), buffer_data_at_offset, (size_t)e.tensor->size(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void CCLGPUAllreduce::ScaleBuffer(double scale_factor,
                                  const std::vector<TensorTableEntry>& entries,
                                  const void* fused_input_data,
                                  void* buffer_data, int64_t num_elements) {
  gpu_context_->ScaleBufferImpl(
      fused_input_data, buffer_data, num_elements, scale_factor,
      entries[0].tensor->dtype(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device]);
}

bool CCLGPUAllreduce::Enabled(const ParameterManager& param_manager,
                              const std::vector<TensorTableEntry>& entries,
                              const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

void CCLGPUAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    // TODO(Maozhou): replace with barrier
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

// Broadcast
Status CCLGPUBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  CheckTensorTableEntry(entries);
  auto first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  // On root rank, ccl broadcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (process_set.controller->GetRank() == first_entry.root_rank) {
    data_ptr = (void*)first_entry.tensor->data();
  } else {
    data_ptr = (void*)first_entry.output->data();
  }

  // cache
  auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl::broadcast(
        data_ptr,
        /* size */ first_entry.tensor->shape().num_elements() *
            DataType_Size(first_entry.tensor->dtype()),
        ccl::datatype::int8, first_entry.root_rank,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()), attr);
  });

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_BCAST,
                              *gpu_op_context_.stream);
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

bool CCLGPUBroadcast::Enabled(const ParameterManager& param_manager,
                              const std::vector<TensorTableEntry>& entries,
                              const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

void CCLGPUBroadcast::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    // TODO(Maozhou): replace with barrier
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

// Allgather
bool CCLGPUAllgather::Enabled(const ParameterManager& param_manager,
                              const std::vector<TensorTableEntry>& entries,
                              const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

Status CCLGPUAllgather::Execute(std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  CheckTensorTableEntry(entries);
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t*[entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t*[entries.size()];

  int global_size = process_set.controller->GetSize();

  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  global_state_->timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status =
      AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }

  global_state_->timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts,
                           entry_component_offsets);

  size_t element_size = DataType_Size(first_entry.tensor->dtype());

  const void* fused_input_data = nullptr;
  void* buffer_data;
  int64_t num_elements = NumElements(entries);
  int64_t gather_size = 0;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = const_cast<void*>(first_entry.output->data());
  }

  std::vector<size_t> rcounts(global_size);
  for (unsigned int rc = 0; rc < global_size; rc++) {
    rcounts[rc] = recvcounts[rc] * element_size;
    gather_size += rcounts[rc];
  }

  // Do allgather
  LOG(DEBUG) << "Do CCLGPUAllgather, number of elements: " << num_elements
             << ", dtype: " << DataType_Name(first_entry.tensor->dtype());
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl::allgatherv(
        fused_input_data != nullptr ? (void*)fused_input_data : buffer_data,
        num_elements * element_size, buffer_data, rcounts, ccl::datatype::int8,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()));
  });

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

void CCLGPUAllgather::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const int* displcmnts,
    int element_size, void*& buffer_data) {
  // Access first entry for retrieving context
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  int64_t offset = (int64_t)displcmnts[process_set.controller->GetRank()] *
                   (int64_t)element_size;

  // TODO(IOH): Enable batch memory copy for allgather
  // use the default AllgatherOps implementation
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
    offset += e.tensor->size();
  }
}

void CCLGPUAllgather::MemcpyOutFusionBuffer(
    const int64_t* const* entry_component_offsets,
    const int64_t* const* entry_component_sizes, const void* buffer_data,
    int element_size, std::vector<TensorTableEntry>& entries) {
  // TODO(IOH): Enable batch memory copy for allgather
  // use the default AllgatherOps implementation
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    int global_size = process_set.controller->GetSize();
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_size; ++rc) {
      int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
      int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
      const void* buffer_data_at_offset = (uint8_t*)buffer_data + entry_offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e, copy_offset,
                                 entry_size);
      copy_offset += entry_size;
    }
  }
}

Status CCLGPUAllgather::AllocateOutput(std::vector<TensorTableEntry>& entries,
                                       const Response& response,
                                       int64_t**& entry_component_sizes,
                                       int*& recvcounts) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    int global_size = process_set.controller->GetSize();
    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    int64_t total_entry_dimension_size = 0;
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_size; ++rc) {
      auto component_size = tensor_sizes[ec * global_size + rc];
      total_entry_dimension_size += component_size;

      if (recvcounts) {
        recvcounts[rc] += component_size * single_slice_shape.num_elements();
      }

      if (entry_component_sizes) {
        entry_component_sizes[ec][rc] =
            component_size * single_slice_shape.num_elements();
      }
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t)total_entry_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    Status status =
        e.context->AllocateOutput(e.output_index, output_shape, &e.output);
    if (!status.ok()) {
      LOG(WARNING) << "CCLGPUAllgather::AllocateOutput failed: "
                   << status.reason();
      return status;
    }
  }

  return Status::OK();
}

void CCLGPUAllgather::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

void CCLGPUAllgather::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      buffer_data_at_offset, e.tensor->data(), (size_t)e.tensor->size(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void CCLGPUAllgather::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e,
    int64_t entry_offset, size_t entry_size) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      (int8_t*)e.output->data() + entry_offset, buffer_data_at_offset,
      entry_size,
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

// Alltoall
bool CCLGPUAlltoall::Enabled(const ParameterManager& param_manager,
                             const std::vector<TensorTableEntry>& entries,
                             const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

Status CCLGPUAlltoall::Execute(std::vector<TensorTableEntry>& entries,
                               const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  std::vector<size_t> sdispls, rdispls;
  std::vector<size_t> sendcounts, recvcounts;
  Status status =
      PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  const void* sendbuf = e.tensor->data();
  void* buffer_data = (void*)e.output->data();

  // TODO(Pengfei): support empty sendbuf
  if (sendbuf == nullptr) {
    throw std::logic_error(
        "CCLGPUAlltoall with empty entry not implemented yet.");
  }

  // Do alltoall
  LOG(DEBUG) << "Do CCLGPUAlltoall";
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl::alltoallv(sendbuf, sendcounts, buffer_data, recvcounts,
                   GetCCLDataType(e.tensor),
                   ccl_op_context_.GetCCLComm(e, response.devices()),
                   ccl_op_context_.GetCCLStream(e, response.devices()));
  });

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_ALLTOALL,
                              *gpu_op_context_.stream);
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

void CCLGPUAlltoall::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

Status CCLGPUReducescatter::Execute(std::vector<TensorTableEntry>& entries,
                                    const Response& response) {
  CheckTensorTableEntry(entries);
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  size_t element_size = DataType_Size(first_entry.tensor->dtype());
  const void* fused_input_data = nullptr;
  void* buffer_data = nullptr;
  int global_rank = process_set.controller->GetRank();
  int global_size = process_set.controller->GetSize();
  auto output_shapes = ComputeOutputShapes(entries, global_size);
  std::vector<int> recvcounts = ComputeReceiveCounts(output_shapes);

  global_state_->timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, output_shapes[global_rank]);
  if (!status.ok()) {
    return status;
  }
  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, output_shapes, element_size, buffer_data);
    fused_input_data = buffer_data;

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
  }

  bool same_output_shape = (recvcounts.front() == recvcounts.back());

  if (same_output_shape) {
    // Do reducescatter.
    LOG(DEBUG) << "Do CCLGPUReducescatter, number of elements: "
               << NumElements(entries)
               << ", dtype: " << DataType_Name(first_entry.tensor->dtype());
    CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
      ccl::reduce_scatter(
          fused_input_data, buffer_data, recvcounts[0],
          GetCCLDataType(first_entry.tensor), ccl::reduction::sum,
          ccl_op_context_.GetCCLComm(first_entry, response.devices()),
          ccl_op_context_.GetCCLStream(first_entry, response.devices()));
    });

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_REDUCESCATTER,
                                *gpu_op_context_.stream);
    }
  } else {
    // Simulate "ReduceScatterV" by an equivalent group of reduces.
    // TODO(Pengfei): Assume oneCCL executes events in-order.
    LOG(DEBUG)
        << "Simulate 'ReduceScatterV' by an equivalent group of reduces.";
    size_t offset_bytes = 0;
    for (int recv_rank = 0; recv_rank < global_size; ++recv_rank) {
      const void* send_pointer =
          reinterpret_cast<const int8_t*>(fused_input_data) + offset_bytes;

      LOG(DEBUG) << "Receive rank: " << recv_rank
                 << " receive count: " << recvcounts[recv_rank];
      CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
        ccl::reduce(
            send_pointer, buffer_data, recvcounts[recv_rank],
            GetCCLDataType(first_entry.tensor), ccl::reduction::sum, recv_rank,
            ccl_op_context_.GetCCLComm(first_entry, response.devices()),
            ccl_op_context_.GetCCLStream(first_entry, response.devices()));
      });

      offset_bytes += recvcounts[recv_rank] * element_size;
    }

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_REDUCE,
                                *gpu_op_context_.stream);
    }
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

bool CCLGPUReducescatter::Enabled(const ParameterManager& param_manager,
                                  const std::vector<TensorTableEntry>& entries,
                                  const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

Status CCLGPUReducescatter::AllocateOutput(
    std::vector<TensorTableEntry>& entries,
    const std::vector<TensorShape>& output_shapes) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    const auto& output_shape = output_shapes[ec];

    Status status =
        e.context->AllocateOutput(e.output_index, output_shape, &e.output);
    if (!status.ok()) {
      LOG(WARNING) << "CCLGPUReducescatter::AllocateOutput failed: "
                   << status.reason();
      return status;
    }
  }

  return Status::OK();
}

void CCLGPUReducescatter::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    // TODO(Maozhou): replace with barrier
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

void CCLGPUReducescatter::MemcpyEntryInFusionBuffer(
    const TensorTableEntry& e, size_t entry_offset, size_t entry_size,
    void* buffer_data_at_offset) {
  void* tensor_data_at_offset = (uint8_t*)e.tensor->data() + entry_offset;
  gpu_context_->MemcpyAsyncD2D(
      buffer_data_at_offset, tensor_data_at_offset, entry_size,
      gpu_context_->streams[global_state_->current_nccl_stream][e.device]);
}

void CCLGPUReducescatter::MemcpyEntryOutFusionBuffer(
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  gpu_context_->MemcpyAsyncD2D(
      (void*)e.output->data(), buffer_data_at_offset, (size_t)e.output->size(),
      gpu_context_->streams[global_state_->current_nccl_stream][e.device]);
}

void CCLGPUReducescatter::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const std::vector<std::vector<TensorShape>>& output_shapes,
    std::size_t element_size, void*& buffer_data) {
  // TODO(IOH): Implement batch memory copy
  // use the default ReducescatterOps implementation
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  size_t buffer_offset = 0;
  std::vector<size_t> entry_offsets(entries.size(), 0);

  for (const auto& rank_shapes : output_shapes) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      auto& e = entries[ec];
      const auto& entry_shape = rank_shapes[ec];
      auto entry_offset = entry_offsets[ec];
      size_t entry_size = entry_shape.num_elements() * element_size;
      void* buffer_data_at_offset = (uint8_t*)buffer_data + buffer_offset;
      MemcpyEntryInFusionBuffer(e, entry_offset, entry_size,
                                buffer_data_at_offset);
      entry_offsets[ec] += entry_size;
      buffer_offset += entry_size;
    }
  }
}

void CCLGPUReducescatter::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  // TODO(IOH): Implement batch memory copy
  // use the default ReducescatterOps implementation
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryOutFusionBuffer(buffer_data_at_offset, e);
    offset += e.output->size();
  }
}

} // namespace common
} // namespace horovod
