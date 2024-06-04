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

#include "sycl/sycl_kernels.h"

namespace horovod {
namespace common {

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

  if (ccl_context_->ccl_comms[global_state_->current_nccl_stream].empty() ||
      !ccl_context_->ccl_comms[global_state_->current_nccl_stream].count(
          std::make_tuple(process_set_id, ccl_device_map))) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_CCL);

    int ccl_rank, ccl_size;
    Communicator ccl_id_bcast_comm;
    PopulateCCLCommStrategy(ccl_rank, ccl_size, ccl_id_bcast_comm, process_set);

    if (ccl_rank == 0) {
      if (!kvs_) {
        kvs_ = ccl::create_main_kvs();
      }
      auto main_addr = kvs_->get_address();
      process_set.controller->Bcast((void*)main_addr.data(), main_addr.size(),
                                    0, ccl_id_bcast_comm);
    } else {
      ccl::kvs::address_type main_addr;
      process_set.controller->Bcast((void*)main_addr.data(), main_addr.size(),
                                    0, ccl_id_bcast_comm);
      kvs_ = ccl::create_kvs(main_addr);
    }

    auto queue = stream;
    ccl_streams.push_back(ccl::create_stream(*queue));
    ccl_devices.push_back(ccl::create_device(queue->get_device()));
    ccl_contexts.push_back(ccl::create_context(queue->get_context()));
    // fill ccl_comms via creating communicator
    ccl_context_->ccl_comms[global_state_->current_nccl_stream].insert(
        {std::make_tuple(process_set_id, ccl_device_map),
         ccl4hvd{ccl_streams[0],
                 ccl::create_communicator(ccl_size, ccl_rank, ccl_devices[0],
                                          ccl_contexts[0], kvs_)}});

    process_set.controller->Barrier(Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);
  }

  ccl4hvd_ = &ccl_context_->ccl_comms[global_state_->current_nccl_stream].at(
      std::make_tuple(process_set_id, ccl_device_map));
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
  } else if (communicator_type_ == Communicator::CROSS) {
    ccl_rank = process_set.controller->GetCrossRank();
    ccl_size = process_set.controller->GetCrossSize();
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

Status CCLGPUOpContext::FinalizeGPUQueue(
    GPUOpContext* gpu_op_context_, GPUContext* gpu_context_,
    std::vector<TensorTableEntry>& entries,
    std::shared_ptr<std::vector<ccl::event>> ccl_events, bool free_host_buffer,
    const std::function<void()>& error_check_callback) {
  auto& evt_queue = gpu_op_context_->event_queue;
  auto current_stream = *(gpu_op_context_->stream);
  // Use completion marker via event because it's faster than
  // blocking gpuStreamSynchronize() in this thread.
  if (!global_state_->enable_async_completion) {
    gpu_context_->RecordEvent(evt_queue, "", current_stream);
  }
  auto& first_entry = entries[0];
  void* cpu_buffer = gpu_op_context_->host_buffer;
  auto& timeline = global_state_->timeline;
  GPUContext* gpu_context = gpu_context_;
  auto& global_state = global_state_;

  // Claim a std::shared_ptr to the fusion buffer to prevent its memory from
  // being reclaimed during finalization.
  auto fusion_buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);

  bool elastic = global_state_->elastic_enabled;
  bool enable_async_completion = global_state_->enable_async_completion;
  gpu_context_->finalizer_thread_pool.execute(
      [entries, first_entry, cpu_buffer, fusion_buffer, free_host_buffer,
       evt_queue, &timeline, gpu_context, error_check_callback, elastic,
       enable_async_completion, current_stream, &global_state,
       ccl_events]() mutable {
        gpu_context->SetDevice(first_entry.device);

        Event event;
        bool gpu_evt_failed = false;
        std::string gpu_evt_err_msg;
        if (!enable_async_completion || timeline.Initialized()) {
          // If timeline is enabled, wait for events on CPU for accurate
          // timings.
          if (elastic) {
            try {
              gpu_context->WaitForEventsElastic(evt_queue, entries, timeline,
                                                error_check_callback);
            } catch (std::exception& e) {
              // notify background loop to exit and reinit rather than just
              // aborting the program
              global_state->shut_down = true;
              gpu_evt_failed = true;
              gpu_evt_err_msg = e.what();
            }
          } else {
            gpu_context->WaitForEvents(evt_queue, entries, timeline,
                                       error_check_callback);
          }
        } else {
          gpu_context->ClearEvents(evt_queue, entries, timeline,
                                   error_check_callback, elastic);
          event = gpu_context->RecordEvent(current_stream);
        }

        if (free_host_buffer && cpu_buffer != nullptr) {
          free(cpu_buffer);
        }

        Status status;
        if (gpu_evt_failed) {
          status = Status::UnknownError(gpu_evt_err_msg);
        } else {
          status = Status::OK();
          status.event = event;
        }

        for (auto& e : entries) {
          timeline.End(e.tensor_name, e.output);
          e.FinishWithCallback(status);
        }
        if (enable_async_completion) {
          gpu_context->ReleaseEvent(event);
        }
      });

  // Update current stream
  global_state_->current_nccl_stream =
      (global_state_->current_nccl_stream + 1) %
      global_state_->num_nccl_streams;

  return Status::InProgress();
};

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
  auto ccl_events = std::make_shared<std::vector<ccl::event>>();
  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());
  LOG(DEBUG) << "Do CCLGPUAllreduce, number of elements: " << num_elements
             << ", dtype: " << DataType_Name(first_entry.tensor->dtype());
  auto ccl_event =
      ccl::allreduce(fused_input_data, buffer_data, (size_t)num_elements,
                     GetCCLDataType(first_entry.tensor), ccl_reduction_op,
                     ccl_op_context_.ccl4hvd_->ccl_comm_,
                     ccl_op_context_.ccl4hvd_->ccl_stream_, attr);
  ccl_events->push_back(std::move(ccl_event));

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

  return ccl_op_context_.FinalizeGPUQueue(
      &gpu_op_context_, gpu_context_, entries, ccl_events, true,
      ccl_op_context_.error_check_callback_);
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
    std::vector<gpuEvent_t> wait_list(event_set.cbegin(), event_set.cend());
    auto stream =
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device];
    stream->ext_oneapi_submit_barrier(wait_list);
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
  auto ccl_events = std::make_shared<std::vector<ccl::event>>();
  auto ccl_event =
      ccl::broadcast(data_ptr,
                     /* size */ first_entry.tensor->shape().num_elements() *
                         DataType_Size(first_entry.tensor->dtype()),
                     ccl::datatype::int8, first_entry.root_rank,
                     ccl_op_context_.ccl4hvd_->ccl_comm_,
                     ccl_op_context_.ccl4hvd_->ccl_stream_, attr);
  ccl_events->push_back(std::move(ccl_event));

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_BCAST,
                              *gpu_op_context_.stream);
  }

  return ccl_op_context_.FinalizeGPUQueue(
      &gpu_op_context_, gpu_context_, entries, ccl_events, true,
      ccl_op_context_.error_check_callback_);
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
    std::vector<gpuEvent_t> wait_list(event_set.cbegin(), event_set.cend());
    auto stream =
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device];
    stream->ext_oneapi_submit_barrier(wait_list);
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
  int global_rank = process_set.controller->GetRank();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  global_state_->timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes);
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

  auto element_size = (int)DataType_Size(first_entry.tensor->dtype());
  int padding_elements = 1;

  if (entries.size() > 1) {
    assert(BATCHED_D2D_PADDING % element_size == 0);
    padding_elements = BATCHED_D2D_PADDING / element_size;
  }

  SetRecvcounts(entry_component_sizes, entries.size(), global_size, recvcounts,
                padding_elements);
  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entry_component_sizes, recvcounts, entries.size(),
                           global_size, entry_component_offsets);

  const void* fused_input_data;
  void* buffer_data;
  int64_t num_elements = NumElements(entries);
  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    fused_input_data =
        (uint8_t*)buffer_data + displcmnts[global_rank] * element_size;

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
  }

  // Do allgather
  auto ccl_events = std::make_shared<std::vector<ccl::event>>();
  LOG(DEBUG) << "Do CCLGPUAllgather, number of elements: " << num_elements
             << ", dtype: " << DataType_Name(first_entry.tensor->dtype());
  auto ccl_event = ccl::allgatherv(fused_input_data, rcounts[global_rank],
                                   buffer_data, rcounts, ccl::datatype::int8,
                                   ccl_op_context_.ccl4hvd_->ccl_comm_,
                                   ccl_op_context_.ccl4hvd_->ccl_stream_);
  ccl_events->push_back(std::move(ccl_event));

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

  return ccl_op_context_.FinalizeGPUQueue(
      &gpu_op_context_, gpu_context_, entries, ccl_events, true,
      ccl_op_context_.error_check_callback_);
}

Status CCLGPUAllgather::AllocateOutput(std::vector<TensorTableEntry>& entries,
                                       const Response& response,
                                       int64_t**& entry_component_sizes) {
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

    std::shared_ptr<ReadyEvent> event;
    Status status = e.context->AllocateOutput(e.output_index, output_shape,
                                              &e.output, &event);
    if (!status.ok()) {
      LOG(WARNING) << "CCLGPUAllgather::AllocateOutput failed: "
                   << status.reason();
      return status;
    }

    // Add event dependency for output allocation to stream
    if (event) {
      (*gpu_op_context_.stream)->ext_oneapi_submit_barrier({event->event()});
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
    std::vector<gpuEvent_t> wait_list(event_set.cbegin(), event_set.cend());
    auto stream =
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device];
    stream->ext_oneapi_submit_barrier(wait_list);
  }
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

  // Do alltoall
  auto ccl_events = std::make_shared<std::vector<ccl::event>>();
  LOG(DEBUG) << "Do CCLGPUAlltoall";
  auto ccl_event = ccl::alltoallv(sendbuf, sendcounts, buffer_data, recvcounts,
                                  GetCCLDataType(e.tensor),
                                  ccl_op_context_.ccl4hvd_->ccl_comm_,
                                  ccl_op_context_.ccl4hvd_->ccl_stream_);
  ccl_events->push_back(std::move(ccl_event));

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_ALLTOALL,
                              *gpu_op_context_.stream);
  }

  return ccl_op_context_.FinalizeGPUQueue(
      &gpu_op_context_, gpu_context_, entries, ccl_events, true,
      ccl_op_context_.error_check_callback_);
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
    std::vector<gpuEvent_t> wait_list(event_set.cbegin(), event_set.cend());
    auto stream =
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device];
    stream->ext_oneapi_submit_barrier(wait_list);
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

  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  size_t element_size = DataType_Size(first_entry.tensor->dtype());
  const void* fused_input_data = nullptr;
  void* buffer_data = nullptr;
  int global_rank = process_set.controller->GetRank();
  int global_size = process_set.controller->GetSize();
  auto output_shapes = ComputeOutputShapes(entries, global_size);
  std::vector<int> recvcounts = ComputeReceiveCounts(output_shapes);

  // Copy memory into the fusion buffer. Execute prescaling op if necessary.
  if (entries.size() > 1 || prescale_factor != 1.0) {
    ScaleMemcpyInFusionBuffer(entries, output_shapes, element_size, buffer_data,
                              prescale_factor);
    fused_input_data = buffer_data;
    if (entries.size() == 1) {
      // Unfused prescaled: Send from fusion buffer, receive at output tensor
      buffer_data = (void*)first_entry.output->data();
    } else {
      buffer_data = reinterpret_cast<int8_t*>(buffer_data) +
                    global_rank * recvcounts[0] * element_size;
    }

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    // Unfused without prescaling
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
  }

  bool same_output_shape = (recvcounts.front() == recvcounts.back());

  auto ccl_events = std::make_shared<std::vector<ccl::event>>();
  if (same_output_shape) {
    // Do reducescatter.
    LOG(DEBUG) << "Do CCLGPUReducescatter, number of elements: "
               << NumElements(entries)
               << ", dtype: " << DataType_Name(first_entry.tensor->dtype());
    auto ccl_event = ccl::reduce_scatter(
        fused_input_data, buffer_data, recvcounts[0],
        GetCCLDataType(first_entry.tensor), ccl::reduction::sum,
        ccl_op_context_.ccl4hvd_->ccl_comm_,
        ccl_op_context_.ccl4hvd_->ccl_stream_);
    ccl_events->push_back(std::move(ccl_event));

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_REDUCESCATTER,
                                *gpu_op_context_.stream);
    }
  } else {
    // Simulate "ReduceScatterV" by an equivalent group of reduces.
    LOG(DEBUG)
        << "Simulate 'ReduceScatterV' by an equivalent group of reduces.";
    size_t offset_bytes = 0;
    for (int recv_rank = 0; recv_rank < global_size; ++recv_rank) {
      const void* send_pointer =
          reinterpret_cast<const int8_t*>(fused_input_data) + offset_bytes;

      LOG(DEBUG) << "Receive rank: " << recv_rank
                 << " receive count: " << recvcounts[recv_rank];
      auto ccl_event =
          ccl::reduce(send_pointer, buffer_data, recvcounts[recv_rank],
                      GetCCLDataType(first_entry.tensor), ccl::reduction::sum,
                      recv_rank, ccl_op_context_.ccl4hvd_->ccl_comm_,
                      ccl_op_context_.ccl4hvd_->ccl_stream_);
      ccl_events->push_back(std::move(ccl_event));

      offset_bytes += recvcounts[recv_rank] * element_size;
    }

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_REDUCE,
                                *gpu_op_context_.stream);
    }
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, postscale_factor, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    if (postscale_factor != 1.0) {
      // Execute postscaling ops
      for (auto& e : entries) {
        ScaleBuffer(postscale_factor, entries, e.output->data(),
                    const_cast<void*>(e.output->data()),
                    e.output->shape().num_elements());
      }
    }
  }

  return ccl_op_context_.FinalizeGPUQueue(
      &gpu_op_context_, gpu_context_, entries, ccl_events, true,
      ccl_op_context_.error_check_callback_);
}

bool CCLGPUReducescatter::Enabled(const ParameterManager& param_manager,
                                  const std::vector<TensorTableEntry>& entries,
                                  const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
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
    std::vector<gpuEvent_t> wait_list(event_set.cbegin(), event_set.cend());
    auto stream =
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device];
    stream->ext_oneapi_submit_barrier(wait_list);
  }
}

// TorusAllreduce
bool CCLGPUTorusAllreduce::Enabled(const ParameterManager& param_manager,
                                   const std::vector<TensorTableEntry>& entries,
                                   const Response& response) const {
  if (!GPUAllreduce::Enabled(param_manager, entries, response)) {
    return false;
  }
  return param_manager.TorusAllreduce();
}

Status CCLGPUTorusAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                     const Response& response) {
  auto& first_entry = entries.at(0);
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> local_ccl_device_map;
  local_ccl_device_map.reserve(
      process_set.controller->GetLocalCommRanks().size());
  for (int rank : process_set.controller->GetLocalCommRanks()) {
    int32_t device = response.devices().at(rank);
    local_ccl_device_map.push_back(device);
  }
  std::vector<int32_t> cross_ccl_device_map(
      {response.devices()[process_set.controller->GetCrossRank()]});

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  local_ccl_op_context_.InitCCLComm(stream, entries, local_ccl_device_map);
  cross_ccl_op_context_.InitCCLComm(stream, entries, cross_ccl_device_map);

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

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy tensors into the fusion buffer.
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
    buffer_data = (void*)first_entry.output->data();
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

  // Do allreduce.
  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());
  int element_size = DataType_Size(first_entry.tensor->dtype());
  int local_size = process_set.controller->GetLocalSize();
  int local_rank = process_set.controller->GetLocalRank();

  LOG(DEBUG) << "Do CCLGPUTorusAllreduce, number of elements: " << num_elements
             << ", dtype: " << DataType_Name(first_entry.tensor->dtype());
  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (process_set.controller->IsHomogeneous() && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = local_size * FUSION_BUFFER_ATOMIC_UNIT;
    num_elements = ((num_elements + div - 1) / div) * div;
    buffer_len = num_elements * element_size;
  }

  // Split the elements into two groups: num_elements_per_rank*local_size,
  // and num_elements_remaining. Cross-node reduction for the first group
  // is done by all local_rank's in parallel, while for the second group
  // it it is only done by the root_rank. If the cluster is not
  // homogeneous first group is zero, and root_rank is 0.

  // Homogeneous case:
  // For the part of data divisible by local_size, perform CCL
  // ReduceScatter - Parallelized CCL Allreduce - CCL Allgather. For the
  // non-divisible part (if any), do CCL Reduce (at rank local_size-1),
  // CCL Allreduce (across rank (local_size-1)'s), and CCL Bcast.

  int64_t num_elements_per_rank =
      process_set.controller->IsHomogeneous() ? num_elements / local_size : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_rank;

  int64_t num_elements_remaining = process_set.controller->IsHomogeneous()
                                       ? num_elements % local_size
                                       : num_elements;

  void* buffer_data_remainder =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_size;

  void* fused_input_data_remainder =
      (uint8_t*)fused_input_data + buffer_len_per_rank * local_size;

  int root_rank = process_set.controller->IsHomogeneous() ? local_size - 1 : 0;
  bool is_root_rank = local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;

  auto& timeline = global_state_->timeline;
  auto ccl_events = std::make_shared<std::vector<ccl::event>>();
  if (num_elements_per_rank > 0) {
    auto ccl_event = ccl::reduce_scatter(
        fused_input_data, buffer_data_at_rank_offset,
        (size_t)num_elements_per_rank, GetCCLDataType(first_entry.tensor),
        ccl_reduction_op, local_ccl_op_context_.ccl4hvd_->ccl_comm_,
        local_ccl_op_context_.ccl4hvd_->ccl_stream_);
    ccl_events->push_back(std::move(ccl_event));
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_REDUCESCATTER,
                                *gpu_op_context_.stream);
    }
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    auto ccl_event = ccl::reduce(
        fused_input_data_remainder, buffer_data_remainder,
        (size_t)num_elements_remaining, GetCCLDataType(first_entry.tensor),
        ccl_reduction_op, root_rank, local_ccl_op_context_.ccl4hvd_->ccl_comm_,
        local_ccl_op_context_.ccl4hvd_->ccl_stream_);
    ccl_events->push_back(std::move(ccl_event));
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_REDUCE,
                                *gpu_op_context_.stream);
    }
  }

  if ((process_set.controller->IsHomogeneous() || is_root_rank) &&
      total_num_elements > 0) {
    timeline.ActivityStartAll(entries, CCL_ALLREDUCE);
    auto ccl_event = ccl::allreduce(
        buffer_data_at_rank_offset, buffer_data_at_rank_offset,
        (size_t)total_num_elements, GetCCLDataType(first_entry.tensor),
        ccl_reduction_op, cross_ccl_op_context_.ccl4hvd_->ccl_comm_,
        cross_ccl_op_context_.ccl4hvd_->ccl_stream_);
    ccl_events->push_back(std::move(ccl_event));
    timeline.ActivityEndAll(entries);
  }
  if (num_elements_per_rank > 0) {
    std::vector<size_t> recv_counts(local_size);
    for (unsigned int rc = 0; rc < local_size; rc++) {
      recv_counts[rc] = num_elements_per_rank;
    }
    auto ccl_event = ccl::allgatherv(
        buffer_data_at_rank_offset, (size_t)num_elements_per_rank, buffer_data,
        recv_counts, GetCCLDataType(first_entry.tensor),
        local_ccl_op_context_.ccl4hvd_->ccl_comm_,
        local_ccl_op_context_.ccl4hvd_->ccl_stream_);
    ccl_events->push_back(std::move(ccl_event));
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_ALLGATHER,
                                *gpu_op_context_.stream);
    }
  }
  if (num_elements_remaining > 0) {
    auto ccl_event =
        ccl::broadcast(buffer_data_remainder, (size_t)num_elements_remaining,
                       GetCCLDataType(first_entry.tensor), root_rank,
                       local_ccl_op_context_.ccl4hvd_->ccl_comm_,
                       local_ccl_op_context_.ccl4hvd_->ccl_stream_);
    ccl_events->push_back(std::move(ccl_event));
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_BCAST,
                                *gpu_op_context_.stream);
    }
  }

  // Copy tensors out of the fusion buffer.
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

  return local_ccl_op_context_.FinalizeGPUQueue(
      &gpu_op_context_, gpu_context_, entries, ccl_events, true,
      local_ccl_op_context_.error_check_callback_);
}

void CCLGPUTorusAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    std::vector<gpuEvent_t> wait_list(event_set.cbegin(), event_set.cend());
    auto stream =
        gpu_context_
            ->streams[global_state_->current_nccl_stream][entries[0].device];
    stream->ext_oneapi_submit_barrier(wait_list);
  }
}
} // namespace common
} // namespace horovod
