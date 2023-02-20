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

#include "ccl_gpu_operations.h"

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
    throw("TensorTableEntry is empty!");
  }
}

int CCLGPUContext::GetGpuEvent(Event* event, gpuStream_t& stream) {
  std::mutex mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);

    auto& queue = sycl_events[*stream];
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return 0;
    }
  }

  event->event = std::make_shared<gpuEvent_t>();
  event->stream = stream;

  return 0;
}

gpuError_t CCLGPUContext::ReleaseGpuEvent(Event event) {
  std::mutex mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = sycl_events[*event.stream];
    queue.push(event);
  }
  return sycl::errc::success;
}

void CCLGPUContext::RecordEvent(
    std::queue<std::pair<std::string, Event>>& event_queue, std::string name,
    gpuStream_t& stream) {
  Event event;
  GetGpuEvent(&event, stream);
  *(event.event) = stream->ext_oneapi_submit_barrier();
  event_queue.emplace(name, event);
}

Event CCLGPUContext::RecordEvent(gpuStream_t& stream) {
  Event event;
  GetGpuEvent(&event, stream);
  *(event.event) = stream->ext_oneapi_submit_barrier();
  return event;
}

void CCLGPUContext::WaitForEvents(
    std::queue<std::pair<std::string, Event>>& event_queue,
    const std::vector<TensorTableEntry>& entries, Timeline& timeline,
    const std::function<void()>& error_check_callback) {
  while (!event_queue.empty()) {
    std::string name;
    Event event;
    std::tie(name, event) = event_queue.front();
    event_queue.pop();
    if (name != "") {
      timeline.ActivityStartAll(entries, name);
    }
    CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
      // TODO: need ccl_event for elastic
      auto events_it = sycl_2_ccl_event_map.find(*(event.event));
      if (events_it != sycl_2_ccl_event_map.end()) {
        // TODO(Maozhou): necessary wait here?
        events_it->second->wait();
        sycl_2_ccl_event_map.erase(events_it);
      }
    });
    event.event->wait();
    if (name != "") {
      timeline.ActivityEndAll(entries);
    }
    ReleaseGpuEvent(event);
  }
}

void CCLGPUContext::ClearEvents(
    std::queue<std::pair<std::string, Event>>& event_queue,
    const std::vector<TensorTableEntry>& entries, Timeline& timeline,
    const std::function<void()>& error_check_callback, bool elastic) {
  while (!event_queue.empty()) {
    std::string name;
    Event event;
    std::tie(name, event) = event_queue.front();
    event_queue.pop();
    if (name != "") {
      timeline.ActivityStartAll(entries, name);
    }

    if (name != "") {
      timeline.ActivityEndAll(entries);
    }
    ReleaseGpuEvent(event);
  }
}

void CCLGPUContext::StreamCreate(const TensorTableEntry& e,
                                 gpuStream_t& stream) {
  auto org_q = e.context->SYCLQueue();
  auto property_list = sycl::property_list{sycl::property::queue::in_order()};
  stream.reset(
      new sycl::queue(org_q.get_context(), org_q.get_device(), property_list));
  if (!(stream)->is_in_order()) {
    throw std::runtime_error("SYCL queue should be with in_order property");
  }
}

void CCLGPUContext::Initialize(HorovodGlobalState& state) {
  ccl::init();

  enable_cache = GetBoolEnvOrDefault(HOROVOD_CCL_CACHE, false);
  LOG(INFO) << "CCLGPUContext initialized: \n"
            << "enable_cache " << enable_cache << "\n";

  auto fin_thread_count_env = std::getenv(HOROVOD_CCL_FIN_THREAD_COUNT);
  int fin_thread_count = 1;
  if (fin_thread_count_env != nullptr &&
      std::stol(fin_thread_count_env, nullptr, 10) > 0) {
    fin_thread_count = std::atoi(fin_thread_count_env);
  }

  state.num_ccl_streams = fin_thread_count;
  finalizer_thread_pool.create(state.num_ccl_streams);
  auto fin_thread_affinity_env = std::getenv(HOROVOD_CCL_FIN_THREAD_AFFINITY);
  if (fin_thread_affinity_env) {
    LOG(INFO) << "fin_thread_affinity " << fin_thread_affinity_env;
  }

  int local_size = state.global_controller->GetLocalSize();
  int local_rank = state.global_controller->GetLocalRank();

  if (fin_thread_affinity_env) {
    for (int i = 0; i < fin_thread_count; i++) {
      finalizer_thread_pool.execute([&]() mutable {
        parse_and_set_affinity(fin_thread_affinity_env,
                               fin_thread_count * local_size,
                               fin_thread_count * local_rank + i);
      });
    }
  }
  ccl_comms.resize(state.num_ccl_streams);
  streams.resize(state.num_ccl_streams);
}

void CCLGPUContext::Finalize(HorovodGlobalState& state) {
  finalizer_thread_pool.reset();
  ccl_comms.clear();
}

CCLGPUOpContext::~CCLGPUOpContext() {
  ccl_streams.clear();
  ccl_devices.clear();
  ccl_contexts.clear();
}

void CCLGPUOpContext::InitGPU(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];

  LOG(DEBUG) << "CCLGPUOpContext:: #entries: " << entries.size() << " device "
             << first_entry.device;

  if (ccl_context_->streams[global_state_->current_ccl_stream].empty()) {
    ccl_context_->StreamCreate(first_entry, stream);
    ccl_context_->streams[global_state_->current_ccl_stream].insert(
        {first_entry.device, stream});
  }
}

void CCLGPUOpContext::InitGPUQueue(const std::vector<TensorTableEntry>& entries,
                                   const Response& response) {
  event_queue = std::queue<std::pair<std::string, Event>>();
  stream = ccl_context_
               ->streams[global_state_->current_ccl_stream][entries[0].device];

  if (global_state_->timeline.Initialized()) {
    ccl_context_->RecordEvent(event_queue, QUEUE, stream);
  }
}

Status CCLGPUOpContext::FinalizeGPUQueue(
    std::vector<TensorTableEntry>& entries, ccl_data ccl_util_data,
    bool free_host_buffer /*= true*/,
    const std::function<void()>& error_check_callback) {
  // Use completion marker via event because it's faster than
  // blocking gpuStreamSynchronize() in this thread.
  if (!global_state_->enable_async_completion) {
    CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
      ccl_context_->RecordEvent(event_queue, "", stream);
      ccl_context_->sycl_2_ccl_event_map.emplace(
          *(event_queue.back().second.event), ccl_util_data.ccl_event);
    });
  }
  auto& first_entry = entries[0];
  void* cpu_buffer = host_buffer;
  auto& evt_queue = event_queue;
  auto& timeline = global_state_->timeline;
  auto& ccl_context = ccl_context_;

  // Claim a std::shared_ptr to the fusion buffer to prevent its memory from
  // being reclaimed during finalization.
  auto fusion_buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_ccl_stream); // TODO

  bool elastic = global_state_->elastic_enabled;
  bool enable_async_completion = global_state_->enable_async_completion;
  auto current_stream = stream;
  ccl_context_->finalizer_thread_pool.execute(
      [entries, first_entry, cpu_buffer, fusion_buffer, free_host_buffer,
       evt_queue, &timeline, &ccl_context, error_check_callback, elastic,
       enable_async_completion, current_stream, ccl_util_data]() mutable {
        Event event;
        if (!enable_async_completion || timeline.Initialized()) {
          ccl_context->WaitForEvents(evt_queue, entries, timeline,
                                     error_check_callback);
        } else {
          // TODO: non-validated yet
          ccl_context->ClearEvents(evt_queue, entries, timeline,
                                   error_check_callback, elastic);
          event = ccl_context->RecordEvent(current_stream);
        }

        // TODO: drop it, if don't need
        if (free_host_buffer && cpu_buffer != nullptr) {
          free(cpu_buffer);
        }

        for (auto& e : entries) {
          timeline.End(e.tensor_name, e.output);
          auto status = Status::OK();
          status.event = event;
          e.FinishWithCallback(status);
        }
        if (enable_async_completion) {
          // TODO(Maozhou): ErrorCheck?
          ccl_context->ReleaseGpuEvent(event);
        }
      });

  // Update current stream
  global_state_->current_ccl_stream =
      (global_state_->current_ccl_stream + 1) % global_state_->num_ccl_streams;

  return Status::InProgress();
}

void CCLGPUOpContext::InitCCLComm(const std::vector<TensorTableEntry>& entries,
                                  const std::vector<int32_t>& ccl_device_map) {
  CheckTensorTableEntry(entries);

  auto process_set_id = entries[0].process_set_id;
  auto& process_set = global_state_->process_set_table.Get(process_set_id);

  if (ccl_context_->ccl_comms[global_state_->current_ccl_stream].empty()) {
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

    {
      std::lock_guard<std::mutex> lock(CCLGPUContext::GlobalMutex);
      auto queue = stream;
      ccl_streams.push_back(ccl::create_stream(*queue));
      ccl_devices.push_back(ccl::create_device(queue->get_device()));
      ccl_contexts.push_back(ccl::create_context(queue->get_context()));
      // fill ccl_comms via creating communicator
      ccl_context_->ccl_comms[global_state_->current_ccl_stream].insert(
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
  return ccl_context_->ccl_comms[global_state_->current_ccl_stream]
      .at(std::make_tuple(entry.process_set_id, devices))
      .ccl_comm_;
}

ccl::stream&
CCLGPUOpContext::GetCCLStream(const TensorTableEntry& entry,
                              const std::vector<int32_t>& devices) {
  return ccl_context_->ccl_comms[global_state_->current_ccl_stream]
      .at(std::make_tuple(entry.process_set_id, devices))
      .ccl_stream_;
}

// Allreduce
Status CCLGPUAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  CheckTensorTableEntry(entries);
  auto& first_entry = entries[0];

  ccl_op_context_.InitGPU(entries);
  ccl_op_context_.InitCCLComm(entries, response.devices());
  ccl_op_context_.InitGPUQueue(entries, response);

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
      ccl_context_->RecordEvent(ccl_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                ccl_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = const_cast<void*>(first_entry.output->data());
    buffer_len = (size_t)first_entry.output->size();
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (prescale_factor != 1.0) {
      // Execute prescaling op
      ScaleBuffer(prescale_factor, entries, fused_input_data,
                  buffer_data, num_elements);
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
  std::shared_ptr<ccl::event> ccl_event;
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl_event = std::make_shared<ccl::event>(ccl::allreduce(
        fused_input_data, buffer_data, (size_t)num_elements,
        GetCCLDataType(first_entry.tensor), ccl_reduction_op,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()), attr));
  });

  if (global_state_->timeline.Initialized()) {
    ccl_context_->RecordEvent(ccl_op_context_.event_queue, CCL_ALLREDUCE,
                              ccl_op_context_.stream);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, buffer_len,
                               postscale_factor, entries);
    if (global_state_->timeline.Initialized()) {
      ccl_context_->RecordEvent(ccl_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                ccl_op_context_.stream);
    }
  } else {
    if (postscale_factor != 1.0) {
      // Execute postscaling op
      ScaleBuffer(postscale_factor, entries, buffer_data,
                  buffer_data, num_elements);
    }
  }

  return ccl_op_context_.FinalizeGPUQueue(
      entries, ccl_data{ccl_event}, true,
      ccl_op_context_.error_check_callback_);
}

void CCLGPUAllreduce::ScaleMemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
    void*& buffer_data, size_t& buffer_len, double scale_factor) {
  auto& first_entry = entries[0];
  // Access the fusion buffer.
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_ccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));
  auto stream =
      ccl_context_
          ->streams[global_state_->current_ccl_stream][first_entry.device];

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
      ccl_context_
          ->streams[global_state_->current_ccl_stream][first_entry.device];

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
      ccl_context_
          ->streams[global_state_->current_ccl_stream][first_entry.device]);
}

void CCLGPUAllreduce::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      (void*)e.output->data(), buffer_data_at_offset, (size_t)e.tensor->size(),
      ccl_context_
          ->streams[global_state_->current_ccl_stream][first_entry.device]);
}

void CCLGPUAllreduce::ScaleBuffer(double scale_factor,
                                  const std::vector<TensorTableEntry>& entries,
                                  const void* fused_input_data,
                                  void* buffer_data, int64_t num_elements) {
  gpu_context_->ScaleBufferImpl(
      fused_input_data, buffer_data, num_elements, scale_factor,
      entries[0].tensor->dtype(),
      ccl_context_
          ->streams[global_state_->current_ccl_stream][entries[0].device]);
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

  ccl_op_context_.InitGPU(entries);
  ccl_op_context_.InitCCLComm(entries, response.devices());
  ccl_op_context_.InitGPUQueue(entries, response);

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
  std::shared_ptr<ccl::event> ccl_event;
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl_event = std::make_shared<ccl::event>(ccl::broadcast(
        data_ptr,
        /* size */ first_entry.tensor->shape().num_elements() *
            DataType_Size(first_entry.tensor->dtype()),
        ccl::datatype::int8, first_entry.root_rank,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()), attr));
  });

  if (global_state_->timeline.Initialized()) {
    ccl_context_->RecordEvent(ccl_op_context_.event_queue, CCL_BCAST,
                              ccl_op_context_.stream);
  }

  return ccl_op_context_.FinalizeGPUQueue(
      entries, ccl_data{ccl_event}, true,
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

  ccl_op_context_.InitGPU(entries);
  ccl_op_context_.InitCCLComm(entries, response.devices());
  ccl_op_context_.InitGPUQueue(entries, response);

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
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  size_t element_size = DataType_Size(first_entry.tensor->dtype());

  const void* fused_input_data = nullptr;
  void* buffer_data;
  int64_t num_elements = NumElements(entries);
  int64_t gather_size = 0;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);

    if (global_state_->timeline.Initialized()) {
      ccl_context_->RecordEvent(ccl_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                ccl_op_context_.stream);
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
  std::shared_ptr<ccl::event> ccl_event;
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl_event = std::make_shared<ccl::event>(ccl::allgatherv(
        fused_input_data != nullptr ? (void *)fused_input_data : buffer_data, 
        num_elements * element_size, buffer_data, rcounts,
        ccl::datatype::int8,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices())));
  });

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    if (global_state_->timeline.Initialized()) {
      ccl_context_->RecordEvent(ccl_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                ccl_op_context_.stream);
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
      entries, ccl_data{ccl_event}, true,
      ccl_op_context_.error_check_callback_);
}

void CCLGPUAllgather::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const int* displcmnts,
    int element_size, void*& buffer_data) {
  // Access first entry for retrieving context
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_ccl_stream);
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
    auto& process_set =
        global_state_->process_set_table.Get(e.process_set_id);
    int global_size = process_set.controller->GetSize();
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_size; ++rc) {
      int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
      int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
      const void* buffer_data_at_offset =
          (uint8_t*)buffer_data + entry_offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e,
                                  copy_offset, entry_size);
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

    Status status = e.context->AllocateOutput(e.output_index, output_shape,
                                              &e.output);
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
      ccl_context_
          ->streams[global_state_->current_ccl_stream][first_entry.device]);
}

void CCLGPUAllgather::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e,
    int64_t entry_offset, size_t entry_size) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      (int8_t*)e.output->data() + entry_offset, buffer_data_at_offset,
      entry_size,
      ccl_context_
          ->streams[global_state_->current_ccl_stream][first_entry.device]);
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
  
  ccl_op_context_.InitGPU(entries);
  ccl_op_context_.InitCCLComm(entries, response.devices());
  ccl_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  std::vector<size_t> sdispls, rdispls;
  std::vector<size_t> sendcounts, recvcounts;
  Status status = PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  const void* sendbuf = e.tensor->data();
  void* buffer_data = (void*) e.output->data();

  // TODO(Pengfei): support empty sendbuf
  if (sendbuf == nullptr) {
    throw std::logic_error(
            "CCLGPUAlltoall with empty entry not implemented yet.");
  }

  // Do alltoall
  LOG(DEBUG) << "Do CCLGPUAlltoall";
  std::shared_ptr<ccl::event> ccl_event;
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl_event = std::make_shared<ccl::event>(ccl::alltoallv(
        sendbuf, sendcounts, buffer_data, recvcounts,
        GetCCLDataType(e.tensor), 
        ccl_op_context_.GetCCLComm(e, response.devices()),
        ccl_op_context_.GetCCLStream(e, response.devices())));
  });

  if (global_state_->timeline.Initialized()) {
    ccl_context_->RecordEvent(ccl_op_context_.event_queue, CCL_ALLTOALL,
                              ccl_op_context_.stream);
  }

  return ccl_op_context_.FinalizeGPUQueue(
      entries, ccl_data{ccl_event}, true,
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
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

} // namespace common
} // namespace horovod
