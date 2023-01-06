// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright (C) 2022 Intel Corporation.
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

#if HAVE_GPU
#if HAVE_CUDA
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <mutex>
#include <queue>
#include <unordered_map>
#else
#include <stdexcept>
#endif // HAVE_CUDA
#endif

#include "device_util.h"
#include "ready_event.h"

namespace horovod {
namespace torch {

#if HAVE_GPU
struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<gpuEvent_t>> gpu_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

#if HAVE_CUDA
TorchReadyEvent::TorchReadyEvent(int device) : device_(device) {
  assert(device_ != CPU_DEVICE_ID);

  with_device device_context(device_);
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    if (!queue.empty()) {
      event_ = queue.front();
      queue.pop();
    } else {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(
          &event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  auto stream = c10::cuda::getCurrentCUDAStream(device_);
  C10_CUDA_CHECK(cudaEventRecord(event_, stream));
}

TorchReadyEvent::~TorchReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(event_);
  }
}
#elif HAVE_SYCL
TorchReadyEvent::TorchReadyEvent(int device) : device_(device) {
  ctx_ = new TorchOpContext(device_);
  event_ = ctx_->SYCLQueue().ext_oneapi_submit_barrier();
}

TorchReadyEvent::~TorchReadyEvent() { delete ctx_; }
#endif

bool TorchReadyEvent::Ready() const {
#if HAVE_CUDA
  C10_CUDA_CHECK(cudaEventSynchronize(event_));
  return true;
#elif HAVE_SYCL
  return event_.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
#endif
}

gpuEvent_t TorchReadyEvent::event() const {
  return event_;
}
#endif

// On GPU this event will signal that GPU computations are done and data is
// ready.
std::shared_ptr<ReadyEvent> RecordReadyEvent(int device) {
  if (device == CPU_DEVICE_ID) {
    return std::shared_ptr<ReadyEvent>();
  } else {
#if HAVE_GPU
    return std::make_shared<TorchReadyEvent>(device);
#else
    throw std::logic_error(
        "Internal error. Requested ReadyEvent "
        "with GPU device but not compiled with CUDA or SYCL.");
#endif
  }
}

} // namespace torch
} // namespace horovod
