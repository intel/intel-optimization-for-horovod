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

#ifndef HOROVOD_TORCH_READY_EVENT_H
#define HOROVOD_TORCH_READY_EVENT_H

#if HAVE_GPU
#if HAVE_CUDA
#include "cuda_runtime.h"
#elif HAVE_SYCL
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include "adapter_v2.h"
#endif // HAVE_CUDA
#endif

#include <memory>

#include "../common/common.h"

namespace horovod {
namespace torch {

using namespace horovod::common;

#if HAVE_GPU
class TorchReadyEvent : public ReadyEvent {
public:
  TorchReadyEvent(int device);
#if HAVE_SYCL
  TorchReadyEvent(const TorchReadyEvent& other) = delete;
  TorchReadyEvent& operator=(const TorchReadyEvent& other) = delete;
#endif
  ~TorchReadyEvent();
  virtual bool Ready() const override;
  gpuEvent_t event() const override;

private:
  int device_ = CPU_DEVICE_ID;
  // TODO(Maozhou): == nullptr?
  gpuEvent_t event_;
#if HAVE_SYCL
  bool is_enabled;
  TorchOpContext* ctx_;
#endif
};
#endif

std::shared_ptr<ReadyEvent> RecordReadyEvent(int device);

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_READY_EVENT_H
