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

#ifndef HOROVOD_TORCH_DEVICE_UTIL_H
#define HOROVOD_TORCH_DEVICE_UTIL_H

#if HAVE_SYCL
#include <c10/core/Device.h>
using dev_index_t = c10::DeviceIndex;
#else
using dev_index_t = int;
#endif

#include "../common/common.h"

namespace horovod {
namespace torch {

class with_device {
public:
  with_device(dev_index_t device);
  ~with_device();

private:
  dev_index_t restore_device_ = CPU_DEVICE_ID;
};

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_DEVICE_UTIL_H
