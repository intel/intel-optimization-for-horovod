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

#ifndef HOROVOD_SYCL_KERNELS_H
#define HOROVOD_SYCL_KERNELS_H

#include "../../common.h"
#include "../../global_state.h"
#include "../../message.h"

#define BATCHED_D2D_CAPACITY 162
#define UNSIGNED_INT32_MAX std::numeric_limits<uint32_t>::max()

namespace horovod {
namespace common {

struct BatchedD2DParams {
  void* buffers[BATCHED_D2D_CAPACITY];
  uint32_t offsets[BATCHED_D2D_CAPACITY];
};

// Performs a batched d2d memcopy
void BatchedScaledD2DMemcpyInImpl(BatchedD2DParams& params, void* fusion_buffer,
                                  int num_copies, double scale_factor,
                                  DataType dtype, gpuStream_t stream);

void BatchedScaledD2DMemcpyOutImpl(BatchedD2DParams& params,
                                   void* fusion_buffer, int num_copies,
                                   double scale_factor, DataType dtype,
                                   gpuStream_t stream);

bool enableBatchedScaledD2DMemcpy(HorovodGlobalState* global_state,
                                  const std::vector<TensorTableEntry>& entries);
} // namespace common
} // namespace horovod

#endif // HOROVOD_SYCL_KERNELS_H
