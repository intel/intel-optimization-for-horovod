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

#include "sycl_kernels.h"

// TODO(Fengqing):bfloat16 is only supported by dpcpp,
// not a SYCL official solution.
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
using bfloat16 = sycl::ext::oneapi::bfloat16;
#elif __has_include(<sycl/ext/oneapi/experimental/bfloat16.hpp>)
using bfloat16 = sycl::ext::oneapi::experimental::bfloat16;
#else
#error "compiler unsupports bfloat16"
#endif

namespace horovod {
namespace common {

template <typename T, typename TS> struct BatchedScaledMemcpyInKernel {
  BatchedScaledMemcpyInKernel(BatchedD2DParams params, T* fusion_buffer,
                              TS scale_factor, int groups_per_copy)
      : params_(params), fusion_buffer_(fusion_buffer),
        scale_factor_(scale_factor), groups_per_copy_(groups_per_copy) {}
  void operator()(sycl::nd_item<1> item) const {
    size_t local_id = item.get_local_id(0);
    size_t group_size = item.get_local_range(0);
    size_t group_id = item.get_group(0);

    const size_t idx = group_size * (group_id % groups_per_copy_) + local_id;
    int cur_index = group_id / groups_per_copy_;
    size_t num_words;
    size_t offset;
    if (cur_index >= 1) {
      offset = params_.offsets[cur_index - 1];
      num_words = params_.offsets[cur_index] - offset;
    } else {
      offset = 0;
      num_words = params_.offsets[0];
    }

    T* input = static_cast<T*>(params_.buffers[cur_index]);
    for (size_t i = idx; i < num_words; i += group_size * groups_per_copy_) {
      fusion_buffer_[i + offset] = input[i] * scale_factor_;
    }
  }

private:
  BatchedD2DParams params_;
  T* fusion_buffer_;
  TS scale_factor_;
  int groups_per_copy_;
};

template <typename T, typename TS> struct BatchedScaledMemcpyOutKernel {
  BatchedScaledMemcpyOutKernel(BatchedD2DParams& params, T* fusion_buffer,
                               TS scale_factor, int groups_per_copy)
      : params_(params), fusion_buffer_(fusion_buffer),
        scale_factor_(scale_factor), groups_per_copy_(groups_per_copy) {}
  void operator()(sycl::nd_item<1> item) const {
    size_t local_id = item.get_local_id(0);
    size_t group_size = item.get_local_range(0);
    size_t group_id = item.get_group(0);

    const size_t idx = group_size * (group_id % groups_per_copy_) + local_id;
    int cur_index = group_id / groups_per_copy_;
    size_t num_words;
    size_t offset;
    if (cur_index >= 1) {
      offset = params_.offsets[cur_index - 1];
      num_words = params_.offsets[cur_index] - offset;
    } else {
      offset = 0;
      num_words = params_.offsets[0];
    }

    T* output = static_cast<T*>(params_.buffers[cur_index]);
    for (size_t i = idx; i < num_words; i += group_size * groups_per_copy_) {
      output[i] = fusion_buffer_[i + offset] * scale_factor_;
    }
  }

private:
  BatchedD2DParams params_;
  T* fusion_buffer_;
  TS scale_factor_;
  int groups_per_copy_;
};

#define GROUPS_PER_COPY_D2D_KERNEL 8
template <typename T, typename TS>
void BatchedScaledD2DMemcpy(BatchedD2DParams params, void* fusion_buffer,
                            int num_copies, TS scale_factor, gpuStream_t stream,
                            bool fusion_in = true) {
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int32_t num_workitems =
      max_group_size * num_copies * GROUPS_PER_COPY_D2D_KERNEL;
  if (fusion_in) {
    stream->submit([&](sycl::handler& cgh) {
      BatchedScaledMemcpyInKernel<T, TS> task(
          params, static_cast<T*>(fusion_buffer), scale_factor,
          GROUPS_PER_COPY_D2D_KERNEL);
      cgh.parallel_for<BatchedScaledMemcpyInKernel<T, TS>>(
          sycl::nd_range<1>(num_workitems, max_group_size), task);
    });
  } else {
    stream->submit([&](sycl::handler& cgh) {
      BatchedScaledMemcpyOutKernel<T, TS> task(
          params, static_cast<T*>(fusion_buffer), scale_factor,
          GROUPS_PER_COPY_D2D_KERNEL);
      cgh.parallel_for<BatchedScaledMemcpyOutKernel<T, TS>>(
          sycl::nd_range<1>(num_workitems, max_group_size), task);
    });
  }
}

void BatchedScaledD2DMemcpyInImpl(BatchedD2DParams& params, void* fusion_buffer,
                                  int num_copies, double scale_factor,
                                  DataType dtype, gpuStream_t stream) {
  float float_scale_factor = (float)scale_factor;
  switch (dtype) {
  case HOROVOD_UINT8:
    BatchedScaledD2DMemcpy<uint8_t, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, true);
    break;
  case HOROVOD_INT8:
    BatchedScaledD2DMemcpy<int8_t, float>(params, fusion_buffer, num_copies,
                                           float_scale_factor, stream, true);
    break;
  case HOROVOD_INT32:
    BatchedScaledD2DMemcpy<int32_t, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, true);
    break;
  case HOROVOD_INT64:
    BatchedScaledD2DMemcpy<int64_t, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, true);
    break;
  case HOROVOD_FLOAT16:
    BatchedScaledD2DMemcpy<sycl::half, float>(
        params, fusion_buffer, num_copies, float_scale_factor, stream, true);
    break;
  case HOROVOD_BFLOAT16:
    BatchedScaledD2DMemcpy<bfloat16, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, true);
    break;
  case HOROVOD_FLOAT32:
    BatchedScaledD2DMemcpy<float, float>(params, fusion_buffer, num_copies,
                                         float_scale_factor, stream, true);
    break;
  case HOROVOD_FLOAT64:
    BatchedScaledD2DMemcpy<double, double>(params, fusion_buffer, num_copies,
                                          scale_factor, stream, true);
    break;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " not supported by BatchedScaledD2DMemcpyInImpl.");
  }
}

void BatchedScaledD2DMemcpyOutImpl(BatchedD2DParams& params,
                                   void* fusion_buffer, int num_copies,
                                   double scale_factor, DataType dtype,
                                   gpuStream_t stream) {
  float float_scale_factor = (float)scale_factor;
  switch (dtype) {
  case HOROVOD_UINT8:
    BatchedScaledD2DMemcpy<uint8_t, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, false);
    break;
  case HOROVOD_INT8:
    BatchedScaledD2DMemcpy<int8_t, float>(params, fusion_buffer, num_copies,
                                           float_scale_factor, stream, false);
    break;
  case HOROVOD_INT32:
    BatchedScaledD2DMemcpy<int32_t, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, false);
    break;
  case HOROVOD_INT64:
    BatchedScaledD2DMemcpy<int64_t, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, false);
    break;
  case HOROVOD_FLOAT16:
    BatchedScaledD2DMemcpy<sycl::half, float>(
        params, fusion_buffer, num_copies, float_scale_factor, stream, false);
    break;
  case HOROVOD_BFLOAT16:
    BatchedScaledD2DMemcpy<bfloat16, float>(params, fusion_buffer, num_copies,
                                            float_scale_factor, stream, false);
    break;
  case HOROVOD_FLOAT32:
    BatchedScaledD2DMemcpy<float, float>(params, fusion_buffer, num_copies,
                                         float_scale_factor, stream, false);
    break;
  case HOROVOD_FLOAT64:
    BatchedScaledD2DMemcpy<double, double>(params, fusion_buffer, num_copies,
                                          scale_factor, stream, false);
    break;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " not supported by BatchedScaledD2DMemcpyOutImpl.");
  }
}

bool enableBatchedScaledD2DMemcpy(
    HorovodGlobalState* global_state,
    const std::vector<TensorTableEntry>& entries) {
  bool batch_d2d_memcopies = global_state->batch_d2d_memcopies;
  if (batch_d2d_memcopies) {
    for (auto& e : entries) {
      auto tensor_elements =
          e.tensor->size() / DataType_Size(e.tensor->dtype());
      // Check entry buffer size as we use UINT32 to store offsets when using
      // batch memory copy.
      if (tensor_elements >= UNSIGNED_INT32_MAX) {
        LOG(WARNING)
            << " We use UINT32 to store offsets in BatchedD2DParams as kernel"
               " arguments restriction, so ignore batch memcpy when entry "
               "element"
               " size exceeds "
            << UNSIGNED_INT32_MAX;
        batch_d2d_memcopies = false;
        return batch_d2d_memcopies;
      }
    }
  }
  return batch_d2d_memcopies;
}
} // namespace common
} // namespace horovod
