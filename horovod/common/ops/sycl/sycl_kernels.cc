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

template <typename T, typename TS> struct BatchedScaledMemcpySYCLKernel {
  BatchedScaledMemcpySYCLKernel(BatchedD2DParams& params, TS scale_factor,
                                int groups_per_copy)
      : params_(params), scale_factor_(scale_factor),
        groups_per_copy_(groups_per_copy) {}
  void operator()(sycl::nd_item<1> item) const {
    size_t local_id = item.get_local_id(0);
    size_t group_size = item.get_local_range(0);
    size_t group_id = item.get_group(0);

    const size_t idx = group_size * (group_id % groups_per_copy_) + local_id;
    int cur_index = group_id / groups_per_copy_;

    T* output = static_cast<T*>(params_.out[cur_index]);
    const T* input = static_cast<T*>(params_.in[cur_index]);
    size_t num_words = params_.sizes[cur_index] / sizeof(T);
    for (size_t i = idx; i < num_words; i += group_size * groups_per_copy_) {
      output[i] = input[i] * scale_factor_;
    }
  }

private:
  BatchedD2DParams params_;
  TS scale_factor_;
  int groups_per_copy_;
};

template <typename T> struct BatchedMemcpySYCLKernel {
  BatchedMemcpySYCLKernel(BatchedD2DParams& params, int groups_per_copy)
      : params_(params), groups_per_copy_(groups_per_copy) {}
  void operator()(sycl::nd_item<1> item) const {
    size_t local_id = item.get_local_id(0);
    size_t group_size = item.get_local_range(0);
    size_t group_id = item.get_group(0);

    const size_t idx = group_size * (group_id % groups_per_copy_) + local_id;
    int cur_index = group_id / groups_per_copy_;

    T* output = static_cast<T*>(params_.out[cur_index]);
    const T* input = static_cast<T*>(params_.in[cur_index]);
    size_t num_words = params_.sizes[cur_index] / sizeof(T);
    for (size_t i = idx; i < num_words; i += group_size * groups_per_copy_) {
      output[i] = input[i];
    }
  }

private:
  BatchedD2DParams params_;
  int groups_per_copy_;
};

#define GROUPS_PER_COPY_D2D_KERNEL 8
template <typename T, typename TS>
void BatchedScaledD2DMemcpy(BatchedD2DParams params, int num_copies,
                            TS scale_factor, gpuStream_t stream) {
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int32_t num_workitems =
      max_group_size * num_copies * GROUPS_PER_COPY_D2D_KERNEL;

  stream->submit([&](sycl::handler& cgh) {
    BatchedScaledMemcpySYCLKernel<T, TS> task(params, scale_factor,
                                              GROUPS_PER_COPY_D2D_KERNEL);
    cgh.parallel_for<BatchedScaledMemcpySYCLKernel<T, TS>>(
        sycl::nd_range<1>(num_workitems, max_group_size), task);
  });
}

void BatchedScaledD2DMemcpySYCLImpl(BatchedD2DParams& params, int num_copies,
                                    double scale_factor, DataType dtype,
                                    gpuStream_t stream) {
  float float_scale_factor = (float)scale_factor;
  switch (dtype) {
  case HOROVOD_UINT8:
    BatchedScaledD2DMemcpy<uint8_t, float>(params, num_copies,
                                           float_scale_factor, stream);
    break;
  case HOROVOD_INT8:
    BatchedScaledD2DMemcpy<int8_t, float>(params, num_copies,
                                          float_scale_factor, stream);
    break;
  case HOROVOD_INT32:
    BatchedScaledD2DMemcpy<int32_t, float>(params, num_copies,
                                           float_scale_factor, stream);
    break;
  case HOROVOD_INT64:
    BatchedScaledD2DMemcpy<int64_t, float>(params, num_copies,
                                           float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT16:
    BatchedScaledD2DMemcpy<sycl::half, float>(params, num_copies,
                                              float_scale_factor, stream);
    break;
  case HOROVOD_BFLOAT16:
    BatchedScaledD2DMemcpy<bfloat16, float>(params, num_copies,
                                            float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT32:
    BatchedScaledD2DMemcpy<float, float>(params, num_copies, float_scale_factor,
                                         stream);
    break;
  case HOROVOD_FLOAT64:
    BatchedScaledD2DMemcpy<double, double>(params, num_copies, scale_factor,
                                           stream);
    break;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " not supported by BatchedScaledD2DMemcpySYCLImpl.");
  }
}

void BatchedD2DMemcpySYCLImpl(BatchedD2DParams& params, int num_copies,
                              gpuStream_t stream) {
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int32_t num_workitems =
      max_group_size * num_copies * GROUPS_PER_COPY_D2D_KERNEL;

  stream->submit([&](sycl::handler& cgh) {
    BatchedMemcpySYCLKernel<unsigned char> task(params,
                                                GROUPS_PER_COPY_D2D_KERNEL);
    cgh.parallel_for<BatchedMemcpySYCLKernel<unsigned char>>(
        sycl::nd_range<1>(num_workitems, max_group_size), task);
  });
}
} // namespace common
} // namespace horovod
