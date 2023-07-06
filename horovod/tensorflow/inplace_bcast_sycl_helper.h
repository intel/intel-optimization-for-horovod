// Copyright (C) 2023 Intel CORPORATION. All rights reserved.
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

#ifndef INPLACE_BROADCAST_SYCL_HELPER
#define INPLACE_BROADCAST_SYCL_HELPER

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {

struct XPUDevice {
  sycl::queue GetSYCLQueue() const {
    TF_Status* s = TF_NewStatus();
    auto sp_stream =
        TF_GetStream(reinterpret_cast<TF_OpKernelContext*>(ctx_), s);
    if (TF_GetCode(s) == TF_OK) {
      TF_DeleteStatus(s);
      return *(reinterpret_cast<SP_Stream_st*>(sp_stream)->stream_handle);
    } else {
      std::string err_msg = TF_Message(s);
      TF_DeleteStatus(s);
      throw std::runtime_error("Failed to get stream, error message: " +
                               err_msg);
    }
  }

  OpKernelContext* ctx_ = nullptr;
};

template <> const XPUDevice& OpKernelContext::eigen_device() const {
  static XPUDevice global_xpu_device;
  global_xpu_device.ctx_ = const_cast<OpKernelContext*>(this);
  return global_xpu_device;
}

namespace functor {
template <typename T> struct DenseUpdate<XPUDevice, T, ASSIGN> {
  void operator()(const XPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    if (params.size() != 0) {
      sycl::queue q = d.GetSYCLQueue();
      q.memcpy(params.data(), update.data(), params.size() * sizeof(T));
    }
  }
};

} // end namespace functor
} // end namespace tensorflow

typedef tensorflow::XPUDevice XPUDevice;

#endif // INPLACE_BROADCAST_SYCL_HELPER
