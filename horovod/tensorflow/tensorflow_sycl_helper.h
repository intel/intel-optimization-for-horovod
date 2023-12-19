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

#ifndef TENSORFLOW_SYCL_HELPER
#define TENSORFLOW_SYCL_HELPER

#if TENSORFLOW_VERSION >= 2005000000
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
#else
#error "SYCL supports TensorFlow with PluggableDevice only."
#endif // TENSORFLOW_VERSION >= 2005000000

#if HAVE_TFNPD
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include <dlfcn.h>

static void LoadXpuLibrary() __attribute__((constructor));
static void UnloadXpuLibrary() __attribute__((destructor));

constexpr uintptr_t kTag = 0x1ULL;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Buffer PJRT_Buffer;
struct PjRtBuffer_Info {
  std::string datatype;
  std::vector<int64_t> dimensions;
  std::vector<int64_t> layout;
};

static void* libintel_xla_handle = nullptr;
static void* (*C_ITEXGetStreamFromPjRtDevice)(int device_id,
                                              PJRT_Client*) = nullptr;
static void* (*C_ITEXOpaqueDataPointerFromPjRtBuffer)(PJRT_Buffer*) = nullptr;
static PJRT_Buffer* (*C_ITEXCreatePjRtBuffer)(int device_id, PjRtBuffer_Info*,
                                              PJRT_Client*) = nullptr;

void LoadXpuLibrary() {
  libintel_xla_handle = dlopen("libintel_xla.so", RTLD_NOW | RTLD_LOCAL);
  if (!libintel_xla_handle) {
    const char* error_msg = dlerror();
    throw std::runtime_error(std::string(error_msg) +
                             ". Horovod.Tensorflow module built with "
                             "NextPluggableDevice for ITEX required package "
                             "intel_extension_for_tensorflow >= 2.15.0. Please "
                             "install proper version.");
  }

  C_ITEXGetStreamFromPjRtDevice =
      reinterpret_cast<decltype(C_ITEXGetStreamFromPjRtDevice)>(
          dlsym(libintel_xla_handle, "C_ITEXGetStreamFromPjRtDevice"));
  if (C_ITEXGetStreamFromPjRtDevice == nullptr) {
    const char* error_msg = dlerror();
    dlclose(libintel_xla_handle);
    throw std::runtime_error(error_msg);
  }

  C_ITEXOpaqueDataPointerFromPjRtBuffer =
      reinterpret_cast<decltype(C_ITEXOpaqueDataPointerFromPjRtBuffer)>(
          dlsym(libintel_xla_handle, "C_ITEXOpaqueDataPointerFromPjRtBuffer"));
  if (C_ITEXOpaqueDataPointerFromPjRtBuffer == nullptr) {
    const char* error_msg = dlerror();
    dlclose(libintel_xla_handle);
    throw std::runtime_error(error_msg);
  }

  C_ITEXCreatePjRtBuffer = reinterpret_cast<decltype(C_ITEXCreatePjRtBuffer)>(
      dlsym(libintel_xla_handle, "C_ITEXCreatePjRtBuffer"));
  if (C_ITEXCreatePjRtBuffer == nullptr) {
    const char* error_msg = dlerror();
    dlclose(libintel_xla_handle);
    throw std::runtime_error(error_msg);
  }
}

void UnloadXpuLibrary() {
  if (libintel_xla_handle) {
    dlclose(libintel_xla_handle);
  }
}

std::string GetEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::string();
  }
  return std::string(value);
}

bool UseTFNPD() {
  static bool isItexNPDEnabled_ =
      GetEnv("ITEX_ENABLE_NEXTPLUGGABLE_DEVICE") == "1" ? true : false;
  static bool isXlaAutoJitEnabled_ =
      static_cast<bool>(TF_GetXlaAutoJitEnabled());

  return isItexNPDEnabled_ || isXlaAutoJitEnabled_;
}

bool pointer_is_pjrt_tensor(TF_Tensor* tf_tensor) {
  uintptr_t value = reinterpret_cast<uintptr_t>(TF_TensorData(tf_tensor));
  if (value & kTag) {
    return true;
  } else {
    return false;
  }
}

void* tf_tensor_get_raw_data(TF_Tensor* tf_tensor) {
  if (!tf_tensor)
    throw std::runtime_error(
        "Horovod reads data from TF_Tensor* which is a nullptr.");
  void* data_ptr = TF_TensorData(tf_tensor);
  if (data_ptr == nullptr)
    return nullptr;
  uintptr_t value = reinterpret_cast<uintptr_t>(data_ptr);

  if (value & kTag) {
    TF_Status* tf_status = TF_NewStatus();
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tf_tensor, tf_status);
    return C_ITEXOpaqueDataPointerFromPjRtBuffer(pjrt_c_buffer);
  } else {
    return data_ptr;
  }
}

namespace tensorflow {

Status
npd_allocate_temp(OpKernelContext* context, DataType type,
                  const TensorShape& shape, Tensor* out_temp,
                  TF_Tensor** tf_tmp,
                  AllocatorAttributes alloc_attrs = AllocatorAttributes()) {
  TF_OpKernelContext* tf_ctx = reinterpret_cast<TF_OpKernelContext*>(context);
  TF_Status* status = TF_NewStatus();
  TF_AllocatorAttributes tf_alloc_attrs(
      {TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE, alloc_attrs.on_host()});
  TF_Tensor* tmp = TF_AllocateTemp(tf_ctx, static_cast<TF_DataType>(type),
                                   shape.dim_sizes().data(), shape.dims(),
                                   &tf_alloc_attrs, status);
  if (pointer_is_pjrt_tensor(tmp)) {
    int device_id = TF_GetDeviceId(tf_ctx);
    PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status);

    int rank = shape.dims();
    std::vector<int64_t> dimensions(rank);
    std::vector<int64_t> layout(rank);
    for (int d = 0; d < rank; ++d) {
      dimensions[d] = shape.dim_size(d);
    }
    std::iota(layout.rbegin(), layout.rend(), 0);

    PjRtBuffer_Info pjrt_buffer_info;
    pjrt_buffer_info.datatype = DataTypeString(type);
    pjrt_buffer_info.dimensions = dimensions;
    pjrt_buffer_info.layout = layout;

    TF_CreatePjRtBuffer(
        tmp,
        C_ITEXCreatePjRtBuffer(device_id, &pjrt_buffer_info, pjrt_c_client),
        "XPU", status);
  }

  Status cast_status = TF_TensorToTensor(tmp, out_temp);
  if (!cast_status.ok())
    throw cast_status;

  if (!tf_tmp)
    throw std::runtime_error("npd_allocate_temp requires tf_tmp != nullptr.");
  *tf_tmp = tmp;

  Status ret = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return ret;
}

Status npd_allocate_output(OpKernelContext* context, int index,
                           const TensorShape& shape, Tensor** tensor,
                           TF_Tensor** tf_tensor) {
  TF_OpKernelContext* tf_ctx = reinterpret_cast<TF_OpKernelContext*>(context);
  TF_Status* status = TF_NewStatus();
  DataType out_type =
      static_cast<DataType>(TF_ExpectedOutputDataType(tf_ctx, index));
  TF_Tensor* output =
      TF_AllocateOutput(tf_ctx, index, static_cast<TF_DataType>(out_type),
                        shape.dim_sizes().data(), shape.dims(),
                        shape.num_elements() * DataTypeSize(out_type), status);

  if (pointer_is_pjrt_tensor(output)) {
    int device_id = TF_GetDeviceId(tf_ctx);
    PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", status);
    int rank = shape.dims();
    std::vector<int64_t> dimensions(rank);
    std::vector<int64_t> layout(rank);
    for (int d = 0; d < rank; ++d) {
      dimensions[d] = shape.dim_size(d);
    }
    std::iota(layout.rbegin(), layout.rend(), 0);

    PjRtBuffer_Info pjrt_buffer_info;
    pjrt_buffer_info.datatype = DataTypeString(out_type);
    pjrt_buffer_info.dimensions = dimensions;
    pjrt_buffer_info.layout = layout;

    TF_CreatePjRtBuffer(
        output,
        C_ITEXCreatePjRtBuffer(device_id, &pjrt_buffer_info, pjrt_c_client),
        "XPU", status);
  }

  *tensor = context->mutable_output(index);
  if (!tf_tensor)
    throw std::runtime_error(
        "npd_allocate_output require tf_tensor != nullptr.");
  *tf_tensor = output;

  Status ret = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return ret;
}

} // namespace tensorflow
#endif // HAVE_TFNPD

struct SP_Stream_st {
  explicit SP_Stream_st(gpuStream_t stream_h) : stream_handle(stream_h) {}
  gpuStream_t stream_handle;
};

sycl::queue GetSYCLQueue(tensorflow::OpKernelContext* context) {
  TF_Status* s = TF_NewStatus();
#if HAVE_TFNPD
  if (UseTFNPD()) {
    int device_id =
        TF_GetDeviceId(reinterpret_cast<TF_OpKernelContext*>(context));
    PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", s);
    if (TF_GetCode(s) == TF_OK) {
      TF_DeleteStatus(s);
      sycl::queue q = *(static_cast<sycl::queue*>(
          C_ITEXGetStreamFromPjRtDevice(device_id, pjrt_c_client)));
      return q;
    } else {
      std::string err_msg = TF_Message(s);
      TF_DeleteStatus(s);
      throw std::runtime_error("Failed to get stream, error message: " +
                               err_msg);
    }
  }
#endif // HAVE_TFNPD
  auto sp_stream =
      TF_GetStream(reinterpret_cast<TF_OpKernelContext*>(context), s);
  if (TF_GetCode(s) == TF_OK) {
    TF_DeleteStatus(s);
    return *(reinterpret_cast<SP_Stream_st*>(sp_stream)->stream_handle);
  } else {
    std::string err_msg = TF_Message(s);
    TF_DeleteStatus(s);
    throw std::runtime_error("Failed to get stream, error message: " + err_msg);
  }
}

// below are helper functions for inplace_bcast
namespace tensorflow {

struct XPUDevice {
  sycl::queue GetSYCLQueue() const { return ::GetSYCLQueue(ctx_); }

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

} // namespace functor

#if HAVE_TFNPD
template <typename Device, typename T>
void NpdDenseAssignWrapper(TF_OpKernelContext* tf_ctx, TF_Tensor* tf_source,
                           TF_Tensor* tf_dest) {
  OpKernelContext* ctx = reinterpret_cast<OpKernelContext*>(tf_ctx);
  Tensor dest;
  Status cast_status = TF_TensorToTensor(tf_dest, &dest);
  if (!cast_status.ok())
    throw cast_status;

  const Tensor src;
  cast_status = TF_TensorToTensor(tf_source, const_cast<Tensor*>(&src));
  if (!cast_status.ok())
    throw cast_status;

  if (pointer_is_pjrt_tensor(tf_dest)) {
    TF_Status* tf_status = TF_NewStatus();
    PJRT_Buffer* pjrt_c_buffer = TF_GetPjRtCBuffer(tf_dest, tf_status);
    if (pjrt_c_buffer == nullptr) {
      int device_id = TF_GetDeviceId(tf_ctx);
      PJRT_Client* pjrt_c_client = TF_GetPjRtCClient("XPU", tf_status);

      int rank = dest.shape().dims();
      std::vector<int64_t> dimensions(rank);
      std::vector<int64_t> layout(rank);
      for (int d = 0; d < rank; ++d) {
        dimensions[d] = dest.shape().dim_size(d);
      }
      std::iota(layout.rbegin(), layout.rend(), 0);

      PjRtBuffer_Info pjrt_buffer_info;
      pjrt_buffer_info.datatype = DataTypeString(dest.dtype());
      pjrt_buffer_info.dimensions = dimensions;
      pjrt_buffer_info.layout = layout;

      TF_CreatePjRtBuffer(
          tf_dest,
          C_ITEXCreatePjRtBuffer(device_id, &pjrt_buffer_info, pjrt_c_client),
          "XPU", tf_status);

      TF_DeleteStatus(tf_status);
    }
  }

  if (dest.NumElements() != 0) {
    if constexpr (std::is_same<Device, XPUDevice>::value) {
      sycl::queue q = GetSYCLQueue(ctx);
      q.memcpy(tf_tensor_get_raw_data(tf_dest),
               tf_tensor_get_raw_data(tf_source),
               dest.NumElements() * sizeof(T));
    } else if constexpr (std::is_same<Device, CPUDevice>::value) {
      functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
      copy_functor(ctx->eigen_device<Device>(), dest.flat<T>(), src.flat<T>());
    } else {
      throw std::runtime_error("In Tensorflow SYCL support, DenseAssign only "
                               "support on CPUDevice and XPUDevice");
    }
  }
}

Status NpdGetInputTensorFromVariableHelper(
    OpKernelContext* ctx, int input, bool lock_held, bool sparse, Tensor* out,
    TF_Tensor** tf_out,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest)) {
  bool is_variant_type = false;
  TF_Status* tf_status = TF_NewStatus();
  TF_OpKernelContext* tf_ctx = reinterpret_cast<TF_OpKernelContext*>(ctx);
  TF_Tensor* tf_tensor = nullptr;

  // For ref tensor or dense tensor, the 3th, 4th, 5th arguments are actually
  // useless.
  TF_GetInputTensorFromVariable(tf_ctx, input, lock_held, is_variant_type,
                                sparse, copyFunc, &tf_tensor, tf_status);

  Status cast_status = TF_TensorToTensor(tf_tensor, out);
  if (!cast_status.ok())
    throw cast_status;

  if (!tf_out)
    throw std::runtime_error(
        "XPUGetInputTensorFromVariableHelper requires tf_out "
        "must not be nullptr");
  *tf_out = tf_tensor;

  Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}
#endif // HAVE_TFNPD

} // namespace tensorflow

typedef tensorflow::XPUDevice XPUDevice;

constexpr char* const DEVICE_XPU = "XPU";

#endif // TENSORFLOW_SYCL_HELPER
