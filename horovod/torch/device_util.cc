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

#include <dlfcn.h>
#include <stdexcept>

#if HAVE_GPU
#if HAVE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#elif HAVE_SYCL
#include <c10/core/impl/VirtualGuardImpl.h>
#endif // HAVE_CUDA
#endif // HAVE_GPU

#include "device_util.h"

namespace horovod {
namespace torch {

#if HAVE_GPU && HAVE_CUDA
typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice)(CUdevice* device);
static void* cudalib = nullptr;
static PFN_cuCtxGetDevice pfn_cuCtxGetDevice = nullptr;

static void initialize_driver_api() {
  // Clear previous errors
  (void) dlerror();

  cudalib = dlopen("libcuda.so", RTLD_LAZY);
  if (!cudalib) {
    throw std::logic_error("Internal error. Could not dlopen libcuda.so.");
  }

  pfn_cuCtxGetDevice = (PFN_cuCtxGetDevice) dlsym(cudalib, "cuCtxGetDevice");
  if (!pfn_cuCtxGetDevice) {
    throw std::logic_error("Internal error. Could not load cuCtxGetDevice.");
  }
}
#endif

with_device::with_device(dev_index_t device) {
  if (device == CPU_DEVICE_ID) {
    restore_device_ = CPU_DEVICE_ID;
  } else {
#if HAVE_GPU
#if HAVE_CUDA
    if (!cudalib) initialize_driver_api();
    CUdevice cudev;
    auto err = pfn_cuCtxGetDevice(&cudev);
    if (err == CUDA_ERROR_NOT_INITIALIZED ||
        err == CUDA_ERROR_INVALID_CONTEXT) {
       // If device has never been set on this thread,
       // restore to supplied device.
       restore_device_ = device;
     } else if (err == CUDA_SUCCESS) {
       restore_device_ = static_cast<int>(cudev);
     } else {
       throw std::logic_error("Internal error. cuCtxGetDevice returned error code " +
                              std::to_string(err));
     }
    C10_CUDA_CHECK(cudaSetDevice(device));
#elif HAVE_SYCL
    c10::impl::VirtualGuardImpl impl(at::kXPU);
    restore_device_ = impl.getDevice().index();
    impl.setDevice(at::Device(at::kXPU, device));
#else
    throw std::logic_error(
        "Internal error. Requested device context manager "
        "with GPU device but not compiled with CUDA or SYCL.");
#endif
#endif
  }
}

with_device::~with_device() {
#if HAVE_GPU
#if HAVE_CUDA
  if (restore_device_ != CPU_DEVICE_ID) {
    C10_CUDA_CHECK(cudaSetDevice(restore_device_));
  }
#elif HAVE_SYCL
  if (restore_device_ != CPU_DEVICE_ID) {
    c10::impl::VirtualGuardImpl impl(at::kXPU);
    impl.setDevice(at::Device(at::kXPU, restore_device_));
  }
#endif
#endif
}

} // namespace torch
} // namespace horovod
