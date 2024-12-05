message(STATUS "Build Horovod for SYCL")

set(CMAKE_CXX_STANDARD 17)

add_library(horovod_sycl_kernels sycl_kernels.cc)
target_compile_options(horovod_sycl_kernels PRIVATE ${SYCL_FLAGS})

add_library(compatible_horovod_sycl_kernels sycl_kernels.cc)
target_compile_options(compatible_horovod_sycl_kernels PRIVATE ${SYCL_FLAGS} -fpreview-breaking-changes -D_GLIBCXX_USE_CXX11_ABI=0)
