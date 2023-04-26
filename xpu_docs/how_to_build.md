# Build from Source Code

This guide shows how to build an Intel® Optimization for Horovod* PyPI package from source and install it.



## Prepare

### Install GPU Driver
An Intel GPU driver is needed to build with GPU support. Refer to [Install Intel GPU Driver](../README.md#install-gpu-drivers).



### Install oneAPI Base Toolkit

Need to install components of Intel® oneAPI Base Toolkit:
- Intel® oneAPI DPC++ Compiler
- Intel® oneAPI Math Kernel Library (oneMKL)
- Intel® oneAPI Collective Communications Library (oneCCL)
- Intel® oneAPI MPI Library (IntelMPI)

```bash
$ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7deeaac4-f605-4bcf-a81b-ea7531577c61/l_BaseKit_p_2023.1.0.46401_offline.sh
# 3 components are necessary: DPC++/C++ Compiler with DPC++ Libiary, oneMKL and oneCCL(IntelMPI will be installed automatically as oneCCL's dependency).
$ sudo sh ./l_BaseKit_p_2023.1.0.46401_offline.sh
```

For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.
#### Setup environment variables
```bash
# DPC++ Compiler/oneMKL
source /path to basekit/intel/oneapi/compiler/latest/env/vars.sh
source /path to basekit/intel/oneapi/mkl/latest/env/vars.sh
# oneCCL (and Intel® oneAPI MPI Library as its dependency)
source /path to basekit/intel/oneapi/mpi/latest/env/vars.sh
source /path to basekit/intel/oneapi/ccl/latest/env/vars.sh
```




## Build

Intel® Optimization for Horovod* depends on TensorFlow* or/and Pytorch* to build from source.

```bash
$ pip install tensorflow==2.12.0

$ pip install torch==1.13.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
```



### Download source code

```bash
$ git clone https://github.com/intel/intel-optimization-for-horovod
$ cd intel-optimization-for-horovod
# The repo defaults to the `main` branch. You can also check out a release branch:
$ git checkout <branch_name>

$ git submodule init && git submodule update
```



### Build wheel and install

```bash
$ rm -rf build/

# You could build with single framework support by changing `HOROVOD_WITHOUT_PYTORCH=1` or `HOROVOD_WITHOUT_TENSORFLOW=1`
$ CC=icx CXX=icpx HOROVOD_GPU=DPCPP \
HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITHOUT_GLOO=1 HOROVOD_GPU_OPERATIONS=CCL HOROVOD_WITH_MPI=1 \
python setup.py bdist_wheel

$ pip install dist/*.whl
```



## Runtime

Intel® Optimization for Horovod* depends on Intel® Extension for Tensorflow* or Intel® Extension for Pytorch* at runtime to run on intel GPU.



- To run TensorFlow* applications

```bash
$ pip install --upgrade intel-extension-for-tensorflow[gpu]
```



- To run Pytorch* applications.
```
$ python -m pip install intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

You can then follow the steps in [Contributing guide](../CONTRIBUTING.md) to run some tests to confirm it works as expected.

