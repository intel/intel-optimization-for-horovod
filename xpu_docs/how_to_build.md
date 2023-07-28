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
$ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh
# 3 components are necessary: DPC++/C++ Compiler with DPC++ Libiary, oneMKL and oneCCL(IntelMPI will be installed automatically as oneCCL's dependency).
$ sudo sh ./l_BaseKit_p_2023.2.0.49397_offline.sh
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

Intel® Optimization for Horovod* depends on TensorFlow* to build from source.

```bash
$ pip install tensorflow==2.13.0

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

- set environment variables

```bash
$ source /path/to/intel/oneapi/compiler/latest/env/vars.sh
$ source /path/to/intel/oneapi/mkl/latest/env/vars.sh # for runtime only
$ source /path/to/intel/oneapi/mpi/latest/env/vars.sh
$ source /path/to/intel/oneapi/ccl/latest/env/vars.sh
```


#### Build from source

If you want to change code locally and build from source:

- Option 1: build and install from source code package

```bash
# will get a package located at dist/intel-optimization-for-horovod-*.tar.gz
$ python setup.py sdist

$ CC=icx CXX=icpx \
pip install --no-cache-dir intel-optimization-for-horovod -f dist/intel-optimization-for-horovod-*.tar.gz
```

- Option 2: build python wheels and install

```bash
$ CC=icx CXX=icpx \
python setup.py bdist_wheel

$ pip install dist/intel-optimization-for-horovod-*.whl
```


## Runtime

Intel® Optimization for Horovod* depends on Intel® Extension for Tensorflow* at runtime to run on intel GPU.



- To run TensorFlow* applications

```bash
$ pip install --upgrade intel-extension-for-tensorflow[xpu]
```


You can then follow the steps in [Contributing guide](../CONTRIBUTING.md) to run some tests to confirm it works as expected.

